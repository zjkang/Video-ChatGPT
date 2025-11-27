import os
import torch
import torch.nn as nn
import numpy as np
# 1. 【全局导入】必须放在最外面！
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, BitsAndBytesConfig
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM

def load_video(video_path, num_frames=100):
    # 2. 【干净的函数】没有任何 try-except，直接用全局变量
    # 这样 Python 就绝对不会把它误判为局部变量了
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    idx = np.linspace(0, total_frame_num - 1, num_frames)
    idx = [int(i) for i in idx]
    pixel_values = vr.get_batch(idx).asnumpy()
    return pixel_values

def _ensure_model_on_device(model):
    """
    确保模型（包括 PEFT 包装的模块）在正确的设备上，并设置正确的 dtype。
    
    Args:
        model: 模型对象（可能是 PeftModel 包装的）
    
    Returns:
        model: 移动后的模型
    """
    # 获取模型设备
    model_device = next(model.get_model().embed_tokens.parameters()).device
    # 移动整个模型到正确的设备（这会移动所有子模块，包括 PEFT 包装的模块）
    model = model.to(model_device)
    # 确保 mm_projector 的 dtype 是 float16
    if hasattr(model.get_model(), 'mm_projector'):
        mm_projector = model.get_model().mm_projector
        mm_projector = mm_projector.to(torch.float16)
        print(f"✅ mm_projector moved to {model_device} (dtype: float16)")
    return model

def initialize_model(model_name, projection_path=None):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 3. 注入特殊 Token (解决无输出问题)
    DEFAULT_VIDEO_TOKEN = "<video>"
    DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
    DEFAULT_VID_START_TOKEN = "<vid_start>"
    DEFAULT_VID_END_TOKEN = "<vid_end>"
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    if DEFAULT_VIDEO_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([DEFAULT_VIDEO_TOKEN], special_tokens=True)

    print("Loading LLM with 4-bit quantization (Speed Optimized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    model = VideoChatGPTLlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map='auto',
        use_cache=True
    )

    vision_tower_name = getattr(model.config, 'mm_vision_tower', "openai/clip-vit-large-patch14")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name, torch_dtype=torch.float16, use_safetensors=False)

    print(f"Resizing token embeddings to {len(tokenizer)}...")
    model.resize_token_embeddings(len(tokenizer))
    
    # 4. 注入 Config (解决 bool sum 报错)
    vid_patch_token_idx = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    print(f"Injecting vid_patch_token = {vid_patch_token_idx} into config")
    model.config.vid_patch_token = vid_patch_token_idx
    model.config.use_sim_mask = False
    
    if hasattr(model.get_model(), "vision_config"):
        model.get_model().vision_config.vid_patch_token = vid_patch_token_idx

    # 5. 初始化 mm_projector（如果不存在，在 baseline-2 分支应该是 Linear）
    if not hasattr(model.get_model(), 'mm_projector'):
        # 确保 vision_config 存在
        if not hasattr(model.get_model(), 'vision_config'):
            from video_chatgpt.model.video_chatgpt import VisionConfig
            model.get_model().vision_config = VisionConfig()
        # 初始化 mm_projector（baseline-2 分支使用 Linear）
        model.get_model().initialize_vision_modules()
        print("✅ mm_projector initialized (Linear in baseline-2 branch)")

    # 6. 加载 Adapter
    if projection_path:
        print(f"Loading adapter from {projection_path}")
        
        # 检查是目录（PEFT adapter）还是文件
        if os.path.isdir(projection_path):
            # PEFT adapter 目录格式
            print("📦 Detected PEFT adapter directory, loading with PeftModel...")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, projection_path)
                print("✅ PEFT adapter loaded successfully!")
                # 确保所有模块在正确的设备上
                model = _ensure_model_on_device(model)
            except Exception as e:
                print(f"❌ Error: Failed to load PEFT adapter: {e}")
                raise e
        
        elif os.path.isfile(projection_path):
            # 文件路径：可能是 adapter_model.bin 或旧格式的权重文件
            file_name = os.path.basename(projection_path)
            dir_name = os.path.dirname(projection_path) or "."
            
            # 检查是否是 adapter_model.bin（PEFT 格式）
            if file_name in ["adapter_model.bin", "adapter_model.safetensors"]:
                # 这是 PEFT adapter 文件，需要从父目录加载
                print(f"📦 Detected PEFT adapter file, loading from parent directory: {dir_name}")
                if dir_name and os.path.isdir(dir_name):
                    try:
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(model, dir_name)
                        print("✅ PEFT adapter loaded successfully!")
                        # 确保所有模块在正确的设备上
                        model = _ensure_model_on_device(model)
                    except Exception as e:
                        print(f"❌ Error: Failed to load PEFT adapter from directory: {e}")
                        raise e
                else:
                    print(f"❌ Error: Parent directory not found: {dir_name}")
                    print("   Please use directory path instead: --projection_path ./checkpoints/Baseline_Linear_Adapter")
                    raise ValueError(f"Parent directory not found: {dir_name}")
            else:
                # 旧格式：单个 .bin 文件（非 PEFT 格式）
                print("📄 Detected single weight file (old format), loading manually...")
                state_dict = torch.load(projection_path, map_location='cpu')
                
                # 检查是否是 PEFT LoRA 格式（键名包含 lora_A 或 lora_B）
                is_peft_format = any('lora_A' in k or 'lora_B' in k for k in state_dict.keys())
                
                if is_peft_format:
                    # 这是 PEFT adapter_model.bin，应该从父目录加载
                    print("⚠️ Warning: Detected PEFT LoRA format in file, loading from parent directory")
                    if dir_name and os.path.isdir(dir_name):
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(model, dir_name)
                        print("✅ PEFT adapter loaded from parent directory!")
                        # 确保所有模块在正确的设备上
                        model = _ensure_model_on_device(model)
                    else:
                        raise ValueError(f"Cannot load PEFT adapter: parent directory not found: {dir_name}")
                else:
                    # 旧格式：直接加载权重（mm_projector 应该是 Linear）
                    for k, v in state_dict.items():
                        if 'mm_projector' in k:
                            # 检查 mm_projector 的类型
                            if not hasattr(model.get_model().mm_projector, 'weight'):
                                print(f"⚠️ Warning: mm_projector is {type(model.get_model().mm_projector)}, not Linear")
                                print("   Skipping manual weight injection for non-Linear projector")
                                continue
                            if 'weight' in k:
                                print("✅ Found projector weight, injecting...")
                                model.get_model().mm_projector.weight.data = v.to(model.device).to(torch.float16)
                            elif 'bias' in k:
                                print("✅ Found projector bias, injecting...")
                                model.get_model().mm_projector.bias.data = v.to(model.device).to(torch.float16)
                    
                    print("🎉 Adapter weights injected manually!")

    model.config.mm_use_vid_start_end = True
    model.config.mm_vision_select_layer = -2
    model.config.use_cache = True
    
    print(f"Loading Vision Tower: {vision_tower_name}")
    vision_tower = CLIPVisionModel.from_pretrained(
        vision_tower_name, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=False 
    ).cuda()
    vision_tower = vision_tower.eval()
    model.get_model().vision_tower = vision_tower

    video_token_len = 100
    
    return model, vision_tower, tokenizer, image_processor, video_token_len
