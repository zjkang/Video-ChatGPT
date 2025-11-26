import torch
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

    # 5. 注入 Adapter (解决 Unexpected Keys)
    if projection_path:
        print(f"Loading weights from {projection_path}")
        state_dict = torch.load(projection_path, map_location='cpu')
        
        if not hasattr(model.get_model(), 'mm_projector'):
            from video_chatgpt.model.temporal_transformer import TemporalTransformer
            model.get_model().mm_projector = TemporalTransformer(
                input_dim=1024,
                output_dim=4096,
                num_layers=2
            ).to(model.device)

        for k, v in state_dict.items():
            if 'mm_projector' in k:
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
