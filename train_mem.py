import os
import json
import pickle
import torch
import numpy as np # 确保 numpy 导入，用于数据处理
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM

# --- 1. 配置参数 (保留不变) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./checkpoints/Llama-2-7b-chat-hf")

@dataclass
class DataArguments:
    data_path: str = field(default="data/mini_dataset/mini_train.json", metadata={"help": "Path to the training data."})
    features_folder: str = field(default="data/mini_dataset/features", metadata={"help": "Path to video features."})
    num_frames: Optional[int] = field(default=None, metadata={"help": "Number of frames per video. If None, will use the actual frame count from feature files."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="paged_adamw_32bit")
    max_steps: int = field(default=10, metadata={"help": "For dry run, only run 10 steps"})
    logging_steps: int = field(default=1) # 增加日志输出频率，方便我们观察 Loss
    gradient_checkpointing: bool = field(default=True)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    save_strategy: str = field(default="no") # Dry Run 不保存
    output_dir: str = field(default="./outputs/baseline") # 输出目录
    remove_unused_columns: bool = field(default=False) # 防止 Trainer 移除 'video' 等中间键
    logging_dir: Optional[str] = field(default="./logs", metadata={"help": "TensorBoard log directory"})
    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"], metadata={"help": "Report to tensorboard"})

# --- 2. 数据集加载器 (MiniVideoDataset) ---
class MiniVideoDataset(Dataset):
    def __init__(self, data_path, features_folder, tokenizer, num_frames=None):
        self.data = json.load(open(data_path))
        self.features_folder = features_folder
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.num_frames = num_frames  # 如果为 None，则根据实际加载的特征文件动态确定

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        # 支持不同的 video_id 字段名
        video_id = item.get('video_id') or item.get('id') or item.get('video_id')
        if video_id is None:
            raise ValueError(f"Sample {i} has no video_id or id field")
        
        # 1. 加载视频特征 (.pkl file)
        pkl_path = os.path.join(self.features_folder, f"{video_id}.pkl")
        video_features = None
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    video_features = pickle.load(f)
                # 确保是 numpy 数组
                if not isinstance(video_features, np.ndarray):
                    video_features = np.array(video_features)
            except Exception as e:
                print(f"Warning: Failed to load feature for {video_id}: {e}. Using zeros.")
                video_features = None
        
        # 如果加载失败或文件不存在，使用默认值
        if video_features is None:
            default_frames = self.num_frames if self.num_frames is not None else 16
            video_features = np.zeros((default_frames, 1024), dtype=np.float32)

        # 2. 处理对话文本
        # 支持两种格式：
        # 格式1: {"video_id": "...", "q": "...", "a": "..."}
        # 格式2: {"video_id": "...", "conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}]}
        if 'q' in item and 'a' in item:
            q = item['q']
            a = item['a']
        elif 'conversations' in item:
            conv = item['conversations']
            humans = [c for c in conv if c.get("from") == "human"]
            assists = [c for c in conv if c.get("from") == "assistant"]
            if len(humans) > 0 and len(assists) > 0:
                q = humans[0]["value"]
                a = assists[0]["value"]
            else:
                # 如果没有找到对话，使用默认值
                q = "Please describe this video."
                a = "The video shows some generic content."
        else:
            # 如果两种格式都没有，使用默认值
            q = "Please describe this video."
            a = "The video shows some generic content."
        
        # 构造 Prompt - 添加帧级 <vid_patch> tokens（用空格隔开，确保 tokenizer 识别为多个 token）
        # 格式: "Human: <video> <vid_patch> <vid_patch> ... {q}\nAssistant: {a}</s>"
        num_video_tokens = video_features.shape[0]  # 使用特征长度（动态，可以是 16/32/100 等）
        video_tokens = " ".join(["<vid_patch>"] * num_video_tokens)
        prompt = f"Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        
        input_ids = tokenized.input_ids[0]
        labels = input_ids.clone()
        
        # Mask labels: 只在 Assistant 回答部分计算 loss
        # 找到 "Assistant:" 之后第一个空格的位置，之前的部分设为 -100（ignore）
        # 更可靠的方法：分别 tokenize prompt 的前后部分（使用相同的 add_special_tokens 设置）
        prompt_before_answer = f"Human: <video> {video_tokens} {q}\nAssistant:"
        prompt_answer = a + "</s>"
        
        # Tokenize 分别找位置（使用相同的 add_special_tokens=True 以匹配 input_ids）
        before_tokenized = self.tokenizer(
            prompt_before_answer,
            return_tensors="pt",
            add_special_tokens=True
        )
        before_ids = before_tokenized.input_ids[0].tolist()
        # 找到 Assistant: 之后的开始位置
        labels[:len(before_ids)] = -100  # Mask 掉 Assistant 之前的所有内容
        
        # 保留 padding tokens 也 mask 掉
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        labels[input_ids == pad_token_id] = -100
        
        # 3. 返回数据 - 使用 'video' 键作为中间键名（与原始训练代码保持一致）
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=tokenized.attention_mask[0],
            labels=labels,
        )
        
        # 添加视频特征，使用 'video' 键（中间键名，DataCollator 会转换为模型需要的键名）
        data_dict['video'] = torch.from_numpy(video_features).to(dtype=torch.float16)
        
        return data_dict

# --- 3. 数据整理器 (Data Collator) ---
@dataclass
class DataCollatorForVideo:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([instance['input_ids'] for instance in instances])
        labels = torch.stack([instance['labels'] for instance in instances])
        attention_mask = torch.stack([instance['attention_mask'] for instance in instances])
        
        # 构建基础 batch
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        # 处理视频特征：从 Dataset 的 'video' 键读取，转换为模型需要的 'video_spatio_temporal_features' 键
        if len(instances) > 0 and 'video' in instances[0]:
            features = [instance['video'] for instance in instances]
            # 确保所有特征都是 torch.Tensor
            features = [f if isinstance(f, torch.Tensor) else torch.tensor(f) for f in features]
            # 如果所有特征的形状相同，则堆叠；否则保持列表
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['video_spatio_temporal_features'] = torch.stack(features)
            else:
                batch['video_spatio_temporal_features'] = features
        
        return batch

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- A. 加载 Tokenizer ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    # 检查并添加特殊 tokens（避免重复添加）
    special_tokens = ["<video>", "<vid_patch>"]
    existing_tokens = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in special_tokens if t not in existing_tokens]
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        print(f"✅ Added special tokens: {tokens_to_add}")
    else:
        print(f"✅ Special tokens already exist: {special_tokens}")

    # --- B. 加载模型 (QLoRA 核心) ---
    print("🚀 Loading Model with 4-bit QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = VideoChatGPTLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 初始化视觉模块（使用官方方法，更完整）
    print("📦 Initializing vision modules...")
    
    # 首先确保 vision_config 存在（initialize_vision_modules 需要它）
    if not hasattr(model.get_model(), "vision_config"):
        from video_chatgpt.model.video_chatgpt import VisionConfig
        model.get_model().vision_config = VisionConfig()
        print("✅ vision_config created")
    
    # 使用官方方法初始化 mm_projector（这会设置所有必要的配置）
    # 这个方法会：
    # 1. 设置 config.use_mm_proj = True
    # 2. 设置 config.mm_hidden_size = vision_config.hidden_size (1024)
    # 3. 创建 mm_projector (1024 -> 4096)
    # 4. 计算 video_token_len
    if not hasattr(model.get_model(), "mm_projector"):
        model_vision_dict = model.get_model().initialize_vision_modules(
            pretrain_mm_mlp_adapter=None  # 如果需要预训练权重，可以指定路径
        )
        
        # 找到模型参数所在的设备（用于 device_map="auto"）
        # 获取第一个模型参数的设备位置
        model_device = next(model.get_model().embed_tokens.parameters()).device
        print(f"📱 Model device: {model_device}")
        
        # 将 mm_projector 移动到正确的设备和 dtype
        model.get_model().mm_projector = model.get_model().mm_projector.to(model_device).to(torch.float16)
        
        print(f"✅ Vision modules initialized:")
        print(f"   - mm_projector: {model_vision_dict['vision_config'].hidden_size} -> {model.config.hidden_size}")
        print(f"   - mm_projector type: Linear (Baseline)")
        print(f"   - mm_projector device: {model.get_model().mm_projector.weight.device}")
        print(f"   - video_token_len: {model_vision_dict['video_token_len']}")
    else:
        print("✅ mm_projector already exists")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # 仅在 LLM 上做 LoRA，mm_projector 全量训练
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    
    # 确保 PEFT 包装后，mm_projector 仍然在正确的设备上
    # 检查 mm_projector 是否被正确包装，如果设备不对则修复
    if hasattr(model.get_model(), 'mm_projector'):
        try:
            # 验证设备 - 获取 mm_projector 的设备
            projector_params = list(model.get_model().mm_projector.parameters())
            if len(projector_params) > 0:
                projector_device = projector_params[0].device
                # 获取模型其他部分的设备作为参考
                model_device = next(model.get_model().embed_tokens.parameters()).device
                
                if projector_device != model_device:
                    print(f"⚠️ Warning: mm_projector device ({projector_device}) != model device ({model_device})")
                    print(f"   正在移动 mm_projector 到 {model_device}...")
                    # 检查 mm_projector 是否被 PEFT 包装
                    # 如果被包装，直接移动整个模块（PEFT 会保持引用）
                    # 如果未被包装，直接移动
                    mm_projector = model.get_model().mm_projector
                    mm_projector = mm_projector.to(model_device).to(torch.float16)
                    # 如果被 PEFT 包装，需要更新引用
                    if hasattr(mm_projector, 'base_layer'):
                        # PEFT 包装的情况：更新 base_layer
                        print(f"   mm_projector 被 PEFT 包装，更新 base_layer...")
                    model.get_model().mm_projector = mm_projector
                    print(f"✅ mm_projector 已移动到 {model_device} (dtype: float16)")
                else:
                    print(f"✅ PEFT 包装后，mm_projector device: {projector_device} (正确)")
        except Exception as e:
            print(f"⚠️ Warning: 无法检查 mm_projector 设备: {e}")
    
    model.resize_token_embeddings(len(tokenizer))
    
    # 注入 VideoConfig
    vid_patch_id = tokenizer.convert_tokens_to_ids(["<vid_patch>"])[0]
    model.config.vid_patch_token = vid_patch_id
    if hasattr(model.get_model(), "vision_config"):
        model.get_model().vision_config.vid_patch_token = vid_patch_id

    model.print_trainable_parameters() 

    # --- C. 准备数据 ---
    dataset = MiniVideoDataset(
        data_args.data_path, 
        data_args.features_folder, 
        tokenizer,
        num_frames=data_args.num_frames
    )
    collator = DataCollatorForVideo(tokenizer=tokenizer)

    # --- D. 启动训练 ---
    # 确保 report_to 是列表格式
    if training_args.report_to is None:
        training_args.report_to = ["tensorboard"]
    elif isinstance(training_args.report_to, str):
        training_args.report_to = [training_args.report_to]
    
    # 确保 logging_dir 存在
    if training_args.logging_dir:
        os.makedirs(training_args.logging_dir, exist_ok=True)
    
    print(f"📊 TensorBoard 配置:")
    print(f"   - logging_dir: {training_args.logging_dir}")
    print(f"   - report_to: {training_args.report_to}")
    print(f"   - logging_steps: {training_args.logging_steps}")
    
    # 验证 TensorBoard 是否可用
    try:
        from transformers.integrations import TensorBoardCallback
        print(f"   ✅ TensorBoard callback 可用")
    except ImportError:
        print(f"   ⚠️  TensorBoard callback 不可用，尝试安装: pip install tensorboard")
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    
    # 检查 Trainer 的 callbacks
    print(f"📋 Trainer callbacks: {[type(cb).__name__ for cb in trainer.callback_handler.callbacks]}")

    print("🔥 Starting Training (Dry Run)...")
    trainer.train()
    
    # 保存模型（如果设置了输出目录）
    if hasattr(training_args, 'output_dir') and training_args.output_dir:
        print("✅ Training Finished! Saving Adapter...")
        os.makedirs(training_args.output_dir, exist_ok=True)
        model.save_pretrained(training_args.output_dir)
        print(f"✅ Model saved to {training_args.output_dir}")
    else:
        print("✅ Training Finished! (No output directory specified, skipping save)")

if __name__ == "__main__":
    train()

# Day 3：训练您的创新模块 T-LoRA-VLLM
# python train_mem.py \
#     --output_dir ./checkpoints/T_LoRA_Temporal_Adapter \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 2e-4 \
#     --max_steps 500 \
#     --logging_steps 20 \
#     --save_strategy "steps" \
#     --save_steps 250 \
#     --save_total_limit 2
