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
    prepare_model_for_kbit_training,
    TaskType
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

# --- 2. 数据集加载器 (MiniVideoDataset) ---
class MiniVideoDataset(Dataset):
    def __init__(self, data_path, features_folder, tokenizer):
        self.data = json.load(open(data_path))
        self.features_folder = features_folder
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        video_id = item['video_id']
        
        # 1. 加载视频特征 (.pkl file)
        pkl_path = os.path.join(self.features_folder, f"{video_id}.pkl")
        # 必须初始化为 numpy 数组 (而不是 torch tensor)
        video_features = np.zeros((100, 1024), dtype=np.float32) 
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    video_features = pickle.load(f)
            except:
                print(f"Warning: Failed to load feature for {video_id}. Using zeros.")

        # 2. 处理对话文本 (修复 KeyError: 'conversations')
        q = item['q']
        a = item['a']
        
        # 构造 Prompt (修复 SyntaxError)
        prompt = f"Human: <video> {q}\nAssistant: {a}</s>"
        
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
        
        # 3. 返回数据 (修复 KeyError: 'images' 和 NumPy 转换问题)
        return dict(
            input_ids=input_ids,
            attention_mask=tokenized.attention_mask[0],
            labels=labels,
            # 关键修复: 将 NumPy 转换为 PyTorch Tensor 并使用 'images' 键
            images=torch.from_numpy(video_features).to(dtype=torch.float16) 
        )

# --- 3. 数据整理器 (Data Collator) ---
@dataclass
class DataCollatorForVideo:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([instance['input_ids'] for instance in instances])
        labels = torch.stack([instance['labels'] for instance in instances])
        attention_mask = torch.stack([instance['attention_mask'] for instance in instances])
        
        # 堆叠视频特征
        images = [instance['images'] for instance in instances]
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images) 

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images
        )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- A. 加载 Tokenizer ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["<video>", "<vid_patch>"], special_tokens=True)

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

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "mm_projector"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    
    model.resize_token_embeddings(len(tokenizer))
    
    # 注入 VideoConfig
    vid_patch_id = tokenizer.convert_tokens_to_ids(["<vid_patch>"])[0]
    model.config.vid_patch_token = vid_patch_id
    if hasattr(model.get_model(), "vision_config"):
        model.get_model().vision_config.vid_patch_token = vid_patch_id

    model.print_trainable_parameters() 

    # --- C. 准备数据 ---
    dataset = MiniVideoDataset(data_args.data_path, data_args.features_folder, tokenizer)
    collator = DataCollatorForVideo(tokenizer=tokenizer)

    # --- D. 启动训练 ---
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    print("🔥 Starting Training (Dry Run)...")
    trainer.train()
    
    print("✅ Training Finished! Saving Adapter...")
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
