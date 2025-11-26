# Video-ChatGPT 训练代码逻辑详解

## 📋 目录
1. [整体架构](#整体架构)
2. [数据流程](#数据流程)
3. [模型架构](#模型架构)
4. [训练流程](#训练流程)
5. [关键技术点](#关键技术点)

---

## 🏗️ 整体架构

这是一个 **Video-LLM (视频大语言模型)** 的训练系统，核心特点：

- **基础模型**: Llama-2-7B-Chat
- **量化方式**: 4-bit QLoRA (降低显存占用)
- **微调方式**: LoRA (只训练少量参数)
- **多模态融合**: 视频特征 + 文本对话

```
┌─────────────────────────────────────────────────────────┐
│                   训练系统架构                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  视频特征     │      │  对话文本     │                │
│  │ (100, 1024)  │      │  (Q, A)      │                │
│  └──────┬───────┘      └──────┬───────┘                │
│         │                     │                         │
│         └──────────┬──────────┘                         │
│                    │                                    │
│         ┌──────────▼──────────┐                        │
│         │   MiniVideoDataset   │                        │
│         │   (数据预处理)        │                        │
│         └──────────┬──────────┘                        │
│                    │                                    │
│         ┌──────────▼──────────┐                        │
│         │ DataCollatorForVideo │                        │
│         │   (Batch整理)        │                        │
│         └──────────┬──────────┘                        │
│                    │                                    │
│         ┌──────────▼──────────┐                        │
│         │ VideoChatGPTLlamaModel│                       │
│         │  + mm_projector      │                        │
│         │  + LoRA Adapters    │                        │
│         └──────────┬──────────┘                        │
│                    │                                    │
│         ┌──────────▼──────────┐                        │
│         │   HuggingFace Trainer│                        │
│         │   (训练循环)          │                        │
│         └─────────────────────┘                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 数据流程

### 1. 数据格式

**输入数据** (`data/mini_dataset/mini_train.json`):
```json
{
  "video_id": "xxx",
  "q": "What is happening in this video?",
  "a": "A person is walking..."
}
```

**视频特征** (`data/mini_dataset/features/xxx.pkl`):
- 形状: `(100, 1024)` - 100帧，每帧1024维特征
- 格式: NumPy array (float32)

### 2. Dataset 处理流程 (`MiniVideoDataset`)

```python
# 步骤 1: 加载视频特征
video_features = pickle.load(f"{video_id}.pkl")  # (100, 1024)

# 步骤 2: 构造 Prompt
prompt = f"Human: <video> {'<vid_patch>' * 100} {q}\nAssistant: {a}</s>"
# 示例: "Human: <video> <vid_patch><vid_patch>...<vid_patch> What is happening?\nAssistant: A person is walking...</s>"

# 步骤 3: Tokenize
input_ids = tokenizer(prompt, ...)  # [1, 512]

# 步骤 4: Labels Masking (关键!)
labels = input_ids.clone()
# 只计算 Assistant 回答部分的 loss
# 将 "Human: ... Assistant:" 之前的部分设为 -100 (忽略)
labels[:len(before_answer_ids)] = -100

# 步骤 5: 返回
return {
    'input_ids': input_ids,
    'labels': labels,
    'attention_mask': attention_mask,
    'video': video_features  # (100, 1024) -> torch.float16
}
```

**关键点**:
- `labels` masking: 只在 Assistant 回答部分计算 loss，避免学习重复问题
- 100 个 `<vid_patch>` tokens: 对应 100 帧视频特征

### 3. DataCollator 处理 (`DataCollatorForVideo`)

```python
# 将多个样本整理成 batch
batch = {
    'input_ids': [sample1, sample2, ...],      # (B, 512)
    'labels': [label1, label2, ...],            # (B, 512)
    'attention_mask': [mask1, mask2, ...],      # (B, 512)
    'video_spatio_temporal_features': [...]    # (B, 100, 1024) - 关键!
}
```

**关键转换**:
- Dataset 返回 `'video'` 键
- DataCollator 转换为 `'video_spatio_temporal_features'` 键（模型期望的键名）

---

## 🧠 模型架构

### 1. 整体结构

```
VideoChatGPTLlamaForCausalLM
├── VideoChatGPTLlamaModel (LlamaModel)
│   ├── embed_tokens          # 文本嵌入层
│   ├── layers (LlamaDecoderLayer x 32)
│   │   ├── self_attn (LoRA)
│   │   └── mlp
│   └── norm
├── mm_projector (LoRA)      # 视频特征投影层 (1024 -> 4096)
└── lm_head                   # 语言模型头
```

### 2. 前向传播流程 (`VideoChatGPTLlamaModel.forward`)

```python
# 输入
input_ids: (B, 512)                    # 文本 tokens
video_spatio_temporal_features: (B, 100, 1024)  # 视频特征

# 步骤 1: 文本嵌入
inputs_embeds = embed_tokens(input_ids)  # (B, 512, 4096)

# 步骤 2: 视频特征投影
video_features = mm_projector(video_spatio_temporal_features)  # (B, 100, 4096)
# 关键: 1024 -> 4096，匹配 LLM 的 hidden_size

# 步骤 3: 融合视频特征到文本嵌入
# 找到 input_ids 中 <vid_patch> tokens 的位置
# 用 video_features 替换对应的 embeddings
for each sample in batch:
    # 找到 <vid_patch> tokens 的位置
    vid_patch_indices = where(input_ids == vid_patch_token_id)
    
    # 替换: inputs_embeds[vid_patch_indices] = video_features
    new_input_embeds = concat([
        inputs_embeds[:vid_patch_start],
        video_features,                    # 插入视频特征
        inputs_embeds[vid_patch_end:]
    ])

# 步骤 4: 通过 LLM
outputs = llama_model(inputs_embeds=new_input_embeds)

# 步骤 5: 计算 loss (只在 labels != -100 的位置)
logits = lm_head(outputs.last_hidden_state)
loss = CrossEntropyLoss(logits, labels)
```

**关键点**:
- **视频特征投影**: `mm_projector` 将视频特征 (1024维) 投影到 LLM 的隐藏维度 (4096维)
- **特征融合**: 将视频特征嵌入到文本序列中，替换 `<vid_patch>` tokens
- **dtype 一致性**: 确保所有张量都是 FP16，避免类型不匹配

### 3. mm_projector 的作用

```
视频特征 (100, 1024) 
    ↓
mm_projector (Linear: 1024 -> 4096)
    ↓
视频嵌入 (100, 4096)  ← 与文本嵌入维度一致
    ↓
替换 <vid_patch> tokens 的位置
    ↓
输入到 LLM
```

---

## 🚀 训练流程

### 1. 初始化阶段 (`train()` 函数)

```python
# A. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-chat-hf")
tokenizer.add_tokens(["<video>", "<vid_patch>"])  # 添加特殊 tokens

# B. 加载模型 (4-bit 量化)
model = VideoChatGPTLlamaForCausalLM.from_pretrained(
    "Llama-2-7b-chat-hf",
    quantization_config=bnb_config,  # 4-bit QLoRA
    device_map="auto"
)

# C. 初始化视觉模块
model.get_model().initialize_vision_modules()
# 创建 mm_projector: Linear(1024, 4096)

# D. 配置 LoRA
config = LoraConfig(
    r=8,                                    # LoRA rank
    target_modules=["q_proj", "v_proj", "mm_projector"],  # 只训练这些模块
    ...
)
model = get_peft_model(model, config)

# E. 准备数据
dataset = MiniVideoDataset(...)
collator = DataCollatorForVideo(...)

# F. 启动训练
trainer = Trainer(model=model, train_dataset=dataset, ...)
trainer.train()
```

### 2. 训练循环 (HuggingFace Trainer)

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播
        outputs = model(
            input_ids=batch['input_ids'],
            video_spatio_temporal_features=batch['video_spatio_temporal_features'],
            labels=batch['labels']
        )
        
        # 2. 计算 loss
        loss = outputs.loss
        
        # 3. 反向传播 (只更新 LoRA 参数)
        loss.backward()
        
        # 4. 优化器更新
        optimizer.step()
```

**关键点**:
- **只训练 LoRA 参数**: 大部分模型参数冻结，只更新 LoRA adapters
- **梯度累积**: `gradient_accumulation_steps=4`，模拟更大的 batch size
- **4-bit 量化**: 大幅降低显存占用，可以在单卡上训练

---

## 🔑 关键技术点

### 1. QLoRA (4-bit 量化 + LoRA)

**目的**: 降低显存占用，实现单卡训练

```python
# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用 FP16
    bnb_4bit_quant_type="nf4"       # NF4 量化类型
)
```

**效果**:
- 原始 Llama-2-7B: ~14GB 显存
- 4-bit 量化后: ~4-5GB 显存
- 可以单卡训练！

### 2. LoRA (Low-Rank Adaptation)

**目的**: 只训练少量参数，降低训练成本

```python
# LoRA 配置
LoraConfig(
    r=8,  # rank，控制参数量
    target_modules=["q_proj", "v_proj", "mm_projector"]
)
```

**原理**:
```
原始权重 W (4096, 4096)
    ↓
LoRA: W + ΔW = W + B @ A
    ↓
只训练 A (r, 4096) 和 B (4096, r)
参数量: 4096 * 4096 → 4096 * r * 2 (r=8 时约 65K 参数)
```

**可训练参数**:
- `q_proj`, `v_proj`: LLM 的注意力层 (每个约 65K 参数)
- `mm_projector`: 视频投影层 (约 4M 参数，但用 LoRA 后大幅减少)
- 总计: 约 1-2% 的原始参数量

### 3. Labels Masking

**目的**: 只在 Assistant 回答部分计算 loss

```python
# Prompt: "Human: <video> ... {q}\nAssistant: {a}</s>"
# Labels: [-100, -100, ..., token1, token2, ...]
#          ↑ 忽略部分    ↑ 计算 loss 的部分
```

**为什么重要**:
- 避免模型学习重复问题
- 只学习如何生成回答
- 提高训练效率

### 4. dtype 一致性处理

**问题**: 4-bit 量化 + FP16 混合精度训练时，容易出现 dtype 不匹配

**解决方案**:
```python
# 确保所有张量都是 FP16
video_features = mm_projector(video_spatio_temporal_features)
video_features = video_features.to(torch.float16)  # 强制转换

inputs_embeds = torch.stack(new_input_embeds, dim=0)
if inputs_embeds.dtype != torch.float16:
    inputs_embeds = inputs_embeds.to(torch.float16)
```

### 5. 设备管理

**问题**: `device_map="auto"` 可能将模型分布到多个设备

**解决方案**:
```python
# 确保 mm_projector 在正确的设备上
model_device = next(model.embed_tokens.parameters()).device
mm_projector = mm_projector.to(model_device).to(torch.float16)
```

---

## 📈 训练配置说明

### 关键超参数

```python
TrainingArguments(
    per_device_train_batch_size=2,      # 每个设备的 batch size
    gradient_accumulation_steps=4,      # 梯度累积 (实际 batch = 2 * 4 = 8)
    learning_rate=2e-4,                 # 学习率
    max_steps=10,                       # 最大步数 (测试用)
    gradient_checkpointing=True,        # 梯度检查点 (节省显存)
    optim="paged_adamw_32bit",         # 优化器 (内存优化版)
    remove_unused_columns=False,        # 保留所有列 (包括 'video')
)
```

### 显存优化策略

1. **4-bit 量化**: 模型权重压缩到 4-bit
2. **Gradient Checkpointing**: 用时间换空间
3. **LoRA**: 只训练少量参数
4. **Gradient Accumulation**: 模拟大 batch，但显存占用小

---

## 🎯 总结

### 核心设计思想

1. **高效训练**: 4-bit QLoRA + LoRA，单卡可训练
2. **多模态融合**: 视频特征通过 `mm_projector` 投影后融入文本序列
3. **精确 Loss**: Labels masking 确保只学习回答部分
4. **类型安全**: 严格的 dtype 和设备管理

### 数据流总结

```
视频特征 (100, 1024) + 对话文本 (Q, A)
    ↓
Dataset: 构造 Prompt，Tokenize，Labels Masking
    ↓
DataCollator: 整理 Batch，转换键名
    ↓
Model: mm_projector 投影 → 融合到文本嵌入 → LLM 前向传播
    ↓
Loss: 只在 Assistant 回答部分计算
    ↓
反向传播: 只更新 LoRA 参数
```

---

## 🔍 关键文件说明

- `train_mem.py`: 主训练脚本
- `video_chatgpt/model/video_chatgpt.py`: 模型定义
- `MiniVideoDataset`: 数据集加载器
- `DataCollatorForVideo`: 数据整理器

---

**希望这个文档帮助你理解代码逻辑！如有疑问，欢迎提问。** 🚀

