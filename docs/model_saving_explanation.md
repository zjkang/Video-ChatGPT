# 模型保存机制详解

## 📍 保存代码位置

**文件**: `train_mem.py`

**代码位置**: 第 303-310 行

```python
# 保存模型（如果设置了输出目录）
if hasattr(training_args, 'output_dir') and training_args.output_dir:
    print("✅ Training Finished! Saving Adapter...")
    os.makedirs(training_args.output_dir, exist_ok=True)
    model.save_pretrained(training_args.output_dir)
    print(f"✅ Model saved to {training_args.output_dir}")
else:
    print("✅ Training Finished! (No output directory specified, skipping save)")
```

---

## 🔑 关键点：PEFT 模型保存

### 1. 保存的是什么？

由于使用了 **LoRA (PEFT)**，`model.save_pretrained()` 只会保存：

- ✅ **LoRA Adapter 权重**（可训练的参数）
- ✅ **模型配置**（config.json）
- ✅ **Tokenizer**（如果传入了 tokenizer）
- ❌ **不保存基础模型权重**（Llama-2 的权重不保存，因为使用的是预训练模型）

### 2. 保存的文件结构

```
outputs/baseline/
├── adapter_config.json          # LoRA 配置
├── adapter_model.bin            # LoRA adapter 权重（只有可训练的参数）
├── adapter_model.safetensors    # 安全格式的权重（如果支持）
├── training_args.bin            # 训练参数
└── README.md                    # PEFT 说明文件
```

**重要**: 保存的文件很小（通常只有几 MB 到几十 MB），因为只保存了 LoRA 参数，而不是整个 7B 模型。

---

## 📊 详细保存流程

### 步骤 1: 检查输出目录

```python
# train_mem.py:304
if hasattr(training_args, 'output_dir') and training_args.output_dir:
```

**默认输出目录**: `./outputs/baseline`（在 `TrainingArguments` 中定义）

### 步骤 2: 创建目录

```python
# train_mem.py:306
os.makedirs(training_args.output_dir, exist_ok=True)
```

如果目录不存在，会自动创建。

### 步骤 3: 保存模型

```python
# train_mem.py:307
model.save_pretrained(training_args.output_dir)
```

**PEFT 的 `save_pretrained()` 会**:
1. 提取所有可训练的 LoRA 参数
2. 保存到 `adapter_model.bin` 或 `adapter_model.safetensors`
3. 保存 LoRA 配置到 `adapter_config.json`
4. 生成 `README.md` 说明文件

---

## 📁 保存的文件详解

### 1. `adapter_config.json`

```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "inference_mode": false,
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj", "mm_projector"],
  "lora_dropout": 0.05,
  "base_model_name_or_path": "./checkpoints/Llama-2-7b-chat-hf",
  "bias": "none"
}
```

**作用**: 记录 LoRA 配置，加载时需要这个文件。

### 2. `adapter_model.bin` 或 `adapter_model.safetensors`

**内容**: 只包含可训练的参数

```python
{
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": tensor(...),
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": tensor(...),
    "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": tensor(...),
    "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": tensor(...),
    ...
    "base_model.model.model.mm_projector.lora_A.default.weight": tensor(...),
    "base_model.model.model.mm_projector.lora_B.default.weight": tensor(...),
}
```

**大小**: 通常只有几 MB 到几十 MB（取决于 LoRA rank 和 target_modules）

### 3. `training_args.bin`

保存训练参数（如学习率、batch size 等），用于恢复训练或参考。

---

## 🔄 如何加载保存的模型

### 方法 1: 使用 PEFT 加载

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM

# 1. 加载基础模型
base_model = VideoChatGPTLlamaForCausalLM.from_pretrained(
    "./checkpoints/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. 加载 LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./outputs/baseline"  # adapter 保存路径
)

# 3. 合并 adapter（可选，用于推理）
model = model.merge_and_unload()
```

### 方法 2: 直接使用 PEFT 的 from_pretrained

```python
from peft import PeftModel

# 会自动加载 adapter_config.json 中指定的 base_model
model = PeftModel.from_pretrained(
    base_model,
    "./outputs/baseline"
)
```

---

## 💡 关键理解点

### 1. 为什么只保存 Adapter？

- **基础模型权重很大**（7B 模型约 14GB）
- **LoRA adapter 很小**（通常只有几 MB）
- **基础模型不变**，只需要保存 adapter 即可

### 2. 保存的 Adapter 包含什么？

- ✅ `q_proj` 的 LoRA 参数（所有层的）
- ✅ `v_proj` 的 LoRA 参数（所有层的）
- ✅ `mm_projector` 的 LoRA 参数（如果被 LoRA 包装）
- ❌ 不包含基础模型的权重

### 3. 如果 mm_projector 是 TemporalTransformer？

如果 `mm_projector` 是 `TemporalTransformer`（不是简单的 Linear），PEFT 会：
- 尝试对 `TemporalTransformer` 内部的 `output_proj` 层应用 LoRA
- 或者对整个 `TemporalTransformer` 应用 LoRA（取决于配置）

---

## 📈 保存大小估算

### LoRA 参数计算

假设：
- `r = 8`（LoRA rank）
- `target_modules = ["q_proj", "v_proj", "mm_projector"]`
- Llama-2-7B: 32 层，hidden_size = 4096

**参数量**:
```
q_proj: 32 layers × (4096 × 8 + 8 × 4096) = 32 × 65536 = 2,097,152
v_proj: 32 layers × (4096 × 8 + 8 × 4096) = 32 × 65536 = 2,097,152
mm_projector: (1024 × 8 + 8 × 4096) = 40960
总计: ~4.2M 参数
```

**文件大小**: 约 16-20 MB（FP16 格式）

---

## 🔍 检查保存的文件

训练结束后，可以检查保存的文件：

```bash
# 查看保存目录
ls -lh outputs/baseline/

# 查看 adapter 配置
cat outputs/baseline/adapter_config.json

# 查看文件大小
du -sh outputs/baseline/
```

---

## ⚠️ 注意事项

### 1. 需要基础模型才能加载

保存的 adapter 不能单独使用，必须配合基础模型（Llama-2-7B）加载。

### 2. 保存路径配置

确保 `TrainingArguments` 中设置了 `output_dir`:

```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./outputs/baseline")
```

### 3. 保存策略

当前代码在训练结束后保存一次。如果需要中间 checkpoint，可以设置：

```python
save_strategy: str = field(default="steps")  # 改为 "steps"
save_steps: int = field(default=500)         # 每 500 步保存一次
```

---

## 📝 总结

**保存流程**:
1. 训练结束后，检查 `output_dir` 是否设置
2. 创建输出目录
3. 调用 `model.save_pretrained()` 保存 LoRA adapter
4. 只保存可训练的参数（几 MB），不保存基础模型（14GB）

**保存内容**:
- ✅ LoRA adapter 权重
- ✅ LoRA 配置
- ✅ 训练参数
- ❌ 基础模型权重（不保存）

**加载方式**:
- 需要先加载基础模型
- 然后加载 adapter
- 可以合并 adapter 到基础模型（用于推理）

---

**希望这个解释帮助你理解模型保存机制！** 🚀

