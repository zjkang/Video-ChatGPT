# 训练配置对比分析

## 📊 原始论文/代码配置 vs 当前配置

### 原始 Video-ChatGPT 训练配置

**来源**：`docs/train_video_chatgpt.md` 和 `使用指南.md`

```bash
torchrun --nproc_per_node=8 --master_port 29001 video_chatgpt/train/train_mem.py \
    --model_name_or_path <LLaVA-Lightning-7B-v1-1> \
    --version v1 \
    --data_path <training_data.json> \
    --video_folder <features_folder> \
    --tune_mm_mlp_adapter True \
    --mm_use_vid_start_end \
    --bf16 True \
    --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
    --num_train_epochs 3 \                    # ✅ 3个epochs
    --per_device_train_batch_size 4 \        # ✅ batch size 4
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \        # ✅ 不累积
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \                   # ✅ 学习率 2e-5
    --weight_decay 0. \                       # ✅ 无权重衰减
    --warmup_ratio 0.03 \                    # ✅ 3% warmup
    --lr_scheduler_type "cosine" \           # ✅ Cosine学习率调度
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \                 # ✅ 最大长度 2048
    --gradient_checkpointing True \
    --lazy_preprocess True
```

**关键特点**：
- ✅ **学习率**：`2e-5` (较小，稳定训练)
- ✅ **训练轮数**：`3 epochs` (完整训练)
- ✅ **学习率调度**：`cosine` + `warmup_ratio 0.03`
- ✅ **Batch size**：`4 * 1 = 4` (每设备4，不累积)
- ✅ **模型长度**：`2048` tokens
- ✅ **训练方式**：只训练 `mm_projector` (adapter)

---

### 当前训练配置 (`train_mem.py`)

**默认配置**：
```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = "paged_adamw_32bit"
    max_steps: int = 10                    # ⚠️ 默认只有10步（测试用）
    learning_rate: float = 2e-4            # ⚠️ 2e-4 (比原始大10倍！)
    per_device_train_batch_size: int = 2    # ⚠️ batch size 2
    gradient_accumulation_steps: int = 4    # ⚠️ 累积4步
    gradient_checkpointing: bool = True
    # ⚠️ 缺少：warmup_ratio, lr_scheduler_type, weight_decay, model_max_length
```

**实际使用**（从之前的对话）：
```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \                 # ⚠️ 2e-4 (比原始大10倍)
    --max_steps 5000 \                      # ⚠️ 5000步（可能不够）
    --logging_steps 20 \
    --save_strategy "steps" \
    --save_steps 1000
```

**关键特点**：
- ⚠️ **学习率**：`2e-4` (比原始大10倍！)
- ⚠️ **训练步数**：`5000 steps` (可能不够，取决于数据集大小)
- ⚠️ **学习率调度**：默认 `linear`，没有 warmup
- ⚠️ **Batch size**：`2 * 4 = 8` (每设备2，累积4)
- ⚠️ **模型长度**：默认 `512` (比原始的2048小)
- ✅ **训练方式**：使用 QLoRA (4-bit量化 + LoRA)

---

## 🚨 关键差异对比表

| 参数 | 原始配置 | 当前配置 | 差异 | 影响 |
|------|---------|---------|------|------|
| **learning_rate** | `2e-5` | `2e-4` | **10倍** | ⚠️ 学习率过大，可能导致训练不稳定或过拟合 |
| **训练量** | `3 epochs` | `5000 steps` | 取决于数据集 | ⚠️ 如果数据集大，5000步可能不够 |
| **lr_scheduler** | `cosine` | `linear` (默认) | 不同 | ⚠️ Cosine更平滑，有助于收敛 |
| **warmup_ratio** | `0.03` | 无 | 缺少 | ⚠️ 没有warmup，训练初期可能不稳定 |
| **weight_decay** | `0.` | 无（默认0.01） | 不同 | ⚠️ 可能影响正则化 |
| **model_max_length** | `2048` | `512` (默认) | **4倍** | ⚠️ 序列长度限制，可能截断长对话 |
| **batch_size** | `4 * 1 = 4` | `2 * 4 = 8` | 不同 | ✅ 实际batch size更大，可能更好 |
| **训练方式** | 全精度 + adapter | QLoRA (4-bit) | 不同 | ⚠️ QLoRA可能影响性能 |

---

## 🔍 问题分析

### 问题 1: 学习率过大 (2e-4 vs 2e-5)

**影响**：
- 学习率过大可能导致：
  - 训练不稳定（loss震荡）
  - 难以收敛到最优解
  - 过拟合风险增加
  - 模型性能下降

**建议**：
- 使用原始学习率 `2e-5`
- 或者如果使用 QLoRA，可以稍微提高，但不要超过 `5e-5`

### 问题 2: 训练步数可能不够

**计算**：
- 如果数据集有 100K 样本
- Batch size = 2 * 4 = 8
- Steps per epoch = 100K / 8 = 12,500 steps
- 3 epochs = 37,500 steps
- **当前只有 5000 steps ≈ 0.4 epochs** ⚠️

**建议**：
- 至少训练 1-2 个完整 epochs
- 或者使用 `--num_train_epochs 3` 替代 `--max_steps`

### 问题 3: 缺少学习率调度和 Warmup

**影响**：
- 没有 warmup：训练初期学习率过大，可能破坏预训练权重
- Linear scheduler：学习率线性下降，不如 cosine 平滑

**建议**：
- 添加 `--warmup_ratio 0.03`
- 添加 `--lr_scheduler_type "cosine"`

### 问题 4: 模型最大长度太小 (512 vs 2048)

**影响**：
- 如果对话较长，可能被截断
- 影响模型学习长对话的能力

**建议**：
- 设置 `--model_max_length 2048`

---

## ✅ 推荐的训练配置

### 方案 1: 完全对齐原始配置（推荐）

```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16_OriginalConfig \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \                    # ✅ 原始学习率
    --num_train_epochs 3 \                    # ✅ 3个epochs
    --warmup_ratio 0.03 \                     # ✅ 添加warmup
    --lr_scheduler_type "cosine" \            # ✅ Cosine调度
    --weight_decay 0. \                       # ✅ 无权重衰减
    --model_max_length 2048 \                  # ✅ 最大长度2048
    --logging_steps 100 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True
```

### 方案 2: QLoRA 优化配置（如果显存有限）

```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16_QLoRA \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \                    # ⚠️ 稍微提高（QLoRA通常需要稍高学习率）
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --model_max_length 2048 \
    --logging_steps 100 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True
```

---

## 📈 训练监控建议

1. **观察 Loss 曲线**：
   - Loss 应该平滑下降
   - 如果震荡，说明学习率过大
   - 如果下降太慢，可能需要更多训练

2. **检查验证集性能**（如果有）：
   - 定期在验证集上测试
   - 防止过拟合

3. **保存多个 Checkpoint**：
   - 保存中间checkpoint，方便回退
   - 选择验证集上最好的模型

---

## 🎯 总结

**主要问题**：
1. ⚠️ **学习率过大**：`2e-4` vs `2e-5` (10倍差异)
2. ⚠️ **训练步数可能不够**：5000 steps vs 3 epochs
3. ⚠️ **缺少学习率调度**：没有 warmup 和 cosine scheduler
4. ⚠️ **模型长度限制**：512 vs 2048

**建议**：
- ✅ 使用 `--learning_rate 2e-5` (或 QLoRA 用 `5e-5`)
- ✅ 使用 `--num_train_epochs 3` 替代 `--max_steps`
- ✅ 添加 `--warmup_ratio 0.03` 和 `--lr_scheduler_type "cosine"`
- ✅ 设置 `--model_max_length 2048`

**为了可靠的生成结果，应该使用与原始论文相同的训练过程！**

