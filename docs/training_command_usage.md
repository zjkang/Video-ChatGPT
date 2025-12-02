# 训练命令使用指南

## ⚠️ 常见错误

### 错误 1: 参数格式问题

**错误示例**：
```bash
python train_mem.py --data_path ...
--logging_steps 50 \          # ❌ 错误：这是单独的命令，不是参数
--logging_dir ./logs ...
```

**正确方式**：
```bash
python train_mem.py \
    --data_path ... \
    --logging_steps 50 \      # ✅ 正确：作为 python 命令的参数
    --logging_dir ./logs ...
```

### 错误 2: 多余的空格

**错误示例**：
```bash
python train_mem.py --data_path ... ' ' --logging_steps 50
# ❌ 错误：' ' 被当作参数
```

**正确方式**：
```bash
python train_mem.py --data_path ... --logging_steps 50
# ✅ 正确：没有多余空格
```

---

## ✅ 正确的使用方式

### 方式 1: 使用脚本文件（推荐）

```bash
# 1. 给脚本添加执行权限
chmod +x train_4090.sh

# 2. 运行脚本
./train_4090.sh
```

### 方式 2: 单行命令

```bash
python train_mem.py --data_path data/VideoInstruct-100K/VideoInstruct100K.json --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f --num_frames 16 --output_dir ./checkpoints/Baseline_T16_4090 --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 5e-5 --num_train_epochs 3 --warmup_ratio 0.03 --lr_scheduler_type cosine --weight_decay 0.0 --model_max_length 2048 --logging_steps 50 --logging_dir ./logs/baseline_4090 --report_to tensorboard --save_strategy steps --save_steps 3000 --save_total_limit 3 --gradient_checkpointing True --bf16 True
```

### 方式 3: 多行命令（使用反斜杠）

```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16_4090 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --model_max_length 2048 \
    --logging_steps 50 \
    --logging_dir ./logs/baseline_4090 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --bf16 True
```

**注意**：
- 每行末尾的反斜杠 `\` 表示命令继续
- 反斜杠后面不能有空格
- 最后一行不需要反斜杠

---

## 🔧 参数说明

### 字符串参数（不需要引号）

以下参数是字符串，但**不需要引号**：
- `--lr_scheduler_type cosine` ✅（不是 `"cosine"`）
- `--save_strategy steps` ✅（不是 `"steps"`）
- `--report_to tensorboard` ✅（不是 `"tensorboard"`）

### 布尔参数

- `--gradient_checkpointing True` ✅
- `--bf16 True` ✅

### 数值参数

- `--learning_rate 5e-5` ✅
- `--weight_decay 0.0` ✅（不是 `0.`）
- `--num_train_epochs 3` ✅

---

## 📝 完整训练命令（RTX 4090）

```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16_4090 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --model_max_length 2048 \
    --logging_steps 50 \
    --logging_dir ./logs/baseline_4090 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --bf16 True
```

---

## 🚀 快速开始

1. **使用脚本（最简单）**：
```bash
chmod +x train_4090.sh
./train_4090.sh
```

2. **或者直接复制单行命令**：
```bash
# 复制 train_4090_oneline.sh 中的内容，直接粘贴运行
```

