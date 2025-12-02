# max_steps vs num_train_epochs 修复说明

## 🐛 问题描述

训练只运行了 10 步就停止了，而不是预期的 3 个 epochs。

**原因**：
- `max_steps` 的默认值被设置为 `10`（用于 dry run）
- HuggingFace Trainer **优先使用 `max_steps`**，如果 `max_steps > 0`，会忽略 `num_train_epochs`
- 即使设置了 `--num_train_epochs 3`，由于 `max_steps=10`，训练只运行 10 步

## ✅ 修复方案

### 1. 修改 `max_steps` 默认值

**之前**：
```python
max_steps: int = field(default=10, metadata={"help": "For dry run, only run 10 steps"})
```

**现在**：
```python
max_steps: int = field(default=-1, metadata={"help": "Maximum number of training steps. -1 means use num_train_epochs instead."})
```

### 2. 添加参数检查逻辑

在训练开始前检查 `max_steps` 和 `num_train_epochs` 的关系，并给出明确提示。

## 📋 使用方式

### 方式 1: 使用 epochs（推荐）

```bash
python train_mem.py \
    --num_train_epochs 3 \
    --max_steps -1 \          # ✅ 明确设置为 -1，使用 epochs
    ...
```

### 方式 2: 使用 max_steps

```bash
python train_mem.py \
    --max_steps 5000 \         # ✅ 使用 steps
    # 不设置 num_train_epochs
    ...
```

### 方式 3: 不设置 max_steps（默认 -1）

```bash
python train_mem.py \
    --num_train_epochs 3 \
    # max_steps 默认为 -1，会自动使用 epochs
    ...
```

## ⚠️ 重要提示

1. **HuggingFace Trainer 的行为**：
   - 如果 `max_steps > 0`，**优先使用 `max_steps`**，`num_train_epochs` 会被忽略
   - 如果 `max_steps = -1`，使用 `num_train_epochs`

2. **同时设置两者的后果**：
   ```bash
   --max_steps 10 --num_train_epochs 3
   # ❌ 只会训练 10 步，3 个 epochs 被忽略
   ```

3. **推荐做法**：
   - 使用 epochs：设置 `--max_steps -1` 或省略（默认 -1）
   - 使用 steps：只设置 `--max_steps`，不设置 `num_train_epochs`

## 🔍 训练步数计算

如果使用 `num_train_epochs`：

```
total_steps = (dataset_size / effective_batch_size) * num_train_epochs

其中：
- effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
- dataset_size = 训练样本数量
```

**示例**：
- 数据集：100K 样本
- Batch size：4 * 2 = 8
- Epochs：3
- **Total steps = (100000 / 8) * 3 = 37,500 steps**

## ✅ 验证修复

训练开始时会显示：

```
✅ 使用 num_train_epochs=3 进行训练
```

而不是：

```
✅ 使用 max_steps=10 进行训练
```

---

**修复完成！现在可以正常使用 `--num_train_epochs 3` 进行完整训练了。**

