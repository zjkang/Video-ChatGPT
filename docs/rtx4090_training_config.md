# RTX 4090 (24GB) 训练配置建议

## 🎯 硬件情况

- **GPU**: RTX 4090
- **显存**: 24GB
- **单卡训练**

## 📊 显存需求分析

### 全精度训练（原始方法）

**显存需求估算**：
- Llama-2-7B (BF16): ~14GB
- 视频特征 batch: ~2-4GB
- 梯度 + 优化器状态: ~14GB
- **总计**: ~30-32GB

**结论**: ❌ **24GB 不够全精度训练**

### QLoRA 训练（4-bit量化）

**显存需求估算**：
- Llama-2-7B (4-bit): ~4-5GB
- LoRA 参数: ~0.5GB
- 视频特征 batch: ~2-4GB
- 梯度 + 优化器状态: ~2-3GB
- **总计**: ~8-12GB

**结论**: ✅ **24GB 足够 QLoRA 训练，还有余量**

---

## ✅ 推荐方案：QLoRA + 优化配置

### 训练配置（针对 4090 优化）

```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16_4090 \
    --per_device_train_batch_size 4 \        # ✅ 4090 可以支持更大的 batch
    --gradient_accumulation_steps 2 \        # ✅ 实际 batch = 4 * 2 = 8
    --learning_rate 5e-5 \                  # ✅ QLoRA 使用稍高学习率
    --num_train_epochs 3 \                   # ✅ 完整训练 3 个 epochs
    --warmup_ratio 0.03 \                    # ✅ 添加 warmup
    --lr_scheduler_type "cosine" \           # ✅ Cosine 学习率调度
    --weight_decay 0. \                      # ✅ 无权重衰减
    --model_max_length 2048 \                # ✅ 最大长度 2048
    --logging_steps 100 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --bf16 True                              # ✅ 使用 BF16（如果支持）
```

### 关键优化点

1. **Batch Size**: 
   - `per_device_train_batch_size=4`（4090 可以支持）
   - `gradient_accumulation_steps=2`（实际 batch = 8）

2. **学习率**:
   - `5e-5`（QLoRA 可以稍高，但不要超过这个值）

3. **训练轮数**:
   - `num_train_epochs=3`（确保充分训练）

4. **学习率调度**:
   - `warmup_ratio=0.03` + `lr_scheduler_type="cosine"`

5. **精度**:
   - 使用 `bf16=True`（如果 4090 支持，比 FP16 更稳定）

---

## 🔧 如果显存仍然不足

### 方案 A: 减小 Batch Size

```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \    # 实际 batch = 2 * 4 = 8
```

### 方案 B: 启用更多优化

```bash
--gradient_checkpointing True \       # 已启用
--dataloader_num_workers 0 \         # 减少 CPU 内存占用
--dataloader_pin_memory False        # 减少内存占用
```

### 方案 C: 使用更激进的量化

如果仍然不够，可以考虑：
- 使用 8-bit 量化（而不是 4-bit）
- 或者进一步减小 batch size

---

## 📈 训练监控

### 显存监控

训练时监控显存使用：
```bash
watch -n 1 nvidia-smi
```

**预期显存使用**：
- 正常情况：~10-15GB
- 如果超过 20GB，考虑减小 batch size

### Loss 监控

使用 TensorBoard：
```bash
tensorboard --logdir ./logs/baseline --port 6006
```

**预期 Loss 曲线**：
- 应该平滑下降
- 如果震荡，说明学习率过大
- 如果下降太慢，可能需要更多训练

---

## 🎯 完整训练命令示例

```bash
# 1. 激活环境
conda activate video_chatgpt

# 2. 启动训练
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
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --model_max_length 2048 \
    --logging_steps 100 \
    --logging_dir ./logs/baseline_4090 \
    --report_to tensorboard \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --bf16 True

# 3. 监控训练（另一个终端）
tensorboard --logdir ./logs/baseline_4090 --port 6006
```

---

## ⚠️ 重要提醒

### 1. 不要使用过大的学习率

**错误**：
```bash
--learning_rate 2e-4  # ❌ 太大！
```

**正确**：
```bash
--learning_rate 5e-5  # ✅ QLoRA 推荐
```

### 2. 确保训练充分

**错误**：
```bash
--max_steps 5000  # ❌ 可能不够
```

**正确**：
```bash
--num_train_epochs 3  # ✅ 完整训练
```

### 3. 使用正确的学习率调度

**错误**：
```bash
# 缺少 warmup 和 cosine scheduler
```

**正确**：
```bash
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine"
```

---

## 📊 预期训练时间

**估算**（基于 100K 样本，batch=8）：
- Steps per epoch: 100K / 8 = 12,500 steps
- 3 epochs: 37,500 steps
- 每步时间: ~0.5-1秒（取决于数据加载）
- **总时间**: ~5-10小时

---

## ✅ 总结

**对于 RTX 4090 (24GB)**：

1. ✅ **使用 QLoRA**（4-bit量化 + LoRA）
2. ✅ **Batch size**: 4 * 2 = 8
3. ✅ **学习率**: `5e-5`
4. ✅ **训练轮数**: `3 epochs`
5. ✅ **学习率调度**: warmup + cosine
6. ✅ **模型长度**: `2048`

**关键点**：
- 4090 足够 QLoRA 训练
- 使用正确的超参数（不要用 `2e-4`！）
- 确保训练充分（3 epochs）
- 监控显存和 loss 曲线

