# 当前代码问题总结

## 🔴 核心问题：训练与推理不一致

### 问题 1: 特征维度不匹配

**训练时**：
- 训练数据（`.pkl` 文件）：`(100, 1024)` - 只包含 temporal tokens
- Prompt 中：100 个 `<vid_patch>` tokens
- 代码位置：`train_mem.py` 第 61, 75 行

**推理时**：
- 实际生成的特征：`(356, 1024)` - 包含 100 temporal + 256 spatial tokens
- Prompt 中：356 个 `<vid_patch>` tokens（已修复）
- 代码位置：`get_spatio_temporal_features_torch()` 函数

**影响**：
- 模型在训练时学习的是 `(100, 1024)` 特征格式
- 但在推理时看到的是 `(356, 1024)` 特征格式
- 这会导致模型性能下降，因为从未见过这种特征格式

---

## 📊 详细分析

### 1. 训练数据生成流程

**方式 A：`extract_features.py`（当前使用）**
```python
# 提取特征：[100, 257, 1024]
features = outputs.last_hidden_state[:, 1:]  # [100, 256, 1024]
features = features.mean(dim=1)  # [100, 1024] - 对空间维度平均
```
- 输出：`(100, 1024)` - 只有 temporal tokens
- 每个时间帧的空间特征被平均成一个向量

**方式 B：`scripts/save_spatio_temporal_clip_features.py`（官方脚本）**
```python
# 提取特征：[t, 256, 1024]
temporal_tokens = np.mean(features, axis=1)  # [t, 1024] - 对空间维度平均
spatial_tokens = np.mean(features, axis=0)   # [256, 1024] - 对时间维度平均
sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)  # [356, 1024]
```
- 输出：`(356, 1024)` - temporal + spatial tokens
- 包含更丰富的空间信息

### 2. 推理流程

**`get_spatio_temporal_features_torch()`**
```python
# 输入：[t, 256, 1024]
temporal_tokens = torch.mean(features, dim=1)  # [100, 1024]
spatial_tokens = torch.mean(features, dim=0)   # [256, 1024]
concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0)  # [356, 1024]
```
- 输出：`(356, 1024)` - 与方式 B 一致

---

## ⚠️ 问题根源

1. **训练数据格式**：使用 `extract_features.py` 生成 `(100, 1024)` 特征
2. **推理代码**：使用 `get_spatio_temporal_features_torch()` 生成 `(356, 1024)` 特征
3. **不一致**：训练和推理的特征格式完全不同

---

## 🔧 解决方案

### 方案 1：统一为 (100, 1024) 格式（简单，但丢失空间信息）

**修改推理代码**：
```python
# 在 run_cli.py 中，修改特征提取
# 只使用 temporal tokens，丢弃 spatial tokens
temporal_tokens = torch.mean(frame_features, dim=1)  # [100, 1024]
video_spatio_temporal_features = temporal_tokens  # 只使用 temporal
```

**优点**：
- 与训练数据一致
- 修改简单

**缺点**：
- 丢失空间信息（256 个 spatial tokens）
- 可能影响模型性能

### 方案 2：统一为 (356, 1024) 格式（推荐，保留完整信息）

**修改训练代码**：
1. 修改 `train_mem.py` 第 61 行：
   ```python
   video_features = np.zeros((356, 1024), dtype=np.float32)  # 改为 356
   ```

2. 修改 `train_mem.py` 第 75 行：
   ```python
   num_video_tokens = 356  # 改为 356
   ```

3. 重新生成训练数据：
   - 使用 `scripts/save_spatio_temporal_clip_features.py` 生成 `(356, 1024)` 特征
   - 或者修改 `extract_features.py` 使用 spatio-temporal 格式

**优点**：
- 保留完整的空间和时间信息
- 与推理代码一致
- 可能提升模型性能

**缺点**：
- 需要重新生成训练数据
- 需要重新训练模型
- Token 数量增加，可能影响序列长度限制

### 方案 3：检查实际训练数据格式

**先确认**：
- 检查现有的 `.pkl` 文件实际形状
- 如果已经是 `(356, 1024)`，只需修改训练代码
- 如果是 `(100, 1024)`，需要选择方案 1 或 2

---

## 📝 当前状态

✅ **已修复**：
- `run_cli.py` 中根据实际特征数量动态设置 token 数量（356）

❌ **待修复**：
- 训练代码与推理代码的特征格式不一致
- 需要统一训练和推理的特征格式

---

## 🎯 建议

1. **立即检查**：确认训练数据（`.pkl` 文件）的实际形状
2. **选择方案**：根据数据格式选择方案 1 或 2
3. **统一格式**：确保训练和推理使用相同的特征格式
4. **重新训练**：如果选择方案 2，需要重新训练模型

