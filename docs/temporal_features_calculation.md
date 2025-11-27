# Temporal Features 计算方式详解

## 🔍 当前实现分析

### 1. 训练时（`extract_features.py`）

```python
# 步骤1: 提取CLIP特征
outputs = vision_tower(video_process, output_hidden_states=True)
features = outputs.last_hidden_state  # [100, 257, 1024]
# 注意：使用最后一层（last_hidden_state）

# 步骤2: 去掉CLS token
features = features[:, 1:]  # [100, 256, 1024]
# 去掉第一个CLS token，保留256个空间patch

# 步骤3: 计算temporal features（对空间维度平均）
temporal_features = features.mean(dim=1)  # [100, 1024]
# 对每个时间帧的256个空间patch做平均，得到该时间帧的全局特征
```

**关键点**：
- 使用 `last_hidden_state`（最后一层）
- 输入形状：`[100, 257, 1024]` → `[100, 256, 1024]` → `[100, 1024]`
- 输出：`(100, 1024)` - 100个时间帧，每个帧一个1024维特征向量

---

### 2. 推理时（`run_cli.py` + `get_spatio_temporal_features_torch()`）

```python
# 步骤1: 提取CLIP特征
image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [t, 256, 1024]
# 注意：使用倒数第二层（hidden_states[-2]），这是LLaVA的做法

# 步骤2: 计算temporal features（在get_spatio_temporal_features_torch中）
temporal_tokens = torch.mean(frame_features, dim=1)  # [t, 1024]
# 对空间维度平均，得到temporal tokens

# 步骤3: Padding到100
padding_size = 100 - t
if padding_size > 0:
    padding = torch.zeros(padding_size, c, device=features.device)
    temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)  # [100, 1024]
```

**关键点**：
- 使用 `hidden_states[-2]`（倒数第二层）
- 输入形状：`[t, 256, 1024]` → `[t, 1024]` → `[100, 1024]`（padding后）
- 输出：`(100, 1024)` - 但计算方式略有不同

---

## ⚠️ 发现的不一致问题

### 问题1：隐藏层选择不同

| 阶段 | 使用的层 | 代码位置 |
|------|---------|---------|
| 训练 | `last_hidden_state`（最后一层） | `extract_features.py:60` |
| 推理 | `hidden_states[-2]`（倒数第二层） | `run_cli.py:38`, `inference.py:89` |

**影响**：
- 不同层的特征表示可能不同
- 训练和推理看到的特征分布不一致

**LLaVA的做法**：
- LLaVA使用倒数第二层（`hidden_states[-2]`）
- 原因：倒数第二层通常包含更丰富的视觉信息，最后一层可能过度抽象

---

### 问题2：特征处理流程

**训练时**：
```python
features = outputs.last_hidden_state  # [100, 257, 1024]
features = features[:, 1:]            # [100, 256, 1024]
temporal_features = features.mean(dim=1)  # [100, 1024]
```

**推理时**：
```python
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [t, 256, 1024]
temporal_tokens = torch.mean(frame_features, dim=1)  # [t, 1024]
# 然后padding到100
```

**相同点**：
- 都去掉CLS token（`[:, 1:]`）
- 都对空间维度平均（`mean(dim=1)`）

**不同点**：
- 训练时：直接得到100帧
- 推理时：可能少于100帧，需要padding

---

## ✅ 正确的 Temporal Features 计算方式

### 统一的计算流程

```python
def calculate_temporal_features(frame_features, target_length=100):
    """
    计算temporal features（与训练时一致）
    
    Args:
        frame_features: [t, 256, 1024] - t个时间帧，每个帧256个空间patch，1024维特征
        target_length: 目标长度（默认100，对应100帧）
    
    Returns:
        [target_length, 1024] - temporal features
    """
    # 步骤1: 对空间维度平均（每个时间帧的256个patch平均成一个向量）
    temporal_features = torch.mean(frame_features, dim=1)  # [t, 1024]
    # 含义：每个时间帧的全局空间特征
    
    # 步骤2: Padding到目标长度
    t = temporal_features.shape[0]
    if t < target_length:
        padding_size = target_length - t
        padding = torch.zeros(padding_size, 1024, device=temporal_features.device)
        temporal_features = torch.cat([temporal_features, padding], dim=0)  # [100, 1024]
    elif t > target_length:
        # 如果超过100帧，截断（或使用插值）
        temporal_features = temporal_features[:target_length]  # [100, 1024]
    
    return temporal_features
```

### 关键步骤说明

**步骤1：空间维度平均**
```python
temporal_features = torch.mean(frame_features, dim=1)
```
- 输入：`[t, 256, 1024]` - t个时间帧，每个帧256个空间patch
- 操作：对每个时间帧的256个空间patch做平均
- 输出：`[t, 1024]` - 每个时间帧一个全局特征向量
- **含义**：每个temporal token代表一个时间帧的全局空间信息

**步骤2：Padding**
```python
padding_size = 100 - t
if padding_size > 0:
    padding = torch.zeros(padding_size, 1024, ...)
    temporal_features = torch.cat([temporal_features, padding], dim=0)
```
- 目的：确保所有视频都是100个tokens
- 如果视频少于100帧，用零向量padding
- 如果视频超过100帧，截断或插值

---

## 🔧 需要统一的地方

### 1. 隐藏层选择

**选项A：统一使用倒数第二层（推荐，与LLaVA一致）**
```python
# 训练时（extract_features.py）
features = outputs.hidden_states[-2]  # 改为倒数第二层
features = features[:, 1:]  # [100, 256, 1024]
temporal_features = features.mean(dim=1)  # [100, 1024]
```

**选项B：统一使用最后一层**
```python
# 推理时（run_cli.py）
frame_features = image_forward_outs.last_hidden_state[:, 1:]  # 改为最后一层
```

**推荐选项A**：
- 与LLaVA一致
- 倒数第二层通常包含更丰富的视觉信息

### 2. 特征处理流程

**统一流程**：
```python
# 1. 提取特征（使用倒数第二层）
features = vision_tower_outputs.hidden_states[-2]  # [t, 257, 1024]

# 2. 去掉CLS token
features = features[:, 1:]  # [t, 256, 1024]

# 3. 计算temporal features（对空间维度平均）
temporal_features = features.mean(dim=1)  # [t, 1024]

# 4. Padding到100
if temporal_features.shape[0] < 100:
    padding = torch.zeros(100 - temporal_features.shape[0], 1024, ...)
    temporal_features = torch.cat([temporal_features, padding], dim=0)  # [100, 1024]
```

---

## 📝 总结

### Temporal Features 的计算公式

```
Temporal Features = Mean(Spatial Patches) for each Time Frame

具体步骤：
1. 输入: [t, 256, 1024] - t个时间帧，每个帧256个空间patch
2. 对空间维度平均: mean(dim=1) → [t, 1024]
3. Padding: 如果 t < 100，padding到100 → [100, 1024]
4. 输出: [100, 1024] - 100个temporal tokens
```

### 关键理解

1. **Temporal token的含义**：
   - 每个temporal token代表一个时间帧的**全局空间特征**
   - 通过平均该时间帧的所有空间patch得到
   - 包含了该时间帧的整体视觉信息

2. **为什么对空间维度平均**：
   - 将256个空间patch的信息压缩成一个向量
   - 保留时间序列信息，简化空间信息
   - 与训练时的处理方式一致

3. **Padding的作用**：
   - 确保所有视频都是100个tokens
   - 便于batch处理和模型输入

---

## 🎯 建议

1. **统一隐藏层选择**：使用 `hidden_states[-2]`（倒数第二层）
2. **统一特征处理流程**：都使用 `mean(dim=1)` 对空间维度平均
3. **创建统一函数**：`get_temporal_features_torch()` 用于训练和推理

