# 推理代码只使用 Temporal Tokens 的讨论

## 🎯 核心观点

如果训练时只使用 **temporal tokens (100个)**，那么推理时也应该只使用 **temporal tokens**，保持训练和推理的一致性。

---

## 📊 当前情况

### 训练时（`train_mem.py` + `extract_features.py`）

```python
# extract_features.py
features = outputs.last_hidden_state[:, 1:]  # [100, 256, 1024]
features = features.mean(dim=1)  # [100, 1024] - 只对空间维度平均
# 输出：只有 temporal tokens，形状 (100, 1024)
```

### 推理时（`run_cli.py` + `get_spatio_temporal_features_torch()`）

```python
# get_spatio_temporal_features_torch()
temporal_tokens = torch.mean(features, dim=1)  # [100, 1024]
spatial_tokens = torch.mean(features, dim=0)   # [256, 1024]
concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0)  # [356, 1024]
# 输出：temporal + spatial tokens，形状 (356, 1024)
```

**问题**：训练和推理不一致！

---

## ✅ 解决方案：推理时也只使用 Temporal Tokens

### 修改方案

**方案 1：修改 `get_spatio_temporal_features_torch()` 函数**

```python
def get_spatio_temporal_features_torch(features, use_spatial=False):
    """
    Computes spatio-temporal features from given features.
    
    Args:
        features: [t, s, c] - 时间帧数 t，空间patch数 s，特征维度 c
        use_spatial: 是否使用 spatial tokens（默认 False，只使用 temporal）
    
    Returns:
        [t, c] 或 [t+s, c] - 根据 use_spatial 决定
    """
    t, s, c = features.shape
    
    # Temporal tokens: 对空间维度平均
    temporal_tokens = torch.mean(features, dim=1)  # [t, c]
    
    # Padding to 100
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    
    if use_spatial:
        # Spatial tokens: 对时间维度平均
        spatial_tokens = torch.mean(features, dim=0)  # [s, c]
        # Concatenate
        concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()
        return concat_tokens
    else:
        # 只返回 temporal tokens
        return temporal_tokens.half()
```

**方案 2：创建新函数 `get_temporal_features_torch()`**

```python
def get_temporal_features_torch(features):
    """
    只提取 temporal tokens（与训练时一致）
    
    Args:
        features: [t, s, c] - 时间帧数 t，空间patch数 s，特征维度 c
    
    Returns:
        [100, c] - temporal tokens，padding到100
    """
    t, s, c = features.shape
    
    # Temporal tokens: 对空间维度平均
    temporal_tokens = torch.mean(features, dim=1)  # [t, c]
    
    # Padding to 100
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    
    return temporal_tokens.half()
```

**方案 3：在 `run_cli.py` 中直接处理**

```python
# 在 run_cli.py 中
with torch.no_grad():
    image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
    frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [t, 256, 1024]

# 只使用 temporal tokens（与训练时一致）
temporal_tokens = torch.mean(frame_features, dim=1)  # [t, 1024]

# Padding to 100
padding_size = 100 - temporal_tokens.shape[0]
if padding_size > 0:
    padding = torch.zeros(padding_size, 1024, device=temporal_tokens.device)
    temporal_tokens = torch.cat([temporal_tokens, padding], dim=0)

video_spatio_temporal_features = temporal_tokens.half()  # [100, 1024]
```

---

## 🔍 需要修改的文件

### 1. `run_cli.py`
- 当前：使用 `get_spatio_temporal_features_torch()` 生成 (356, 1024)
- 修改：只生成 temporal tokens (100, 1024)

### 2. `video_chatgpt/inference.py`
- 当前：`get_spatio_temporal_features_torch()` 返回 (356, 1024)
- 修改：添加参数或新函数，只返回 (100, 1024)

### 3. `video_chatgpt/single_video_inference.py`
- 同样需要修改

### 4. `video_chatgpt/demo/chat.py`
- 同样需要修改

---

## ✅ 优点

1. **训练推理一致**：
   - 训练：100个tokens
   - 推理：100个tokens
   - 模型看到的数据格式一致

2. **简单直接**：
   - 不需要处理spatial tokens
   - 代码更简单

3. **与 TemporalTransformer 一致**：
   - 如果使用TemporalTransformer，输入是100个tokens
   - 推理时也是100个tokens，完全匹配

4. **计算效率**：
   - 处理100个tokens vs 356个tokens
   - 更少的计算量

---

## ⚠️ 注意事项

1. **Prompt 中的 token 数量**：
   - 需要修改为100个 `<vid_patch>` tokens
   - 之前我们修复为动态计算（356个），现在需要改回100个

2. **向后兼容性**：
   - 如果其他代码依赖 `get_spatio_temporal_features_torch()` 返回356个tokens
   - 需要检查并更新

3. **特征提取的一致性**：
   - 确保推理时的特征提取方式与训练时完全一致
   - 都是：`torch.mean(features, dim=1)` 对空间维度平均

---

## 📝 实现建议

### 推荐方案：方案 2（创建新函数）

**优点**：
- 保持 `get_spatio_temporal_features_torch()` 不变（向后兼容）
- 创建新函数 `get_temporal_features_torch()` 专门用于训练/推理
- 代码清晰，职责分明

**实现**：
```python
# 在 video_chatgpt/inference.py 中
def get_temporal_features_torch(features):
    """只提取 temporal tokens（与训练时一致）"""
    t, s, c = features.shape
    temporal_tokens = torch.mean(features, dim=1)  # [t, c]
    
    # Padding to 100
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    
    return temporal_tokens.half()

# 在 run_cli.py 中
from video_chatgpt.inference import get_temporal_features_torch
video_spatio_temporal_features = get_temporal_features_torch(frame_features)  # [100, 1024]
actual_video_token_len = 100  # 固定为100
```

---

## 🎯 总结

**核心观点**：
- 如果训练时只使用 temporal tokens (100个)
- 推理时也应该只使用 temporal tokens (100个)
- 保持训练和推理的一致性

**实现方式**：
- 创建新函数 `get_temporal_features_torch()` 专门用于训练/推理
- 修改 `run_cli.py` 使用新函数
- Prompt 中使用100个 `<vid_patch>` tokens（固定，不再动态计算）

**好处**：
- 训练推理一致
- 代码更简单
- 与 TemporalTransformer 设计匹配
- 计算效率更高

