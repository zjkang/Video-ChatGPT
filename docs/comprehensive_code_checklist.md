# 代码一致性检查清单

## 🔍 检查结果总结

### ✅ 已一致的地方

1. **推理代码的隐藏层选择**：
   - ✅ `run_cli.py`: 使用 `hidden_states[-2]`
   - ✅ `video_chatgpt/inference.py`: 使用 `hidden_states[-2]`
   - ✅ `video_chatgpt/single_video_inference.py`: 使用 `hidden_states[-2]`
   - ✅ `video_chatgpt/demo/chat.py`: 使用 `hidden_states[-2]`
   - ✅ `scripts/save_spatio_temporal_clip_features.py`: 使用 `hidden_states[-2]`

2. **训练代码的 token 数量**：
   - ✅ `train_mem.py`: 使用 100 个 tokens

---

## ❌ 需要修改的地方

### 问题1：隐藏层选择不一致

**训练时**：
- ❌ `extract_features.py:60`: 使用 `last_hidden_state`（最后一层）
- ❌ 应该改为：`outputs.hidden_states[-2]`（倒数第二层）

**影响**：
- 训练数据用最后一层提取
- 推理时用倒数第二层提取
- 特征分布不一致 → 需要重新训练

---

### 问题2：特征格式不一致（最关键）

**训练时**：
- ✅ `extract_features.py`: 只生成 temporal tokens `(100, 1024)`
- ✅ `train_mem.py`: 使用 100 个 tokens

**推理时**：
- ❌ `run_cli.py`: 使用 `get_spatio_temporal_features_torch()` → 生成 `(356, 1024)`
- ❌ `video_chatgpt/inference.py`: 使用 `get_spatio_temporal_features_torch()` → 生成 `(356, 1024)`
- ❌ `video_chatgpt/single_video_inference.py`: 使用 `get_spatio_temporal_features_torch()` → 生成 `(356, 1024)`
- ❌ `video_chatgpt/demo/chat.py`: 使用 `get_spatio_temporal_features_torch()` → 生成 `(356, 1024)`

**影响**：
- 训练：100 个 tokens
- 推理：356 个 tokens
- 完全不一致 → 模型无法正常工作

---

### 问题3：Token 数量动态计算

**推理时**：
- ❌ `run_cli.py:44`: 动态计算 `actual_video_token_len = video_spatio_temporal_features.shape[0]`（356）
- ❌ `run_cli.py:63`: 使用动态计算的 token 数量

**应该**：
- ✅ 固定为 100 个 tokens（与训练一致）

---

## 📝 需要修改的文件清单

### 1. `extract_features.py`（训练数据生成）

**修改位置**：第 60 行

**当前代码**：
```python
features = outputs.last_hidden_state # [100, 257, 1024]
```

**应该改为**：
```python
features = outputs.hidden_states[-2]  # [100, 257, 1024] - 使用倒数第二层，与推理一致
```

**影响**：
- 需要重新生成所有训练数据
- 需要重新训练模型

---

### 2. `video_chatgpt/inference.py`（推理函数）

**需要添加新函数**：`get_temporal_features_torch()`

**当前代码**：
```python
def get_spatio_temporal_features_torch(features):
    # 返回 (356, 1024) - temporal + spatial
    ...
```

**应该添加**：
```python
def get_temporal_features_torch(features):
    """
    只提取 temporal tokens（与训练时一致）
    输入: [t, s, c] - 时间帧数 t，空间patch数 s，特征维度 c
    输出: [100, c] - temporal tokens，padding到100
    """
    t, s, c = features.shape
    temporal_tokens = torch.mean(features, dim=1)  # [t, c] - 对空间维度平均
    
    # Padding to 100
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    
    return temporal_tokens.half()  # [100, 1024]
```

**保持 `get_spatio_temporal_features_torch()` 不变**（向后兼容）

---

### 3. `run_cli.py`（CLI 推理）

**修改位置1**：第 11 行（导入）

**当前代码**：
```python
from video_chatgpt.inference import get_spatio_temporal_features_torch
```

**应该改为**：
```python
from video_chatgpt.inference import get_temporal_features_torch
```

**修改位置2**：第 41 行（特征提取）

**当前代码**：
```python
video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)
```

**应该改为**：
```python
video_spatio_temporal_features = get_temporal_features_torch(frame_features)  # [100, 1024]
```

**修改位置3**：第 43-45 行（Token 数量）

**当前代码**：
```python
# 计算实际的视频 token 数量（temporal: 100 + spatial: s）
actual_video_token_len = video_spatio_temporal_features.shape[0]  # 实际特征数量
print(f"📊 Actual video token length: {actual_video_token_len} (temporal: 100 + spatial: {actual_video_token_len - 100})")
```

**应该改为**：
```python
# 固定为 100 个 tokens（与训练一致）
actual_video_token_len = 100
print(f"📊 Using {actual_video_token_len} temporal tokens (consistent with training)")
```

---

### 4. `video_chatgpt/inference.py`（推理函数）

**修改位置**：第 90 行

**当前代码**：
```python
video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)
```

**应该改为**：
```python
video_spatio_temporal_features = get_temporal_features_torch(frame_features)  # [100, 1024]
```

**注意**：需要先添加 `get_temporal_features_torch()` 函数

---

### 5. `video_chatgpt/single_video_inference.py`（单视频推理）

**修改位置1**：需要添加 `get_temporal_features_torch()` 函数（或从 `inference.py` 导入）

**修改位置2**：第 109 行

**当前代码**：
```python
video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)
```

**应该改为**：
```python
video_spatio_temporal_features = get_temporal_features_torch(frame_features)  # [100, 1024]
```

---

### 6. `video_chatgpt/demo/chat.py`（Demo）

**修改位置1**：需要添加 `get_temporal_features_torch()` 方法（或从 `inference.py` 导入）

**修改位置2**：第 113 行

**当前代码**：
```python
video_spatio_temporal_features = self.get_spatio_temporal_features_torch(frame_features)
```

**应该改为**：
```python
video_spatio_temporal_features = self.get_temporal_features_torch(frame_features)  # [100, 1024]
```

**或者**：从 `inference.py` 导入 `get_temporal_features_torch`

---

## 🎯 修改优先级

### 高优先级（必须修改）

1. ✅ **`extract_features.py`**: 修改隐藏层选择（如果决定统一使用倒数第二层）
2. ✅ **`video_chatgpt/inference.py`**: 添加 `get_temporal_features_torch()` 函数
3. ✅ **`run_cli.py`**: 使用 `get_temporal_features_torch()`，固定 token 数量为 100

### 中优先级（建议修改）

4. ✅ **`video_chatgpt/inference.py`**: 修改 `video_chatgpt_infer()` 函数使用 temporal only
5. ✅ **`video_chatgpt/single_video_inference.py`**: 修改使用 temporal only
6. ✅ **`video_chatgpt/demo/chat.py`**: 修改使用 temporal only

---

## 📋 修改后的预期状态

### 训练时

1. **特征提取** (`extract_features.py`):
   - 使用 `hidden_states[-2]`（倒数第二层）
   - 输出：`(100, 1024)` - 只有 temporal tokens

2. **训练数据** (`train_mem.py`):
   - 加载：`(100, 1024)` 特征
   - Prompt：100 个 `<vid_patch>` tokens

### 推理时

1. **特征提取** (所有推理文件):
   - 使用 `hidden_states[-2]`（倒数第二层）✅ 已一致
   - 使用 `get_temporal_features_torch()` → 输出 `(100, 1024)`

2. **Prompt 构造**:
   - 固定使用 100 个 `<vid_patch>` tokens

### 一致性检查

- ✅ 隐藏层选择：训练和推理都使用 `hidden_states[-2]`
- ✅ 特征格式：训练和推理都是 `(100, 1024)`
- ✅ Token 数量：训练和推理都是 100 个

---

## ⚠️ 注意事项

1. **重新生成训练数据**：
   - 修改 `extract_features.py` 后，需要重新生成所有 `.pkl` 文件

2. **重新训练模型**：
   - 使用新的训练数据重新训练模型

3. **向后兼容**：
   - 保持 `get_spatio_temporal_features_torch()` 不变
   - 如果其他代码依赖它，不会受影响

4. **测试**：
   - 修改后需要测试训练和推理是否正常工作
   - 确保特征维度匹配

---

## ✅ 检查完成后的状态

修改完成后，应该达到：

1. ✅ 训练和推理使用相同的隐藏层（倒数第二层）
2. ✅ 训练和推理使用相同的特征格式（100 个 temporal tokens）
3. ✅ 训练和推理使用相同的 token 数量（100 个）
4. ✅ 所有推理代码统一使用 `get_temporal_features_torch()`

这样训练和推理就完全一致了！

