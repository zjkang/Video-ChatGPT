# 代码修改总结

## ✅ 已完成的修改

### 1. `extract_features.py` - 训练数据生成

**修改位置**：第 60 行

**修改内容**：
- 从 `outputs.last_hidden_state`（最后一层）改为 `outputs.hidden_states[-2]`（倒数第二层）
- 与推理代码保持一致

**影响**：
- ⚠️ **需要重新生成所有训练数据**
- ⚠️ **需要重新训练模型**

---

### 2. `video_chatgpt/inference.py` - 推理函数库

**修改内容**：
1. **添加新函数** `get_temporal_features_torch()`：
   - 只返回 temporal tokens `(100, 1024)`
   - 与训练时一致

2. **保留旧函数** `get_spatio_temporal_features_torch()`：
   - 保持向后兼容
   - 返回 spatio-temporal tokens `(356, 1024)`

3. **修改** `video_chatgpt_infer()` 函数：
   - 使用 `get_temporal_features_torch()` 而不是 `get_spatio_temporal_features_torch()`
   - 固定使用 100 个 tokens

---

### 3. `run_cli.py` - CLI 推理脚本

**修改内容**：
1. **导入**：从 `get_spatio_temporal_features_torch` 改为 `get_temporal_features_torch`
2. **特征提取**：使用 `get_temporal_features_torch()` 只提取 temporal tokens
3. **Token 数量**：固定为 100（不再动态计算）

---

### 4. `video_chatgpt/single_video_inference.py` - 单视频推理

**修改内容**：
1. **导入**：从 `inference.py` 导入 `get_temporal_features_torch`
2. **特征提取**：使用 `get_temporal_features_torch()` 只提取 temporal tokens
3. **保留旧函数**：`get_spatio_temporal_features_torch()` 保留用于向后兼容

---

### 5. `video_chatgpt/demo/chat.py` - Demo 界面

**修改内容**：
1. **添加新方法** `get_temporal_features_torch()`：
   - 只返回 temporal tokens `(100, 1024)`
   - 与训练时一致

2. **修改特征提取**：
   - 使用 `get_temporal_features_torch()` 而不是 `get_spatio_temporal_features_torch()`

---

## 📊 修改后的状态

### 训练时

1. **特征提取** (`extract_features.py`):
   - ✅ 使用 `hidden_states[-2]`（倒数第二层）
   - ✅ 输出：`(100, 1024)` - 只有 temporal tokens

2. **训练数据** (`train_mem.py`):
   - ✅ 加载：`(100, 1024)` 特征
   - ✅ Prompt：100 个 `<vid_patch>` tokens

### 推理时

1. **特征提取** (所有推理文件):
   - ✅ 使用 `hidden_states[-2]`（倒数第二层）
   - ✅ 使用 `get_temporal_features_torch()` → 输出 `(100, 1024)`

2. **Prompt 构造**:
   - ✅ 固定使用 100 个 `<vid_patch>` tokens

### 一致性检查

- ✅ **隐藏层选择**：训练和推理都使用 `hidden_states[-2]`
- ✅ **特征格式**：训练和推理都是 `(100, 1024)`
- ✅ **Token 数量**：训练和推理都是 100 个

---

## ⚠️ 重要提醒

### 必须执行的操作

1. **重新生成训练数据**：
   ```bash
   # 删除旧的训练数据（或备份）
   rm -rf data/mini_dataset/features/*.pkl
   
   # 重新运行特征提取
   python extract_features.py
   ```

2. **重新训练模型**：
   ```bash
   # 使用新的训练数据重新训练
   python train_mem.py \
       --output_dir ./checkpoints/new_baseline \
       ...
   ```

### 向后兼容性

- ✅ `get_spatio_temporal_features_torch()` 函数保留
- ✅ 如果其他代码依赖它，不会受影响
- ✅ 但建议所有新代码使用 `get_temporal_features_torch()`

---

## 🎯 修改完成

所有必要的修改已完成！现在训练和推理完全一致：

1. ✅ 使用相同的隐藏层（倒数第二层）
2. ✅ 使用相同的特征格式（100 个 temporal tokens）
3. ✅ 使用相同的 token 数量（100 个）

下一步：重新生成训练数据并重新训练模型。

