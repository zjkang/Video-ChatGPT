# 视频特征融合到文本的代码位置详解

## 📍 核心代码位置

**文件**: `video_chatgpt/model/video_chatgpt.py`

**主要函数**: `VideoChatGPTLlamaModel.forward()` (第 75-199 行)

---

## 🔄 完整流程代码位置

### 步骤 1: 视频特征投影 (第 108 行)

```python
# 位置: video_chatgpt/model/video_chatgpt.py:108
video_features = self.mm_projector(video_spatio_temporal_features)
```

**输入**: `video_spatio_temporal_features` - (B, 100, 1024)
**输出**: `video_features` - (B, 100, 4096)

**作用**: 将 100 个视频特征从 1024 维投影到 4096 维（匹配 LLM 的 hidden_size）

---

### 步骤 2: 找到 <vid_patch> tokens 的位置 (第 173 行)

```python
# 位置: video_chatgpt/model/video_chatgpt.py:173
masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
mask_index_start = masked_indices[0]
```

**作用**: 在 `input_ids` 中找到所有 `<vid_patch>` token 的位置

**示例**:
```python
input_ids = [1, 2, 32000, 32000, 32000, ..., 32000, 3, 4, ...]
#            ↑ 特殊token  ↑ 100个<vid_patch> tokens  ↑ 后续文本
masked_indices = [2, 3, 4, ..., 101]  # <vid_patch> 的位置
mask_index_start = 2  # 第一个 <vid_patch> 的位置
```

---

### 步骤 3: 融合视频特征到文本嵌入 (第 184-185 行)

```python
# 位置: video_chatgpt/model/video_chatgpt.py:184-185
cur_new_input_embeds = torch.cat((
    cur_input_embeds[:mask_index_start],      # 前面的文本嵌入
    cur_video_features,                       # 100个视频特征嵌入 (100, 4096)
    cur_input_embeds[mask_index_start + num_patches:]  # 后面的文本嵌入
), dim=0)
```

**关键操作**: `torch.cat()` 将三部分拼接

**详细说明**:
```python
# 原始文本嵌入
cur_input_embeds = [
    [emb1],      # "Human:" 的嵌入
    [emb2],      # "<video>" 的嵌入
    [emb3],      # 第一个 <vid_patch> 的嵌入 (将被替换)
    [emb4],      # 第二个 <vid_patch> 的嵌入 (将被替换)
    ...
    [emb103],    # 第100个 <vid_patch> 的嵌入 (将被替换)
    [emb104],    # 问题文本的嵌入
    ...
]

# 视频特征 (经过 mm_projector 投影后)
cur_video_features = [
    [video_emb1],  # 第1帧的嵌入 (4096维)
    [video_emb2],  # 第2帧的嵌入 (4096维)
    ...
    [video_emb100],  # 第100帧的嵌入 (4096维)
]

# 融合后的结果
cur_new_input_embeds = [
    [emb1],         # "Human:" 的嵌入 (保留)
    [emb2],         # "<video>" 的嵌入 (保留)
    [video_emb1],   # 第1帧的嵌入 (替换 <vid_patch>)
    [video_emb2],   # 第2帧的嵌入 (替换 <vid_patch>)
    ...
    [video_emb100], # 第100帧的嵌入 (替换 <vid_patch>)
    [emb104],       # 问题文本的嵌入 (保留)
    ...
]
```

---

### 步骤 4: LLM 处理 (第 194 行)

```python
# 位置: video_chatgpt/model/video_chatgpt.py:194
return super(VideoChatGPTLlamaModel, self).forward(
    input_ids=None,  # 使用 inputs_embeds 而不是 input_ids
    inputs_embeds=inputs_embeds,  # 融合后的嵌入
    ...
)
```

**作用**: 将融合后的 `inputs_embeds` 输入到 Llama-2 模型中进行处理

---

### 步骤 5: 计算 Loss (第 252-263 行)

```python
# 位置: video_chatgpt/model/video_chatgpt.py:252-263
# 在 VideoChatGPTLlamaForCausalLM.forward() 中

hidden_states = outputs[0]  # LLM 的输出
logits = self.lm_head(hidden_states)  # 预测下一个 token

if labels is not None:
    # Shift: tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 计算 loss (只在 labels != -100 的位置)
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
```

**作用**: 计算预测损失，只在 Assistant 回答部分计算（labels != -100 的位置）

---

## 📊 完整数据流

```
输入:
  video_spatio_temporal_features: (B, 100, 1024)
  input_ids: (B, 512)  # 包含 100 个 <vid_patch> tokens

步骤 1: 投影 (第 108 行)
  video_features = mm_projector(video_spatio_temporal_features)
  → (B, 100, 4096)

步骤 2: 文本嵌入 (第 94 行)
  inputs_embeds = embed_tokens(input_ids)
  → (B, 512, 4096)

步骤 3: 找到 <vid_patch> 位置 (第 173 行)
  masked_indices = where(input_ids == vid_patch_token)
  → [2, 3, 4, ..., 101]

步骤 4: 融合 (第 184-185 行)
  new_input_embeds = concat([
      inputs_embeds[:mask_index_start],      # 前面的文本
      video_features,                        # 100个视频特征
      inputs_embeds[mask_index_start+100:]  # 后面的文本
  ])
  → (B, 512, 4096)  # 维度不变，但内容已融合

步骤 5: LLM 处理 (第 194 行)
  outputs = llama_model(inputs_embeds=new_input_embeds)
  → hidden_states: (B, 512, 4096)

步骤 6: 计算 Loss (第 252-263 行)
  logits = lm_head(hidden_states)
  loss = CrossEntropyLoss(logits, labels)
  → loss: scalar
```

---

## 🎯 关键代码行总结

| 步骤 | 代码行 | 文件 | 功能 |
|------|--------|------|------|
| 1. 视频特征投影 | 108 | `video_chatgpt/model/video_chatgpt.py` | `video_features = self.mm_projector(...)` |
| 2. 文本嵌入 | 94 | `video_chatgpt/model/video_chatgpt.py` | `inputs_embeds = self.embed_tokens(...)` |
| 3. 找到位置 | 173 | `video_chatgpt/model/video_chatgpt.py` | `masked_indices = torch.where(...)` |
| 4. 融合特征 | 184-185 | `video_chatgpt/model/video_chatgpt.py` | `torch.cat([..., cur_video_features, ...])` |
| 5. LLM 处理 | 194 | `video_chatgpt/model/video_chatgpt.py` | `super().forward(inputs_embeds=...)` |
| 6. 计算 Loss | 252-263 | `video_chatgpt/model/video_chatgpt.py` | `loss = loss_fct(shift_logits, shift_labels)` |

---

## 💡 关键理解点

### 1. 为什么用 `inputs_embeds` 而不是 `input_ids`?

- `input_ids` 是 token IDs，需要先通过 `embed_tokens` 转换为嵌入
- 融合视频特征后，我们需要直接使用嵌入，所以用 `inputs_embeds`
- 在 `forward` 时传入 `input_ids=None`，使用 `inputs_embeds` 替代

### 2. 为什么是 100 个视频特征对应 100 个 <vid_patch> tokens?

- 每个 `<vid_patch>` token 对应一帧视频
- 100 帧视频 → 100 个 `<vid_patch>` tokens → 100 个视频特征嵌入
- 一一对应替换

### 3. 维度变化

```
视频特征: (100, 1024) → mm_projector → (100, 4096)
文本嵌入: (512, 4096) → 融合后 → (512, 4096) [维度不变，内容改变]
```

---

## 🔍 调试建议

如果想查看融合过程，可以在以下位置添加 print：

```python
# 在 video_chatgpt/model/video_chatgpt.py:184 之前
print(f"cur_input_embeds shape: {cur_input_embeds.shape}")
print(f"cur_video_features shape: {cur_video_features.shape}")
print(f"mask_index_start: {mask_index_start}, num_patches: {num_patches}")

# 在 video_chatgpt/model/video_chatgpt.py:185 之后
print(f"cur_new_input_embeds shape: {cur_new_input_embeds.shape}")
```

---

**总结**: 视频特征融合的核心代码在 `video_chatgpt/model/video_chatgpt.py` 的 `VideoChatGPTLlamaModel.forward()` 方法中，主要通过 `torch.cat()` 将视频特征嵌入替换 `<vid_patch>` tokens 对应的文本嵌入。

