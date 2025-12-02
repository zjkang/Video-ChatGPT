# 视频特殊 Token 详解

## 🎯 这些是特殊 Token

是的，这些**都是特殊token**，用于在文本序列中标记视频内容的位置。

---

## 📋 Token 列表

| Token | 名称 | 作用 | 训练中使用 |
|-------|------|------|-----------|
| `<video>` | 视频标记 | 标记视频内容的开始 | ✅ 使用 |
| `<vid_patch>` | 视频帧标记 | 每个视频帧对应一个，占位符 | ✅ 使用 |
| `<vid_start>` | 视频开始标记 | 可选的开始标记 | ❌ 训练中不使用 |
| `<vid_end>` | 视频结束标记 | 可选的结束标记 | ❌ 训练中不使用 |

---

## 🔧 训练中的使用方式

### 1. Token 注册（添加到 Tokenizer）

```python
# train_mem.py 第220-227行
special_tokens = ["<video>", "<vid_patch>"]
existing_tokens = set(tokenizer.get_vocab().keys())
tokens_to_add = [t for t in special_tokens if t not in existing_tokens]
if tokens_to_add:
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    print(f"✅ Added special tokens: {tokens_to_add}")
```

**作用**：
- 将 `<video>` 和 `<vid_patch>` 添加到 tokenizer 的词汇表中
- 每个 token 会被分配一个唯一的 token ID（例如：32000, 32001）
- 模型会为这些 token 学习嵌入向量

### 2. Prompt 构造（训练数据格式）

```python
# train_mem.py 第108-112行
# 格式: "Human: <video> <vid_patch> <vid_patch> ... {q}\nAssistant: {a}</s>"
num_video_tokens = video_features.shape[0]  # 例如：16帧
video_tokens = " ".join(["<vid_patch>"] * num_video_tokens)  # " <vid_patch> <vid_patch> ... <vid_patch>"
prompt = f"Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
```

**示例**：
```
"Human: <video> <vid_patch> <vid_patch> <vid_patch> ... <vid_patch> What is happening?\nAssistant: A person is walking.</s>"
```

**关键点**：
- `<video>` 只出现**一次**，标记视频开始
- `<vid_patch>` 出现**多次**（等于视频帧数，例如16次）
- 用**空格分隔**，确保 tokenizer 识别为多个独立的 token

### 3. Tokenize 后的结构

```python
# train_mem.py 第115-122行
tokenized = self.tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
    add_special_tokens=True
)
input_ids = tokenized.input_ids[0]
```

**Tokenize 后的结构**：
```python
input_ids = [
    1,           # <s> (特殊token，tokenizer自动添加)
    2,           # 其他特殊token
    32000,       # <video> token ID
    32001,       # <vid_patch> token ID (第1帧)
    32001,       # <vid_patch> token ID (第2帧)
    32001,       # <vid_patch> token ID (第3帧)
    ...
    32001,       # <vid_patch> token ID (第16帧)
    1234,        # "Human:" token
    5678,        # "What" token
    ...
    7890,        # "Assistant:" token
    1111,        # "A" token (回答)
    ...
]
```

---

## 🔄 模型 Forward 时的处理

### 关键步骤：替换 `<vid_patch>` 为视频特征

```python
# video_chatgpt/model/video_chatgpt.py 第173行
masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
mask_index_start = masked_indices[0]
```

**作用**：
1. 找到所有 `<vid_patch>` token 的位置
2. 将这些位置的**文本嵌入**替换为**视频特征嵌入**

**替换过程**：
```python
# 原始文本嵌入
cur_input_embeds = [
    [emb1],      # "Human:" 的嵌入
    [emb2],      # <video> 的嵌入
    [emb3],      # <vid_patch> 的嵌入 (会被替换)
    [emb4],      # <vid_patch> 的嵌入 (会被替换)
    ...
    [emb20],     # <vid_patch> 的嵌入 (会被替换)
    [emb21],     # "What" 的嵌入
    ...
]

# 视频特征嵌入 (16, 4096)
cur_video_features = [
    [feat1],     # 第1帧的特征
    [feat2],     # 第2帧的特征
    ...
    [feat16],    # 第16帧的特征
]

# 替换后
cur_new_input_embeds = torch.cat((
    cur_input_embeds[:mask_index_start],      # "Human:", <video> 的嵌入
    cur_video_features,                       # 16个视频特征嵌入
    cur_input_embeds[mask_index_start + 16:]  # "What", ... 的嵌入
), dim=0)
```

---

## 🎯 为什么这样设计？

### 1. `<video>` Token 的作用

- **标记视频内容的开始**
- 告诉模型："接下来是视频内容"
- 类似于图像模型中的 `<image>` token

### 2. `<vid_patch>` Token 的作用

- **占位符**：在文本序列中为视频帧预留位置
- **动态数量**：根据视频帧数（16/32/100）动态调整
- **特征替换**：在 forward 时被实际的视频特征替换

### 3. 为什么用多个 `<vid_patch>`？

- **时序建模**：每个 `<vid_patch>` 对应一帧，保持时序信息
- **位置对应**：第1个 `<vid_patch>` → 第1帧特征，第2个 → 第2帧特征
- **灵活长度**：支持不同长度的视频（16帧、32帧、100帧等）

---

## 📊 训练 vs 推理对比

| 阶段 | `<video>` | `<vid_patch>` | 数量 |
|------|-----------|---------------|------|
| **训练** | ✅ 使用 | ✅ 使用 | 等于视频帧数（例如16） |
| **推理** | ✅ 使用 | ✅ 使用 | 等于视频帧数（例如16） |

**格式一致**：
- 训练：`"Human: <video> <vid_patch> ... <vid_patch> {q}\nAssistant: {a}</s>"`
- 推理：`"Human: <video> <vid_patch> ... <vid_patch> {q}\nAssistant:"`

---

## 🔍 代码位置总结

| 功能 | 代码位置 | 说明 |
|------|----------|------|
| Token 注册 | `train_mem.py:220-227` | 添加到 tokenizer |
| Prompt 构造 | `train_mem.py:108-112` | 构建训练 prompt |
| Token 替换 | `video_chatgpt/model/video_chatgpt.py:173` | 找到 `<vid_patch>` 位置 |
| 特征融合 | `video_chatgpt/model/video_chatgpt.py:184-185` | 替换为视频特征 |

---

## 💡 关键理解

1. **这些是特殊token**，不是普通文本
2. **训练时注册**：通过 `tokenizer.add_tokens()` 添加到词汇表
3. **Prompt中使用**：作为字符串直接写在 prompt 中
4. **Forward时替换**：`<vid_patch>` 的位置会被视频特征替换
5. **数量对应**：`<vid_patch>` 的数量 = 视频帧数

