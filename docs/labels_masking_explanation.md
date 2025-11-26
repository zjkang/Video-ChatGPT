# Labels Masking 详解：为什么只在 Assistant 回答部分计算 Loss

## 🎯 核心概念

**`-100` 是 PyTorch `CrossEntropyLoss` 的 `ignore_index`**，表示在计算 loss 时**忽略该位置的预测**。

---

## 📝 完整流程

### 步骤 1: 构造 Prompt

```python
# train_mem.py:78
prompt = f"Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
```

**示例**:
```
"Human: <video> <vid_patch>...<vid_patch> What is happening in this video?\nAssistant: A person is walking down the street.</s>"
```

### 步骤 2: Tokenize 并创建 Labels

```python
# train_mem.py:90-91
input_ids = tokenized.input_ids[0]  # [1, 2, 32000, ..., 32000, 3, 4, 5, ...]
labels = input_ids.clone()          # 初始时，labels 和 input_ids 相同
```

**初始状态**:
```python
input_ids = [1, 2, 32000, 32000, ..., 32000, 3, 4, 5, 6, 7, ...]
#            ↑ 特殊token  ↑ 100个<vid_patch>  ↑ "Human:"  ↑ 问题文本  ↑ "Assistant:"  ↑ 回答文本

labels = [1, 2, 32000, 32000, ..., 32000, 3, 4, 5, 6, 7, ...]
#         ↑ 和 input_ids 相同
```

### 步骤 3: Labels Masking（关键！）

```python
# train_mem.py:96-107
prompt_before_answer = f"Human: <video> {video_tokens} {q}\nAssistant:"
before_tokenized = self.tokenizer(prompt_before_answer, ...)
before_ids = before_tokenized.input_ids[0].tolist()

# 将 "Assistant:" 之前的所有 tokens 设为 -100
labels[:len(before_ids)] = -100
```

**Masking 后的结果**:
```python
input_ids = [1, 2, 32000, 32000, ..., 32000, 3, 4, 5, 6, 7, 8, 9, ...]
#            ↑ 特殊token  ↑ 100个<vid_patch>  ↑ "Human:"  ↑ 问题  ↑ "Assistant:"  ↑ 回答

labels = [-100, -100, -100, -100, ..., -100, -100, -100, -100, 6, 7, 8, 9, ...]
#         ↑ 全部设为 -100 (忽略)                                    ↑ 保留原值 (计算 loss)
#         所有 "Assistant:" 之前的内容                               Assistant 回答部分
```

---

## 🔍 为什么这样做？

### 问题：如果不在 Assistant 回答部分 mask，会发生什么？

**如果不对 labels 进行 masking**:
```python
labels = input_ids  # 不进行 masking
```

模型会学习：
- 预测 "Human:" 后面的 token
- 预测问题文本
- 预测 "Assistant:" 后面的 token（这是我们想要的）
- **重复学习问题**（这是浪费的！）

**示例**:
```
输入: "Human: What is happening?\nAssistant: A person is walking."
```

如果不对 labels mask，模型会学习：
- 预测 "What"（❌ 不需要）
- 预测 "is"（❌ 不需要）
- 预测 "happening?"（❌ 不需要）
- 预测 "A"（✅ 需要）
- 预测 "person"（✅ 需要）
- ...

### 解决方案：Labels Masking

**只让模型学习生成 Assistant 的回答**:
```python
labels = [-100, -100, ..., -100, 6, 7, 8, 9, ...]
#         ↑ 忽略所有问题部分              ↑ 只学习回答部分
```

模型只会学习：
- 预测 "A"（✅ 需要）
- 预测 "person"（✅ 需要）
- 预测 "is"（✅ 需要）
- ...

---

## 💡 CrossEntropyLoss 如何处理 -100？

### PyTorch 的 CrossEntropyLoss 行为

```python
# video_chatgpt/model/video_chatgpt.py:258-263
loss_fct = CrossEntropyLoss()
loss = loss_fct(shift_logits, shift_labels)
```

**CrossEntropyLoss 的默认行为**:
- 当 `labels[i] == -100` 时，**忽略该位置的 loss 计算**
- 只计算 `labels[i] != -100` 的位置

**示例**:
```python
logits = [[0.1, 0.2, 0.7],    # 位置 0 的预测
          [0.3, 0.5, 0.2],    # 位置 1 的预测
          [0.8, 0.1, 0.1],    # 位置 2 的预测
          [0.2, 0.6, 0.2]]    # 位置 3 的预测

labels = [-100, -100, 2, 1]   # 位置 0,1 忽略，位置 2,3 计算 loss

# CrossEntropyLoss 只会计算：
# - 位置 2: loss = -log(0.1)  # 预测 token 2 的概率
# - 位置 3: loss = -log(0.6)  # 预测 token 1 的概率
# 位置 0,1 的 loss = 0（被忽略）
```

---

## 📊 完整示例

### 输入数据

```python
q = "What is happening in this video?"
a = "A person is walking down the street."
```

### Prompt 构造

```python
prompt = "Human: <video> <vid_patch>...<vid_patch> What is happening in this video?\nAssistant: A person is walking down the street.</s>"
```

### Tokenize 后

```python
input_ids = [
    1,           # <s> (特殊 token)
    2,           # 其他特殊 token
    32000,       # <vid_patch> token 1
    32000,       # <vid_patch> token 2
    ...
    32000,       # <vid_patch> token 100
    1234,        # "Human:" token
    5678,        # "What" token
    9012,        # "is" token
    3456,        # "happening" token
    ...
    7890,        # "Assistant:" token
    1111,        # "A" token (回答开始)
    2222,        # "person" token
    3333,        # "is" token
    4444,        # "walking" token
    ...
]
```

### Labels Masking

```python
# 找到 "Assistant:" 之前的所有 tokens
before_answer_ids = [1, 2, 32000, ..., 32000, 1234, 5678, 9012, ..., 7890]
# 长度: 假设是 110

# Mask 掉这些位置
labels = [
    -100,        # <s> (忽略)
    -100,        # 其他特殊 token (忽略)
    -100,        # <vid_patch> token 1 (忽略)
    -100,        # <vid_patch> token 2 (忽略)
    ...
    -100,        # <vid_patch> token 100 (忽略)
    -100,        # "Human:" token (忽略)
    -100,        # "What" token (忽略)
    -100,        # "is" token (忽略)
    -100,        # "happening" token (忽略)
    ...
    -100,        # "Assistant:" token (忽略)
    1111,        # "A" token (✅ 计算 loss)
    2222,        # "person" token (✅ 计算 loss)
    3333,        # "is" token (✅ 计算 loss)
    4444,        # "walking" token (✅ 计算 loss)
    ...
]
```

### Loss 计算

```python
# 模型预测每个位置的 token
logits = model(input_ids)  # (batch, seq_len, vocab_size)

# 计算 loss（只计算 labels != -100 的位置）
loss = CrossEntropyLoss(logits, labels)
# 只会计算 "A", "person", "is", "walking", ... 这些位置的 loss
# 忽略所有 "Assistant:" 之前的位置
```

---

## 🎯 关键理解点

### 1. 为什么用 -100？

- **-100 是 PyTorch CrossEntropyLoss 的默认 `ignore_index`**
- 任何其他值（如 -1, 0）都会被当作有效的 token ID，导致错误
- -100 是标准做法，表示"忽略该位置"

### 2. 为什么只学习回答部分？

- **避免重复学习问题**：模型不需要学习如何重复问题
- **提高训练效率**：只关注需要学习的内容
- **符合对话模型训练标准**：这是训练对话模型的常见做法

### 3. Shift 操作

```python
# video_chatgpt/model/video_chatgpt.py:255-256
shift_logits = logits[..., :-1, :]  # 去掉最后一个位置
shift_labels = labels[..., 1:]      # 去掉第一个位置
```

**为什么需要 shift？**

- 语言模型是**自回归**的：用前 n 个 tokens 预测第 n+1 个 token
- `logits[i]` 预测的是 `input_ids[i+1]`
- 所以需要 shift 对齐：
  - `logits[0]` 预测 `labels[1]`
  - `logits[1]` 预测 `labels[2]`
  - ...

**示例**:
```python
input_ids = [1, 2, 3, 4]
logits = model(input_ids)  # logits[i] 预测 input_ids[i+1]

# logits[0] 预测 input_ids[1] = 2
# logits[1] 预测 input_ids[2] = 3
# logits[2] 预测 input_ids[3] = 4

# 所以需要：
shift_logits = logits[:-1]  # [logits[0], logits[1], logits[2]]
shift_labels = labels[1:]   # [labels[1], labels[2], labels[3]]
```

---

## 📈 效果对比

### 不使用 Labels Masking

```
Loss 计算位置: 所有位置
模型学习内容:
  - 学习重复问题 ❌
  - 学习生成回答 ✅
训练效率: 低（浪费计算资源）
```

### 使用 Labels Masking

```
Loss 计算位置: 只在 Assistant 回答部分
模型学习内容:
  - 只学习生成回答 ✅
训练效率: 高（专注学习目标）
```

---

## 🔍 代码位置总结

| 步骤 | 代码位置 | 功能 |
|------|----------|------|
| 1. 创建 labels | `train_mem.py:91` | `labels = input_ids.clone()` |
| 2. Mask 问题部分 | `train_mem.py:107` | `labels[:len(before_ids)] = -100` |
| 3. Mask padding | `train_mem.py:111` | `labels[input_ids == pad_token_id] = -100` |
| 4. 计算 loss | `video_chatgpt.py:263` | `loss = loss_fct(shift_logits, shift_labels)` |

---

## 💡 总结

**Labels Masking 的核心思想**:
1. 将不需要学习的位置设为 `-100`（忽略）
2. 只保留需要学习的位置（Assistant 回答部分）
3. CrossEntropyLoss 自动忽略 `-100` 的位置
4. 模型只学习生成回答，不学习重复问题

**这样做的好处**:
- ✅ 提高训练效率
- ✅ 避免学习无用内容
- ✅ 符合对话模型训练标准
- ✅ 让模型专注于学习如何生成回答

---

**希望这个解释帮助你理解 Labels Masking 的作用！** 🚀

