# 训练代码 vs 评估代码不一致分析

## 📋 代码对比

### 训练代码 (`train_mem.py`)

**特征加载**：
```python
# 从 .pkl 文件加载预提取的特征
video_features = pickle.load(f)  # shape: (num_frames, 1024)
num_video_tokens = video_features.shape[0]  # 动态：16/32/100等
```

**Prompt 构建**：
```python
# 使用空格分隔，确保 tokenizer 识别为多个 token
video_tokens = " ".join(["<vid_patch>"] * num_video_tokens)
prompt = f"Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
```

**关键特点**：
- ✅ Token 数量：**动态**，根据特征文件的实际帧数（16/32/100等）
- ✅ Prompt 格式：**直接构造字符串**，格式固定
- ✅ Token 分隔：**使用空格** `" ".join(...)`
- ✅ 特征来源：**预提取的 .pkl 文件**

---

### 评估代码 1: `video_chatgpt/inference.py` (video_chatgpt_infer)

**特征提取**：
```python
# 从视频帧实时提取特征
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [100, 256, 1024]
video_spatio_temporal_features = get_temporal_features_torch(frame_features)  # [100, 1024]
```

**Prompt 构建**：
```python
# 硬编码 100 个 tokens
num_tokens = 100  # ⚠️ 硬编码！
if model.get_model().vision_config.use_vid_start_end:
    qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * num_tokens + DEFAULT_VID_END_TOKEN
else:
    qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * num_tokens  # ⚠️ 直接拼接，没有空格！

# 使用 conv_templates
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
prompt = conv.get_prompt()  # ⚠️ 格式可能不同
```

**关键特点**：
- ❌ Token 数量：**硬编码 100**，不匹配训练时的 16 帧
- ❌ Prompt 格式：**使用 conv_templates**，格式可能与训练不一致
- ❌ Token 分隔：**直接拼接** `DEFAULT_VIDEO_PATCH_TOKEN * num_tokens`，没有空格
- ✅ 特征来源：**实时提取**（但固定为 100 帧）

---

### 评估代码 2: `run_cli.py` (我们的修复版本)

**特征提取**：
```python
# 从视频帧实时提取特征
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [num_frames, 256, 1024]
temporal_features = torch.mean(frame_features, dim=1)  # [num_frames, 1024]
video_spatio_temporal_features = temporal_features.half()  # 动态帧数
```

**Prompt 构建**：
```python
# 动态 token 数量，与训练一致
actual_video_token_len = video_spatio_temporal_features.shape[0]  # 16/32/100等
video_tokens = " ".join([DEFAULT_VIDEO_PATCH_TOKEN] * actual_video_token_len)  # ✅ 使用空格
prompt = f"Human: <video> {video_tokens} {args.question}\nAssistant:"  # ✅ 直接构造，与训练一致
```

**关键特点**：
- ✅ Token 数量：**动态**，与训练一致
- ✅ Prompt 格式：**直接构造字符串**，与训练完全一致
- ✅ Token 分隔：**使用空格** `" ".join(...)`，与训练一致
- ✅ 特征来源：**实时提取**，但支持动态帧数

---

## 🚨 关键不一致点

### 不一致 1: Token 数量硬编码

**问题**：
- **训练时**：使用 16 帧（或其他动态数量）
- **`video_chatgpt_infer`**：硬编码 100 个 tokens
- **结果**：Token 数量不匹配，模型会跳过视频分支！

**影响**：
```python
# 在 forward 方法中
if (torch.as_tensor(cur_input_ids, device=self.device) == self.vision_config.vid_patch_token).sum() != num_patches:
    raise ValueError("The number of video patch tokens should be the same as the number of video patches.")
```

如果 token 数量不匹配，会抛出错误或跳过视频特征。

### 不一致 2: Prompt 格式不同

**问题**：
- **训练时**：`"Human: <video> <vid_patch> <vid_patch> ... {q}\nAssistant: {a}</s>"`
- **`video_chatgpt_infer`**：使用 `conv_templates`，可能添加额外格式

**示例**：
```python
# conv_templates 可能生成：
"### Human: <video> <vid_patch>...<vid_patch> {question}\n### Assistant:"
# 而不是训练时的：
"Human: <video> <vid_patch> <vid_patch> ... {q}\nAssistant:"
```

**影响**：模型可能无法正确识别 prompt 格式，导致性能下降。

### 不一致 3: Token 分隔方式不同

**问题**：
- **训练时**：`" ".join(["<vid_patch>"] * num_video_tokens)` → `"<vid_patch> <vid_patch> <vid_patch> ..."`
- **`video_chatgpt_infer`**：`DEFAULT_VIDEO_PATCH_TOKEN * num_tokens` → `"<vid_patch><vid_patch><vid_patch>..."`

**影响**：
- 有空格：tokenizer 会识别为多个独立的 `<vid_patch>` tokens
- 无空格：tokenizer 可能识别为一个长 token 或无法正确分割

**验证**：
```python
# 有空格
tokenizer(" ".join(["<vid_patch>"] * 3))
# 输出: ['<vid_patch>', '<vid_patch>', '<vid_patch>']  # ✅ 3个tokens

# 无空格
tokenizer("<vid_patch>" * 3)
# 输出: ['<vid_patch><vid_patch><vid_patch>']  # ❌ 可能只有1个token
```

### 不一致 4: 特征提取方式

**问题**：
- **训练时**：从预提取的 .pkl 文件加载，已经是 temporal features `(num_frames, 1024)`
- **`video_chatgpt_infer`**：实时提取，使用 `get_temporal_features_torch()`，固定输出 100 帧

**影响**：
- 如果训练用 16 帧，但推理用 100 帧，特征数量不匹配
- `get_temporal_features_torch()` 内部会进行 padding/采样，可能改变特征分布

---

## 🔍 为什么会出现这些不一致？

### 历史原因

1. **原始代码设计**：
   - 原始 Video-ChatGPT 可能设计为使用 100 帧
   - `video_chatgpt_infer` 是官方评估函数，假设使用 100 帧

2. **训练代码修改**：
   - 我们修改了训练代码，支持动态帧数（16/32/100等）
   - 但评估代码没有同步更新

3. **代码演进**：
   - `run_cli.py` 是我们新写的推理脚本，修复了这些问题
   - `video_chatgpt_infer` 是原始代码，没有更新

### 设计问题

1. **硬编码假设**：
   - `video_chatgpt_infer` 假设所有模型都使用 100 帧
   - 没有考虑不同训练配置的差异

2. **格式不统一**：
   - 训练和评估使用不同的 prompt 构建方式
   - 没有统一的 prompt 模板

3. **缺少验证**：
   - 没有检查 token 数量是否匹配
   - 没有验证 prompt 格式是否一致

---

## ✅ 解决方案

### 方案 1: 使用修复后的 `run_cli.py`（推荐）

**优点**：
- ✅ 与训练代码完全一致
- ✅ 支持动态帧数
- ✅ 有完整的验证和日志

**使用**：
```bash
python run_cli.py \
    --projection_path ./checkpoints/Baseline_T16 \
    --video_path ./sample.mp4 \
    --question "Describe the video." \
    --num_frames 16
```

### 方案 2: 修复 `video_chatgpt_infer` 函数

**需要修改**：

1. **支持动态帧数**：
```python
def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, num_frames=16):
    # 使用动态帧数
    num_tokens = num_frames  # 不再硬编码100
    ...
```

2. **修复 token 分隔**：
```python
# 使用空格分隔
video_tokens = " ".join([DEFAULT_VIDEO_PATCH_TOKEN] * num_tokens)
qs = question + '\n' + video_tokens
```

3. **统一 prompt 格式**：
```python
# 直接构造，不使用 conv_templates
prompt = f"Human: <video> {video_tokens} {question}\nAssistant:"
```

4. **添加验证**：
```python
# 验证 token 数量匹配
actual_num_tokens = video_spatio_temporal_features.shape[0]
if actual_num_tokens != num_tokens:
    raise ValueError(f"Token count mismatch: {num_tokens} tokens != {actual_num_tokens} features")
```

### 方案 3: 修复 `run_inference_activitynet_qa.py`

**需要修改**：

1. **传递帧数参数**：
```python
# 加载视频时指定帧数
video_frames = load_video(video_path, num_frames=16)  # 与训练一致

# 调用时传递帧数
output = video_chatgpt_infer(video_frames, question, conv_mode, model, ..., num_frames=16)
```

---

## 📊 对比总结表

| 特性 | 训练代码 | video_chatgpt_infer | run_cli.py |
|------|---------|---------------------|------------|
| Token 数量 | ✅ 动态（16/32/100） | ❌ 硬编码 100 | ✅ 动态（16/32/100） |
| Prompt 格式 | ✅ 直接构造 | ❌ conv_templates | ✅ 直接构造 |
| Token 分隔 | ✅ 使用空格 | ❌ 直接拼接 | ✅ 使用空格 |
| 特征来源 | ✅ 预提取 .pkl | ✅ 实时提取 | ✅ 实时提取 |
| 帧数支持 | ✅ 动态 | ❌ 固定 100 | ✅ 动态 |
| 验证检查 | ⚠️ 无 | ❌ 无 | ✅ 完整验证 |
| 与训练一致性 | ✅ 基准 | ❌ 不一致 | ✅ 完全一致 |

---

## 🎯 结论

**主要问题**：
1. `video_chatgpt_infer` 硬编码 100 tokens，不匹配训练时的 16 帧
2. `video_chatgpt_infer` 使用 `conv_templates`，prompt 格式与训练不一致
3. `video_chatgpt_infer` 不使用空格分隔 tokens，可能导致 tokenizer 识别问题

**推荐方案**：
- ✅ 使用修复后的 `run_cli.py` 进行推理
- ✅ 或者修复 `video_chatgpt_infer` 函数，使其支持动态帧数和正确的 prompt 格式

**关键原则**：
- 训练和推理必须使用**完全相同的**：
  1. Token 数量
  2. Prompt 格式
  3. Token 分隔方式
  4. 特征处理方式

