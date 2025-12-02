# 推理方法对比分析

## 📋 概述

本文档对比分析两种推理方法：
1. **`run_inference_activitynet_qa.py`** - 官方评估脚本
2. **`run_cli.py`** - 我们修改后的推理脚本

## 🔍 代码流程对比

### 1. `run_inference_activitynet_qa.py` 的处理流程

```python
# 1. 加载视频（固定100帧）
video_frames = load_video(video_path)  # 默认 num_frames=100

# 2. 调用 video_chatgpt_infer 函数
output = video_chatgpt_infer(video_frames, question, conv_mode, model, ...)
```

**`video_chatgpt_infer` 函数内部（`video_chatgpt/inference.py`）：**

```python
# 1. 固定使用 100 个 tokens
num_tokens = 100  # ⚠️ 硬编码

# 2. 构建 prompt（使用 conv_templates）
qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * num_tokens  # 100个 <vid_patch>
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
prompt = conv.get_prompt()

# 3. 特征提取
image_tensor = image_processor.preprocess(video_frames, ...)
image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [100, 256, 1024]

# 4. 转换为 temporal features（没有指定 target_frames）
video_spatio_temporal_features = get_temporal_features_torch(frame_features)
# 返回: [100, 1024] (因为输入是100帧)

# 5. 生成
output_ids = model.generate(
    input_ids,
    video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
    do_sample=True,
    temperature=0.2,
    max_new_tokens=1024,
    stopping_criteria=[stopping_criteria]  # ⚠️ 没有 eos_token_id
)
```

### 2. `run_cli.py` 的处理流程

```python
# 1. 加载视频（动态帧数，默认16）
num_frames_to_load = args.num_frames if args.num_frames is not None else 16
video_frames = load_video(args.video_path, num_frames=num_frames_to_load)

# 2. 特征提取（自己实现）
image_tensor = image_processor.preprocess(video_frames, ...)
image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [num_frames, 256, 1024]

# 3. 转换为 temporal features
temporal_features = torch.mean(frame_features, dim=1)  # [num_frames, 1024]

# 4. 如果需要，进行采样/填充
if args.num_frames is not None and temporal_features.shape[0] != args.num_frames:
    video_spatio_temporal_features = get_temporal_features_torch(
        frame_features, 
        target_frames=args.num_frames
    )
else:
    video_spatio_temporal_features = temporal_features.half()

# 5. 构建 prompt（直接构造，不使用 conv_templates）
actual_video_token_len = video_spatio_temporal_features.shape[0]  # 动态
video_tokens = " ".join(["<vid_patch>"] * actual_video_token_len)
prompt = f"Human: <video> {video_tokens} {args.question}\nAssistant:"

# 6. 生成（增强的停止条件）
output_ids = model.generate(
    input_ids=input_ids,
    video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
    eos_token_id=eos_token_id,  # ✅ 明确设置
    repetition_penalty=1.5,  # ✅ 防止重复
    no_repeat_ngram_size=3,  # ✅ 防止重复
    stopping_criteria=[stopping_criteria],
    ...
)
```

## ⚠️ 关键差异

### 1. **帧数处理**

| 项目 | `run_inference_activitynet_qa.py` | `run_cli.py` |
|------|-----------------------------------|--------------|
| 默认帧数 | **100帧**（硬编码） | **16帧**（可配置） |
| 帧数来源 | `load_video()` 默认参数 | `args.num_frames` 或默认16 |
| 与训练一致性 | ❌ 如果训练用16帧，推理用100帧会不匹配 | ✅ 可配置，与训练一致 |

### 2. **Prompt 构建方式**

| 项目 | `run_inference_activitynet_qa.py` | `run_cli.py` |
|------|-----------------------------------|--------------|
| Prompt格式 | 使用 `conv_templates` | 直接构造字符串 |
| Token数量 | **硬编码100个** `<vid_patch>` | **动态**，根据实际特征长度 |
| 格式示例 | `"Human: <video> <vid_patch>...<vid_patch> {question}\nAssistant:"` (通过conv_template) | `"Human: <video> <vid_patch> <vid_patch> ... {question}\nAssistant:"` (直接构造) |

**问题**：如果 `run_inference_activitynet_qa.py` 加载了100帧，但实际特征只有16帧，prompt中会有100个 `<vid_patch>` tokens，但实际特征只有16个，会导致不匹配！

### 3. **特征提取**

| 项目 | `run_inference_activitynet_qa.py` | `run_cli.py` |
|------|-----------------------------------|--------------|
| 特征提取 | `get_temporal_features_torch(frame_features)` | `torch.mean(frame_features, dim=1)` |
| 目标帧数 | 不指定（使用实际帧数） | 可指定 `target_frames` |
| 结果 | 如果输入100帧，输出 `[100, 1024]` | 如果输入16帧，输出 `[16, 1024]` |

**注意**：两者都使用 `torch.mean(features, dim=1)` 对空间维度平均，这是**一致的**。

### 4. **生成参数**

| 项目 | `run_inference_activitynet_qa.py` | `run_cli.py` |
|------|-----------------------------------|--------------|
| `eos_token_id` | ❌ 未设置 | ✅ 明确设置 |
| `repetition_penalty` | ❌ 未设置 | ✅ 1.5 |
| `no_repeat_ngram_size` | ❌ 未设置 | ✅ 3 |
| `temperature` | 0.2 | 0.7 |
| `max_new_tokens` | 1024 | 256 |
| `stopping_criteria` | ✅ 使用 | ✅ 使用（并修复了bug） |

### 5. **停止条件**

| 项目 | `run_inference_activitynet_qa.py` | `run_cli.py` |
|------|-----------------------------------|--------------|
| 停止标记 | `conv.sep` 或 `conv.sep2` | `"</s>"` |
| `KeywordsStoppingCriteria` | ✅ 使用 | ✅ 使用（并修复了 `keyword_ids` 为空的问题） |
| `eos_token_id` | ❌ 未设置 | ✅ 明确设置并传递给 `generate()` |

## 🐛 潜在问题

### 问题1: 帧数不匹配

**`run_inference_activitynet_qa.py` 的问题：**

```python
# 加载100帧
video_frames = load_video(video_path)  # 默认100帧

# 但 prompt 中硬编码了100个 tokens
num_tokens = 100  # ⚠️ 硬编码
qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * num_tokens

# 如果实际视频只有16帧，特征只有16个，但prompt有100个tokens！
# 这会导致模型混淆
```

**`run_cli.py` 的解决方案：**

```python
# 动态加载帧数
num_frames_to_load = args.num_frames if args.num_frames is not None else 16
video_frames = load_video(args.video_path, num_frames=num_frames_to_load)

# 动态设置 token 数量
actual_video_token_len = video_spatio_temporal_features.shape[0]  # 16
video_tokens = " ".join(["<vid_patch>"] * actual_video_token_len)  # 16个tokens
```

### 问题2: 停止条件不完善

**`run_inference_activitynet_qa.py` 的问题：**

```python
# 没有设置 eos_token_id
output_ids = model.generate(
    input_ids,
    video_spatio_temporal_features=...,
    stopping_criteria=[stopping_criteria]  # ⚠️ 只有 stopping_criteria
    # ❌ 没有 eos_token_id
    # ❌ 没有 repetition_penalty
)
```

这可能导致：
- 生成无法在 `</s>` token 处停止
- 生成重复内容

**`run_cli.py` 的解决方案：**

```python
# 明确设置 eos_token_id 和防重复参数
output_ids = model.generate(
    input_ids=input_ids,
    video_spatio_temporal_features=...,
    eos_token_id=eos_token_id,  # ✅
    repetition_penalty=1.5,  # ✅
    no_repeat_ngram_size=3,  # ✅
    stopping_criteria=[stopping_criteria]  # ✅ 并修复了bug
)
```

### 问题3: Prompt 格式不一致

**`run_inference_activitynet_qa.py`：**

使用 `conv_templates`，可能添加额外的格式（如 `### Human:` 等），与训练时的格式可能不一致。

**`run_cli.py`：**

直接构造与训练时完全一致的格式：
```python
prompt = f"Human: <video> {video_tokens} {args.question}\nAssistant:"
```

## ✅ 建议

### 如果要使用 `run_inference_activitynet_qa.py` 评估16帧训练的模型：

1. **修改 `video_chatgpt/inference.py` 的 `video_chatgpt_infer` 函数：**

```python
def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, num_frames=16):
    # 使用动态帧数
    num_tokens = num_frames  # 不再硬编码100
    
    # 特征提取后，确保帧数匹配
    video_spatio_temporal_features = get_temporal_features_torch(frame_features, target_frames=num_frames)
    
    # 使用实际特征长度
    actual_num_tokens = video_spatio_temporal_features.shape[0]
    qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * actual_num_tokens
    
    # 添加 eos_token_id 和防重复参数
    eos_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        input_ids,
        video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
        eos_token_id=eos_token_id,  # ✅
        repetition_penalty=1.5,  # ✅
        ...
    )
```

2. **修改 `run_inference_activitynet_qa.py`：**

```python
# 加载视频时指定帧数
video_frames = load_video(video_path, num_frames=16)  # ✅ 与训练一致

# 调用时传递帧数
output = video_chatgpt_infer(video_frames, question, conv_mode, model, ..., num_frames=16)
```

### 或者直接使用 `run_cli.py` 进行评估

`run_cli.py` 已经解决了所有这些问题，可以直接使用。

## 📊 总结

| 特性 | `run_inference_activitynet_qa.py` | `run_cli.py` |
|------|-----------------------------------|--------------|
| 帧数灵活性 | ❌ 固定100帧 | ✅ 可配置（默认16） |
| Prompt格式 | ⚠️ 使用conv_templates | ✅ 与训练一致 |
| Token数量匹配 | ❌ 可能不匹配 | ✅ 动态匹配 |
| 停止条件 | ⚠️ 不完善 | ✅ 完善（eos_token_id + repetition_penalty） |
| 与训练一致性 | ❌ 可能不一致 | ✅ 完全一致 |
| 推荐使用 | ❌ 需要修改 | ✅ 可直接使用 |

**结论**：`run_cli.py` 的处理方式更加灵活、一致，且修复了多个潜在问题。建议使用 `run_cli.py` 进行推理，或者按照上述建议修改 `run_inference_activitynet_qa.py`。

