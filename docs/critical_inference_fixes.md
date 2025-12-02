# 关键推理问题修复

## 🚨 问题描述

用户发现训练的模型在推理时"胡说八道"，输出的文本与视频内容无关。经过分析，发现了两个关键问题：

### 问题①：推理脚本没有把视频特征送进模型的正确入口

**问题现象**：
- 模型 forward 处理视频的唯一方式是使用 `video_spatio_temporal_features` 参数
- 但推理脚本可能使用了错误的参数名（如 `images=temporal_features`）
- 或者 `forward` 方法的条件判断导致视频特征被跳过

**根本原因**：
在 `VideoChatGPTLlamaModel.forward()` 方法中，第81行的条件判断：
```python
if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:
```

这个条件的问题是：
- 当 `input_ids.shape[1] == 1` 且 `self.training == False` 时，视频特征会被跳过
- 在生成过程中，当使用 `past_key_values` 时，`prepare_inputs_for_generation` 会将 `input_ids` 设置为 `input_ids[:, -1:]`（只包含最后一个token），这会导致 `input_ids.shape[1] == 1`
- 因此，在生成的第2步及之后，视频特征会被忽略

**修复方案**：
修改 `forward` 方法的条件，确保在推理时（第一次 forward，`past_key_values is None`）也能处理视频特征：

```python
# 【关键修复】处理视频特征的条件：
# 1. 训练时：总是处理视频特征
# 2. 推理时：只在第一次 forward（past_key_values is None）时处理视频特征
#    因为后续步骤使用 past_key_values，不需要再次处理视频特征
# 3. 如果 input_ids.shape[1] == 1 且 past_key_values is None，说明是单token输入，也应该处理视频特征
should_process_video = (
    video_spatio_temporal_features is not None and 
    (self.training or past_key_values is None)
)

if should_process_video:
    # 处理视频特征...
```

**关键点**：
- 视频特征只需要在第一次 forward 时处理（当 `past_key_values is None` 时）
- 后续步骤使用 `past_key_values` 时，视频特征已经被嵌入到 `inputs_embeds` 中，不需要再次处理
- 这个修复确保了在推理的第一次 forward 时，视频特征总是被处理

### 问题②：推理 prompt 的 `<vid_patch>` token 数量和特征数量必须一致

**问题现象**：
- 训练时使用 16 个 `<vid_patch>` tokens
- 但推理时如果 prompt 中的 `<vid_patch>` 数量与特征数量不一致，模型会跳过视频分支

**根本原因**：
在 `VideoChatGPTLlamaModel.forward()` 方法中，第153行有检查：
```python
if (torch.as_tensor(cur_input_ids, device=self.device) == self.vision_config.vid_patch_token).sum() != num_patches:
    raise ValueError(
        "The number of video patch tokens should be the same as the number of video patches."
    )
```

如果 token 数量不匹配，会抛出错误。但更严重的是，如果 prompt 中没有 `<vid_patch>` tokens，代码会执行 fallback path：
```python
if (torch.as_tensor(cur_input_ids, device=self.device) == self.vision_config.vid_patch_token).sum() == 0:
    # Multimodal LLM, but the current sample is not multimodal
    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum()
    new_input_embeds.append(cur_input_embeds)
    cur_video_idx += 1
    continue
```

这会导致模型忽略视频，恢复成纯文本 LLaMA 输出。

**修复方案**：
在 `run_cli.py` 中添加验证，确保 `<vid_patch>` token 数量与特征数量一致：

```python
# 【关键修复】验证 token 数量与特征数量必须一致
if num_vid_patch_tokens != actual_video_token_len:
    logger.error(f"❌ CRITICAL ERROR: Token count mismatch!")
    logger.error(f"   - <vid_patch> tokens in prompt: {num_vid_patch_tokens}")
    logger.error(f"   - Video features count: {actual_video_token_len}")
    logger.error(f"   - This will cause the model to ignore video features!")
    logger.error(f"   - Model will fallback to text-only generation!")
    raise ValueError(
        f"Token count mismatch: {num_vid_patch_tokens} <vid_patch> tokens in prompt "
        f"but {actual_video_token_len} video features. "
        f"They must be equal for video features to be processed correctly."
    )
else:
    logger.info(f"✅ Token count matches: {num_vid_patch_tokens} tokens == {actual_video_token_len} features")
```

**关键点**：
- 训练时使用多少个 `<vid_patch>` tokens，推理时也必须使用相同数量
- 特征数量（`video_spatio_temporal_features.shape[0]`）必须与 `<vid_patch>` token 数量完全一致
- 如果不一致，模型会跳过视频分支，导致"胡说八道"

## ✅ 修复内容

### 1. 修改 `video_chatgpt/model/video_chatgpt.py`

**文件位置**：`video_chatgpt/model/video_chatgpt.py` 第81行

**修改前**：
```python
if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:
```

**修改后**：
```python
# 【关键修复】处理视频特征的条件：
# 1. 训练时：总是处理视频特征
# 2. 推理时：只在第一次 forward（past_key_values is None）时处理视频特征
#    因为后续步骤使用 past_key_values，不需要再次处理视频特征
# 3. 如果 input_ids.shape[1] == 1 且 past_key_values is None，说明是单token输入，也应该处理视频特征
should_process_video = (
    video_spatio_temporal_features is not None and 
    (self.training or past_key_values is None)
)

if should_process_video:
```

### 2. 修改 `run_cli.py`

**文件位置**：`run_cli.py` 第151-163行

**添加内容**：
- 验证 `<vid_patch>` token 数量与特征数量一致
- 如果不一致，抛出清晰的错误信息

**文件位置**：`run_cli.py` 第211-218行

**添加内容**：
- 详细的日志输出，显示视频特征的形状
- 确保特征形状正确：`[batch_size, num_frames, feature_dim]`

## 🔍 验证方法

### 1. 检查日志输出

运行推理时，应该看到以下日志：

```
✅ Token count matches: 16 tokens == 16 features
✅ Video features shape for model: torch.Size([1, 16, 1024])
   - Batch size: 1
   - Num frames/tokens: 16 (must match 16 <vid_patch> tokens)
   - Feature dim: 1024
```

### 2. 检查模型输出

修复后，模型应该：
- 能够看到视频内容
- 输出的文本与视频相关
- 不再"胡说八道"

### 3. 如果仍然有问题

检查以下几点：
1. **特征数量**：确保 `video_spatio_temporal_features.shape[0] == 16`（或训练时使用的帧数）
2. **Token 数量**：确保 prompt 中恰好有 16 个 `<vid_patch>` tokens（或训练时使用的数量）
3. **Prompt 格式**：确保 prompt 格式与训练时完全一致
4. **模型加载**：确保加载的是正确训练的模型checkpoint

## 📝 相关文件

- `video_chatgpt/model/video_chatgpt.py` - 模型 forward 方法修复
- `run_cli.py` - 推理脚本验证和日志增强
- `docs/inference_method_comparison.md` - 推理方法对比分析

## 🎯 总结

这两个修复解决了推理时模型"胡说八道"的根本原因：

1. **问题①修复**：确保视频特征在推理时被正确处理，不会因为条件判断被跳过
2. **问题②修复**：确保 `<vid_patch>` token 数量与特征数量一致，避免模型跳过视频分支

修复后，模型应该能够正确使用视频特征进行推理，输出与视频内容相关的文本。

