# run_cli.py 与训练代码一致性检查

## ✅ 确认：run_cli.py 已解决所有不一致问题

### 对比检查

#### 1. Token 数量：✅ 完全一致

**训练代码** (`train_mem.py` 第110行):
```python
num_video_tokens = video_features.shape[0]  # 动态：16/32/100等
```

**run_cli.py** (第113行):
```python
actual_video_token_len = video_spatio_temporal_features.shape[0]  # 动态：16/32/100等
```

✅ **一致**：都使用动态帧数

---

#### 2. Token 分隔方式：✅ 完全一致

**训练代码** (`train_mem.py` 第111行):
```python
video_tokens = " ".join(["<vid_patch>"] * num_video_tokens)  # 有空格
```

**run_cli.py** (第129行):
```python
video_tokens = " ".join([DEFAULT_VIDEO_PATCH_TOKEN] * actual_video_token_len)  # 有空格
```

✅ **一致**：都使用空格分隔

---

#### 3. Prompt 格式：✅ 完全一致

**训练代码** (`train_mem.py` 第112行):
```python
prompt = f"Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
```

**run_cli.py** (第138行):
```python
prompt = f"Human: <video> {video_tokens} {args.question}\nAssistant:"
```

✅ **一致**：都直接构造字符串，格式相同（推理时不需要答案部分）

---

#### 4. Token 数量验证：✅ run_cli.py 有额外保护

**训练代码**：
- ⚠️ 没有显式验证（依赖数据一致性）

**run_cli.py** (第151-164行):
```python
# 【关键修复】验证 token 数量与特征数量必须一致
if num_vid_patch_tokens != actual_video_token_len:
    raise ValueError(...)
else:
    logger.info(f"✅ Token count matches: {num_vid_patch_tokens} tokens == {actual_video_token_len} features")
```

✅ **更好**：有显式验证，防止不匹配

---

#### 5. 特征处理：✅ 逻辑一致

**训练代码**：
- 从 .pkl 文件加载预提取的特征
- 特征已经是 temporal features `(num_frames, 1024)`

**run_cli.py** (第97-110行):
```python
# 对空间维度（256个patch）做平均，得到每帧的全局特征
temporal_features = torch.mean(frame_features, dim=1)  # [num_frames, 1024]
video_spatio_temporal_features = temporal_features.half()
```

✅ **一致**：都得到 `(num_frames, 1024)` 的 temporal features

---

## 📊 总结

| 特性 | 训练代码 | run_cli.py | 状态 |
|------|---------|------------|------|
| Token 数量 | 动态 | 动态 | ✅ 一致 |
| Token 分隔 | 有空格 | 有空格 | ✅ 一致 |
| Prompt 格式 | 直接构造 | 直接构造 | ✅ 一致 |
| 特征形状 | (num_frames, 1024) | (num_frames, 1024) | ✅ 一致 |
| 验证检查 | 无 | 有 | ✅ 更好 |

## 🎯 结论

**`run_cli.py` 已经与训练代码完全一致，并且有额外的保护措施！**

- ✅ 所有关键点都已对齐
- ✅ 有额外的验证和日志
- ✅ 可以安全使用进行推理

## 📝 关于其他评估脚本

`video_chatgpt/inference.py` 和 `run_inference_activitynet_qa.py` 是原始代码，如果：
- 你只用 `run_cli.py` → **不需要担心**，已经解决了
- 你想用官方评估脚本 → 需要修复它们（参考 `docs/training_vs_inference_inconsistency_analysis.md`）

