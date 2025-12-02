# 训练和推理的特征提取流程对比

## 📊 训练时的特征提取流程

### 步骤1: 预提取特征（训练前完成）

```python
# scripts/export_video_instruct_features.py 或 extract_features.py

# 1. 从视频加载帧（均匀采样）
video_frames = load_video(video_path, num_frames=16)  # 采样16帧
# 返回: [16, H, W, C] uint8

# 2. CLIP预处理
image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
# 形状: [16, 3, 224, 224]

# 3. CLIP特征提取
outputs = vision_tower(image_tensor, output_hidden_states=True)
features = outputs.hidden_states[-2]  # [16, 257, 1024]
# 使用倒数第二层（与LLaVA一致）

# 4. 转换为temporal features
features = features[:, 1:]  # 去掉CLS token: [16, 256, 1024]
temporal_features = features.mean(dim=1)  # 对空间维度平均: [16, 1024]

# 5. 保存为.pkl文件
# 最终格式: (16, 1024) float32
```

### 步骤2: 训练时加载特征

```python
# train_mem.py

# 直接从.pkl文件加载已经提取好的特征
with open(pkl_path, 'rb') as f:
    video_features = pickle.load(f)  # [16, 1024]

# 直接使用，无需再处理
num_video_tokens = video_features.shape[0]  # 16
```

---

## 🔍 推理时的特征提取流程（当前实现）

### 问题：流程不一致

```python
# run_cli.py

# 1. 加载视频帧
load_frames = args.num_frames if args.num_frames is not None else 100
video_frames = load_video(args.video_path, num_frames=load_frames)
# 如果num_frames=None，会加载100帧（浪费！）

# 2. CLIP特征提取
image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [t, 256, 1024]
# t可能是100（如果load_frames=100）

# 3. 转换为temporal features
video_spatio_temporal_features = get_temporal_features_torch(
    frame_features, 
    target_frames=args.num_frames  # 如果指定16，会采样/填充到16
)
# 最终: [16, 1024]
```

### ⚠️ 问题分析

1. **效率问题**：如果指定`--num_frames 16`，但`load_video`加载了100帧，然后`get_temporal_features_torch`再采样到16帧，浪费了计算资源。

2. **一致性问题**：训练时直接从视频采样16帧，推理时先采样100帧再采样到16帧，可能得到不同的帧序列。

---

## ✅ 正确的推理流程（应该与训练一致）

```python
# 1. 从视频采样指定数量的帧（与训练时一致）
num_frames = args.num_frames if args.num_frames is not None else 16
video_frames = load_video(args.video_path, num_frames=num_frames)
# 直接采样16帧，而不是100帧

# 2. CLIP特征提取（与训练时一致）
image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [16, 256, 1024]

# 3. 转换为temporal features（与训练时一致）
temporal_features = frame_features.mean(dim=1)  # [16, 1024]
# 不需要再采样/填充，因为已经是16帧了
```

---

## 🔧 需要修复的地方

1. **`run_cli.py`**: 确保`load_video`使用正确的`num_frames`参数
2. **`get_temporal_features_torch`**: 如果输入帧数已经等于`target_frames`，不需要再处理

