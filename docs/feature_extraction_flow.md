# 特征提取流程详解

## 🎯 核心问题

**训练时**：输入是已经提取好的特征（从.pkl文件加载，格式为`[16, 1024]`）  
**推理时**：需要从视频中实时提取特征

## 📊 训练时的特征提取流程

### 预提取阶段（训练前完成）

```python
# 1. 从视频采样16帧（均匀采样）
video_frames = load_video(video_path, num_frames=16)
# 返回: [16, H, W, C] uint8 numpy array

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

### 训练时加载

```python
# 直接从.pkl文件加载
with open(pkl_path, 'rb') as f:
    video_features = pickle.load(f)  # [16, 1024]
# 直接使用，无需再处理
```

---

## 🔍 推理时的特征提取流程（修复后）

### 与训练时完全一致的流程

```python
# 1. 从视频采样指定数量的帧（与训练时一致）
num_frames_to_load = args.num_frames if args.num_frames is not None else 16
video_frames = load_video(args.video_path, num_frames=num_frames_to_load)
# 如果指定 --num_frames 16，直接采样16帧（不是100帧！）

# 2. CLIP预处理（与训练时一致）
image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
image_tensor = image_tensor.half().to(model.device)

# 3. CLIP特征提取（与训练时一致）
image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [16, 256, 1024]
# 使用倒数第二层，去掉CLS token

# 4. 转换为temporal features（与训练时完全一致）
temporal_features = torch.mean(frame_features, dim=1)  # [16, 1024]
# 对空间维度（256个patch）做平均，得到每帧的全局特征

# 5. 直接使用（已经是16帧，无需再采样/填充）
video_spatio_temporal_features = temporal_features.half()  # [16, 1024]
```

---

## ✅ 关键修复点

### 修复前的问题

1. **效率问题**：
   - 如果指定`--num_frames 16`，但`load_video`加载了100帧
   - 然后`get_temporal_features_torch`再采样到16帧
   - **浪费了84帧的CLIP计算**

2. **一致性问题**：
   - 训练时直接从视频采样16帧
   - 推理时先采样100帧再采样到16帧
   - **可能得到不同的帧序列**

### 修复后的改进

1. **直接采样目标帧数**：
   - 如果指定`--num_frames 16`，直接从视频采样16帧
   - 与训练时的采样方式完全一致

2. **特征提取流程一致**：
   - 使用相同的CLIP层（`hidden_states[-2]`）
   - 使用相同的处理方式（去掉CLS token，空间维度平均）
   - 最终得到相同格式的特征`[16, 1024]`

---

## 📝 使用示例

### 训练时（使用16帧特征）

```bash
python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    ...
```

### 推理时（使用16帧，与训练一致）

```bash
python run_cli.py \
    --projection_path ./checkpoints/Baseline_T16 \
    --video_path ./sample_2.mp4 \
    --question "Describe the video." \
    --num_frames 16
```

**流程**：
1. 从视频采样16帧
2. CLIP提取特征 → `[16, 256, 1024]`
3. 空间维度平均 → `[16, 1024]`
4. 输入模型（与训练时完全一致）

---

## 🔄 数据流对比

| 阶段 | 训练时 | 推理时（修复后） |
|------|--------|------------------|
| 视频帧 | 预提取时采样16帧 | 实时采样16帧 |
| CLIP特征 | `[16, 256, 1024]` | `[16, 256, 1024]` |
| Temporal特征 | `[16, 1024]` | `[16, 1024]` |
| 输入模型 | `[16, 1024]` | `[16, 1024]` |

**现在训练和推理的特征提取流程完全一致！** ✅

