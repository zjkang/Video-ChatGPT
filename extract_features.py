import os
import torch
import pickle
from tqdm import tqdm
# 复用我们昨天修好的 load_video 函数
from video_chatgpt.eval.model_utils import load_video
from transformers import CLIPVisionModel, CLIPImageProcessor

# 1. 配置路径
VIDEO_DIR = "data/mini_dataset/videos"
FEATURES_DIR = "data/mini_dataset/features"

# 自动创建目录
os.makedirs(FEATURES_DIR, exist_ok=True)

print("🚀 Loading CLIP Vision Tower...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# 使用我们已经下载好的 CLIP 模型
vision_tower_name = "openai/clip-vit-large-patch14"

# 加载处理器和模型 (使用 FP16 提速)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name, torch_dtype=torch.float16, use_safetensors=False)
vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=False).to(device)
vision_tower.eval()

video_files = os.listdir(VIDEO_DIR)
print(f"📸 发现 {len(video_files)} 个视频，开始提取特征...")

for file in tqdm(video_files):
    if not file.endswith(".mp4"): continue
    
    video_id = file.split(".")[0]
    save_path = os.path.join(FEATURES_DIR, f"{video_id}.pkl")
    
    # 如果已经提取过，就跳过
    if os.path.exists(save_path):
        continue
        
    video_path = os.path.join(VIDEO_DIR, file)
    
    # 2. 加载视频 (默认采样 100 帧)
    try:
        video_frames = load_video(video_path, num_frames=100)
    except Exception as e:
        print(f"⚠️ 无法加载 {file}: {e}")
        continue

    if video_frames is None:
        print(f"❌ 视频加载为空: {file}")
        continue
        
    # 3. 图像预处理
    video_process = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    video_process = video_process.half().to(device)
    
    # 4. 提取特征
    with torch.no_grad():
        outputs = vision_tower(video_process, output_hidden_states=True)
        # 使用倒数第二层（与推理一致，LLaVA的做法）
        features = outputs.hidden_states[-2]  # [100, 257, 1024]
        
        # 关键步骤：对空间维度 (257) 做平均池化 -> 得到 [100, 1024]
        # 忽略第一个 CLS token，对剩下的 256 个 patch 做平均
        features = features[:, 1:] 
        features = features.mean(dim=1) # Shape: [100, 1024]

        # 转回 CPU 保存
        features = features.cpu().numpy()

    # 5. 保存为 pkl
    with open(save_path, "wb") as f:
        pickle.dump(features, f)

print("✅ 特征提取全部完成！你现在可以去睡觉了！😴")
