import json
import os
import random
import subprocess
from tqdm import tqdm

# 配置
JSON_PATH = "data/VideoInstruct-100K/VideoInstruct100K.json"
VIDEO_DIR = "data/mini_dataset/videos"
FEATURES_DIR = "data/mini_dataset/features"
MINI_JSON_PATH = "data/mini_dataset/mini_train.json"
TARGET_COUNT = 20  # 我们只需要20个样本做测试

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

print(f"📖 读取原始标注: {JSON_PATH}")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

print(f"🔍 原始数据量: {len(data)}")
random.shuffle(data) # 打乱顺序

collected_data = []
count = 0

print("🚀 开始构建迷你数据集...")
for item in tqdm(data):
    if count >= TARGET_COUNT:
        break
        
    video_id = item['video_id']
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    
    # 尝试下载视频 (ActivityNet 很多视频失效了，所以要试)
    # 视频URL通常是 https://www.youtube.com/watch?v={video_id}
    # 注意：VideoInstruct的ID通常就是YT ID，但也有些前缀
    yt_id = video_id
    if video_id.startswith('v_'): #有些ID是v_开头
        yt_id = video_id[2:]
        
    url = f"https://www.youtube.com/watch?v={yt_id}"
    
    if not os.path.exists(video_path):
        try:
            # 使用 yt-dlp 下载，限制高度为 224 以省流量
            cmd = [
                "yt-dlp", 
                "-f", "best[height<=360]", 
                "-o", video_path, 
                url,
                "--no-playlist",
                "--quiet"
            ]
            subprocess.run(cmd, check=True, timeout=60)
        except:
            continue # 下载失败就跳过

    if os.path.exists(video_path):
        # 下载成功，保留这条数据
        collected_data.append(item)
        count += 1
        print(f"✅ 已收集: {count}/{TARGET_COUNT} ({video_id})")

# 保存迷你 JSON
with open(MINI_JSON_PATH, 'w') as f:
    json.dump(collected_data, f, indent=2)

print(f"\n🎉 迷你数据集构建完成！")
print(f"📁 视频路径: {VIDEO_DIR}")
print(f"📄 标注文件: {MINI_JSON_PATH}")
