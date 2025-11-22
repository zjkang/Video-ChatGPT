import json
import os
import shutil
import random

# 配置
JSON_PATH = "data/VideoInstruct-100K/VideoInstruct100K.json"
VIDEO_DIR = "data/mini_dataset/videos"
MINI_JSON_PATH = "data/mini_dataset/mini_train.json"
TARGET_COUNT = 20

# 1. 准备种子视频
seed_video = "test.mp4"
if not os.path.exists(seed_video):
    if os.path.exists("sample_2.mp4"):
        seed_video = "sample_2.mp4"
    else:
        print("正在下载种子视频...")
        os.system("wget https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4 -O test.mp4")
        seed_video = "test.mp4"

print(f"🌱 使用种子视频: {seed_video}")
os.makedirs(VIDEO_DIR, exist_ok=True)

# 2. 读取原始 JSON
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# 3. 生成伪造数据
fake_data = []
print("🚀 正在生成 20 个分身...")

for i in range(TARGET_COUNT):
    # 借用原始数据的问答，但把 ID 改成 video_0, video_1...
    item = data[i].copy()
    fake_id = f"video_{i}"
    item['video_id'] = fake_id
    
    # 复制视频文件
    dst_path = os.path.join(VIDEO_DIR, f"{fake_id}.mp4")
    shutil.copy(seed_video, dst_path)
    
    fake_data.append(item)

# 4. 保存 JSON
with open(MINI_JSON_PATH, 'w') as f:
    json.dump(fake_data, f, indent=2)

print(f"✅ 成功生成 20 个训练样本！")
print(f"📄 标注文件: {MINI_JSON_PATH}")
