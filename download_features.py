import os
# 开启国内镜像加速 (必备！)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
import zipfile

print("🚀 正在检查并下载视频特征数据 (Video Features)...")
print("这可能需要一点时间 (预计 10GB+)...")

try:
    # 尝试下载 video_features 文件夹或 zip 包
    # 我们放宽过滤条件，确保能抓到特征文件
    local_path = snapshot_download(
        repo_id="MBZUAI/VideoInstruct-100K",
        repo_type="dataset",
        local_dir="./data/VideoInstruct-100K",
        local_dir_use_symlinks=False,
        resume_download=True,
        # 这里我们指定下载 zip 或者 pkl 文件
        allow_patterns=["*.zip", "video_features/*", "*.pkl"]
    )
    print(f"✅ 下载完成！数据保存在: {local_path}")

    # 检查是否需要解压
    zip_path = os.path.join(local_path, "video_features.zip")
    if os.path.exists(zip_path):
        print("📦 检测到压缩包，正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
        print("✅ 解压完成！")
        
except Exception as e:
    print(f"❌ 下载出错: {e}")

print("\n🔍 请运行 'ls -lh data/VideoInstruct-100K/' 检查结果。")
