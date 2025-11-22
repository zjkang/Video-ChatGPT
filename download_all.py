import os

# 1. 设置镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import login, snapshot_download

# --- 任务 A: Video-ChatGPT Adapter (保持不变) ---
print("\n🚀 [1/3] 下载 Adapter ...")
try:
    snapshot_download(
        repo_id="MBZUAI/Video-ChatGPT-7B", 
        local_dir="./checkpoints/Video-ChatGPT-7B-Adapter",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✅ Adapter 完成")
except Exception as e:
    print(f"❌ Adapter 失败: {e}")

# --- 任务 B: Llama-2 底座 (🔥 瘦身版：只下 .bin) ---
print("\n🚀 [2/3] 下载 Llama-2-7b-chat (.bin ONLY) ...")
try:
    snapshot_download(
        repo_id="NousResearch/Llama-2-7b-chat-hf", 
        local_dir="./checkpoints/Llama-2-7b-chat-hf",
        local_dir_use_symlinks=False,
        resume_download=True,
        # 👇 只要这些文件，总量约 13.5GB，不要 40GB！
        allow_patterns=["*.json", "*.model", "*.bin", "tokenizer*"] 
    )
    print("✅ Llama-2 底座下载完成 (已过滤多余文件)！")
except Exception as e:
    print(f"❌ Llama-2 下载失败: {e}")

# --- 任务 C: 数据 (保持不变) ---
print("\n🚀 [3/3] 下载数据 ...")
try:
    snapshot_download(
        repo_id="MBZUAI/VideoInstruct-100K",
        local_dir="./data/VideoInstruct-100K",
        local_dir_use_symlinks=False,
        repo_type="dataset",
        resume_download=True
    )
    print("✅ 数据完成")
except Exception as e:
    print(f"❌ 数据失败: {e}")

print("\n🎉 瘦身版下载任务开始！")
