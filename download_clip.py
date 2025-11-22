import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("🚀 正在下载 CLIP 视觉模型 (仅 PyTorch 权重)...")

snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    resume_download=True,
    # 👇 只下载 pytorch 权重和配置文件，忽略 TF/Flax
    allow_patterns=["config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt", "pytorch_model.bin"]
)

print("✅ CLIP 下载完成！")
