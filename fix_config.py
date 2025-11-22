import json
import os

# 你的 Llama-2 配置文件路径
config_path = "./checkpoints/Llama-2-7b-chat-hf/config.json"

if os.path.exists(config_path):
    print(f"正在修复配置文件: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # --- 注入缺失的关键配置 ---
    # 告诉代码：请使用 OpenAI 的 CLIP 模型作为视觉编码器
    config["mm_vision_tower"] = "openai/clip-vit-large-patch14"
    config["mm_projector_type"] = "linear"
    config["model_type"] = "video_chatgpt_llama" # 这一步也很关键，把身份改对
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print("✅ 修复完成！现在代码知道该用哪个 Vision Tower 了。")
else:
    print(f"❌ 找不到文件: {config_path}，请检查路径。")
