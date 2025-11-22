import os

file_path = "video_chatgpt/eval/model_utils.py"

# 1. 读取原文件
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed_resize = False
fixed_4bit = False

for line in lines:
    # --- 修复 A: 开启 4-bit 量化 ---
    if "torch_dtype=torch.float16" in line:
        # 替换为 4-bit 配置
        line = line.replace("torch_dtype=torch.float16", "load_in_4bit=True")
        fixed_4bit = True
    
    # --- 修复 B: 插入扩容代码 ---
    # 寻找加载权重的那一行
    if "model.load_state_dict" in line and "projection_path" in line:
        if not fixed_resize:
            # 获取当前行的缩进 (空格)
            indent = line[:line.find("model.load_state_dict")]
            
            # 插入扩容代码 (带注释)
            new_lines.append(f"{indent}# [Fix] Resize embeddings to match checkpoint (32006)\n")
            new_lines.append(f"{indent}model.resize_token_embeddings(32006)\n")
            
            fixed_resize = True
    
    new_lines.append(line)

# 2. 写回文件
with open(file_path, "w") as f:
    f.writelines(new_lines)

print(f"修复完成！")
print(f"- 4-bit 量化开启: {'✅' if fixed_4bit else '❌ (未找到目标代码)'}")
print(f"- 词表扩容插入: {'✅' if fixed_resize else '❌ (未找到目标代码)'}")
