import os
import torch

file_path = "video_chatgpt/model/video_chatgpt.py"

print(f"正在读取文件: {file_path}")
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed = False

# 这是报错的那行代码的关键特征
target_signature = "self.vision_config.vid_patch_token).sum() == 0"

for i, line in enumerate(lines):
    if target_signature in line:
        print(f"✅ 在第 {i+1} 行找到了目标代码！正在植入修复补丁...")
        
        # 1. 计算这一行的缩进 (Indentation)
        # 我们通过提取开头的空格来保证格式对齐
        indent = line[:line.find("if")]
        
        # 2. 插入强制类型转换代码
        # 这段代码会确保 cur_input_ids 变成了 Tensor，这样 .sum() 就不会报错了
        patch_code = [
            f"{indent}# [Fix] Force convert list to tensor to avoid AttributeError\n",
            f"{indent}if not isinstance(cur_input_ids, torch.Tensor):\n",
            f"{indent}    cur_input_ids = torch.tensor(cur_input_ids, device=self.device)\n"
        ]
        
        new_lines.extend(patch_code)
        new_lines.append(line) # 别忘了把原来那行写回去
        fixed = True
    else:
        new_lines.append(line)

if fixed:
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("🎉 修复成功！文件已重写。")
else:
    print("❌ 严重错误：依然没找到目标行。")
    # 如果还找不到，我打印出第90-100行看看是啥
    print("--- Debug: File Content around line 94 ---")
    start = max(0, 85)
    end = min(len(lines), 105)
    for j in range(start, end):
        print(f"{j+1}: {lines[j].strip()}")

