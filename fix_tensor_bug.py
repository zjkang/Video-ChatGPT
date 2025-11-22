import os

file_path = "video_chatgpt/model/video_chatgpt.py"

# 读取文件
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed = False

for line in lines:
    # 定位到出错的那一行之前的变量赋值
    # 原代码通常是: cur_input_ids = input_ids[i]
    if "cur_input_ids = input_ids[i]" in line:
        new_lines.append(line)
        
        # --- 插入修复代码 ---
        # 获取当前行的缩进
        indent = line[:line.find("cur_input_ids")]
        
        # 强制转换为 Tensor，确保后续的 .sum() 能正常工作
        # 同时确保它在正确的设备(GPU)上
        new_lines.append(f"{indent}if not isinstance(cur_input_ids, torch.Tensor):\n")
        new_lines.append(f"{indent}    cur_input_ids = torch.tensor(cur_input_ids, device=self.device)\n")
        
        print("✅ 已插入 Tensor 强制转换代码")
        fixed = True
    else:
        new_lines.append(line)

if fixed:
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("🎉 修复完成！现在模型能正确识别视频 Token 了。")
else:
    print("⚠️ 未找到目标代码行，可能文件格式不同？")

