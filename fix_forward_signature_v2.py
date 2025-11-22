import os

file_path = "video_chatgpt/model/video_chatgpt.py"

print(f"正在读取文件: {file_path}")
with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed = False
inside_forward = False

for i, line in enumerate(lines):
    # 1. 找到 Line 180 附近的 forward 定义
    if "def forward(" in line and "VideoChatGPTLlamaForCausalLM" not in line: 
        # 注意：我们假设这是类的方法，且之前已经定位到了大概位置
        inside_forward = True
    
    # 2. 在 forward 参数列表中找到 input_ids
    if inside_forward and "input_ids" in line and "images" not in line:
        print(f"✅ 在第 {i+1} 行找到 input_ids，正在插入 images 参数...")
        
        # 在 input_ids 后面追加 images=None
        # 假设行内容类似: input_ids: torch.LongTensor = None,
        # 我们替换为: input_ids: torch.LongTensor = None, images=None,
        
        # 简单的字符串替换：在行末的逗号前，或者直接追加
        if "," in line:
            new_line = line.replace(",", ", images=None,")
        else:
            new_line = line.rstrip() + ", images=None,\n"
            
        new_lines.append(new_line)
        fixed = True
        inside_forward = False # 任务完成，退出标记
    else:
        new_lines.append(line)
        # 如果遇到右括号，说明参数列表结束了
        if ")" in line:
            inside_forward = False

if fixed:
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("🎉 修复成功！已将 images=None 加入 forward 参数列表。")
else:
    print("❌ 依然未修复。正在打印第 180 行附近的内容供手动检查：")
    start = 178
    end = 185
    for j in range(start, min(len(lines), end)):
        print(f"Line {j+1}: {lines[j].strip()}")

