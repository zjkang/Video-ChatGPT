import os

file_path = "video_chatgpt/model/video_chatgpt.py"

with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed = False

# 目标特征：报错的那行代码里肯定包含这个独特的判断逻辑
target_snippet = "== self.vision_config.vid_patch_token).sum()"

for line in lines:
    if target_snippet in line:
        # 1. 获取当前行的缩进
        # 通常这行是以 if 开头的，我们要拿到 if 前面的空格
        indent = line.split("if")[0]
        
        # 2. 在这行之前插入修复代码
        print("✅ 找到报错行，正在插入修复补丁...")
        new_lines.append(f"{indent}# [Fix] Ensure cur_input_ids is a Tensor\n")
        new_lines.append(f"{indent}if not isinstance(cur_input_ids, torch.Tensor):\n")
        new_lines.append(f"{indent}    cur_input_ids = torch.tensor(cur_input_ids, device=self.device)\n")
        
        # 3. 写入原行
        new_lines.append(line)
        fixed = True
    else:
        new_lines.append(line)

if fixed:
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("🎉 修复完成！Bug 已消除。")
else:
    print("❌ 还是没找到？请联系我，我们用 cat 命令查看文件内容。")

