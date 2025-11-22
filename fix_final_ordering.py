import os

file_path = "video_chatgpt/eval/model_utils.py"

with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed = False

for line in lines:
    # 找到加载权重的这一行
    if "model.load_state_dict" in line and "projection_path" in line:
        # 在它之前插入扩容代码
        indent = line[:line.find("model.load_state_dict")] # 保持缩进一致
        print("✅ 找到加载点，正在插入扩容指令...")
        new_lines.append(f"{indent}# Fix: Resize BEFORE loading to prevent mismatch\n")
        new_lines.append(f"{indent}model.resize_token_embeddings(32006)\n")
        new_lines.append(line) # 然后才是加载代码
        fixed = True
    else:
        new_lines.append(line)

if fixed:
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("🎉 修复成功！扩容操作已移动到加载之前。")
else:
    print("⚠️ 未找到目标代码行，可能文件已被修改过？")
    # 打印文件内容以便调试
    # print("".join(lines))

