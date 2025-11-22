import os

file_path = "video_chatgpt/eval/model_utils.py"

with open(file_path, "r") as f:
    content = f.read()

# 1. 开启 4-bit 量化 (替换 torch.float16)
# 原代码: torch_dtype=torch.float16,
# 新代码: load_in_4bit=True,
if "torch_dtype=torch.float16" in content:
    content = content.replace("torch_dtype=torch.float16", "load_in_4bit=True")
    print("✅ 已开启 4-bit 量化")
else:
    print("⚠️ 未找到 float16 定义，可能无需修改")

# 2. 插入 Resize 代码 (在加载权重之前)
# 寻找特定的一行代码作为锚点
target_line = "if projection_path:"
insert_code = """
    # [Fix] Resize to 32006 before loading to match checkpoint
    model.resize_token_embeddings(32006)
"""

if "model.resize_token_embeddings(32006)" not in content:
    if target_line in content:
        # 在 "if projection_path:" 这一行的前面插入
        content = content.replace(target_line, insert_code + "\n    " + target_line)
        print("✅ 已插入 Resize 扩容代码")
    else:
        print("❌ 错误：找不到插入点 'if projection_path:'")
else:
    print("ℹ️ Resize 代码已存在")

# 3. 写入文件
with open(file_path, "w") as f:
    f.write(content)

print("🎉 model_utils.py 修复完成！所有原函数（如 load_video）均已保留。")
