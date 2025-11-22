import os

file_path = "video_chatgpt/eval/model_utils.py"

with open(file_path, "r") as f:
    content = f.read()

# --- 修复 1: 开启 4-bit 量化 (防止 OOM) ---
# 原代码是用 float16 加载的，我们要改成 load_in_4bit
old_code = "torch_dtype=torch.float16,"
new_code = "load_in_4bit=True, device_map='auto',"

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ 已开启 4-bit 量化 (防止显存溢出)")
else:
    print("⚠️ 警告: 未找到 float16 定义，可能已经修改过？跳过此步。")

# --- 修复 2: 调整 Embedding 大小 (修复 RuntimeError) ---
# 我们要在模型加载后，立刻 resize
search_str = "use_cache=True,\n    )"
replace_str = "use_cache=True,\n    )\n    # Fix for embedding size mismatch\n    model.resize_token_embeddings(32006)"

if "model.resize_token_embeddings" not in content:
    if search_str in content:
        content = content.replace(search_str, replace_str)
        print("✅ 已添加 resize_token_embeddings(32006) (修复尺寸不匹配)")
    else:
        # 尝试另一种格式匹配 (以防代码格式不同)
        search_str_2 = "use_cache=True)"
        replace_str_2 = "use_cache=True)\n    model.resize_token_embeddings(32006)"
        if search_str_2 in content:
            content = content.replace(search_str_2, replace_str_2)
            print("✅ 已添加 resize_token_embeddings(32006) (修复尺寸不匹配)")
        else:
            print("❌ 错误: 无法定位插入点，请手动修改文件！")
else:
    print("ℹ️ 看起来 resize 代码已经存在了。")

with open(file_path, "w") as f:
    f.write(content)

print("🎉 文件修改完成！")
