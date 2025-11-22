import os

file_path = "video_chatgpt/eval/model_utils.py"

with open(file_path, "r") as f:
    content = f.read()

# 1. 修复 ImageProcessor (它不能用 4-bit)
bad_code_1 = "image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, load_in_4bit=True)"
good_code_1 = "image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)"

# 2. 修复 Vision Tower (它比较小，用 float16 精度更高，且防止兼容性问题)
bad_code_2 = "vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, load_in_4bit=True,"
good_code_2 = "vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,"

if bad_code_1 in content:
    content = content.replace(bad_code_1, good_code_1)
    print("✅ 已修复 ImageProcessor (回滚为 float16)")

if bad_code_2 in content:
    content = content.replace(bad_code_2, good_code_2)
    print("✅ 已修复 Vision Tower (回滚为 float16)")

with open(file_path, "w") as f:
    f.write(content)

print("🎉 代码现在完美了！")
