import os

file_path = "video_chatgpt/model/video_chatgpt.py"

with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
fixed = False

# 目标：找到 VideoChatGPTLlamaForCausalLM 类的 forward 函数定义
# 原代码通常是: def forward(self, input_ids, attention_mask=None, ...
target_start = "def forward(self, input_ids, attention_mask=None,"

for line in lines:
    if target_start in line and not fixed:
        # 替换为：显式包含 images=None
        print("✅ 找到 forward 函数定义，正在添加 images 参数...")
        new_line = line.replace(
            "def forward(self, input_ids, attention_mask=None,",
            "def forward(self, input_ids, images=None, attention_mask=None,"
        )
        new_lines.append(new_line)
        fixed = True
    else:
        new_lines.append(line)

if fixed:
    with open(file_path, "w") as f:
        f.writelines(new_lines)
    print("🎉 修复成功！forward 函数现在显式接受 images 参数了。")
else:
    print("⚠️ 未找到目标行，可能代码格式有变？")
    # 打印出可能的位置供检查
    for i, line in enumerate(lines):
        if "def forward" in line:
            print(f"Line {i+1}: {line.strip()}")

