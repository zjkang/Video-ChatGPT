import os

file_path = "video_chatgpt/model/video_chatgpt.py"

with open(file_path, "r") as f:
    content = f.read()

# 目标：报错的那行原始代码
bad_code = "(cur_input_ids == self.vision_config.vid_patch_token).sum()"

# 替换：原地强转 Tensor
good_code = "(torch.as_tensor(cur_input_ids, device=self.device) == self.vision_config.vid_patch_token).sum()"

if bad_code in content:
    new_content = content.replace(bad_code, good_code)
    with open(file_path, "w") as f:
        f.write(new_content)
    print("🎉 修复成功！已将判断逻辑改为原地强制 Tensor 转换。")
else:
    print("⚠️ 未找到目标代码。可能已经被修改过？")
    # 打印一下文件看看情况
    # print(content)

