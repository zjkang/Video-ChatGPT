import os

file_path = "video_chatgpt/eval/model_utils.py"

with open(file_path, "r") as f:
    content = f.read()

# 我们要找到加载 Tokenizer 的地方，因为我们需要 Tokenizer 来查 ID
target_line = "tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)"

# 插入的代码：查询 <vid_patch> 的 ID 并注入配置
insert_code = """
    # [Fix] Inject missing video config params
    # Video-ChatGPT 默认使用 <vid_patch> (index=32000 左右)
    # 如果 tokenizer 里找不到，为了防止报错，我们暂时设为一个不可能的 token (如 0) 或 pad_token
    if "<vid_patch>" in tokenizer.vocab:
        vid_patch_token_idx = tokenizer.vocab["<vid_patch>"]
    else:
        # 如果字典没扩充，尝试使用默认扩充后的位置
        vid_patch_token_idx = 32005 
    
    print(f"Injecting vid_patch_token = {vid_patch_token_idx} into config")
    model.config.vid_patch_token = vid_patch_token_idx
    model.config.use_sim_mask = False 
    
    # 确保 vision_config 也同步
    if hasattr(model.get_model(), "vision_config"):
        model.get_model().vision_config.vid_patch_token = vid_patch_token_idx
"""

if "model.config.vid_patch_token =" not in content:
    if target_line in content:
        # 在加载 tokenizer 之后，模型加载之前插入不太好，得在 model 加载之后插入
        # 我们换个锚点：在 initialize_model 函数的 return 之前插入最稳妥
        
        # 找到 return model, ...
        return_line_idx = content.rfind("return model,")
        if return_line_idx != -1:
            # 插入到 return 之前
            new_content = content[:return_line_idx] + insert_code + "\n    " + content[return_line_idx:]
            
            with open(file_path, "w") as f:
                f.write(new_content)
            print("🎉 修复成功！已注入 vid_patch_token 配置。")
        else:
             print("❌ 找不到 return 语句，无法插入。")
    else:
        print("❌ 找不到 Tokenizer 加载行。")
else:
    print("ℹ️ 配置注入代码貌似已存在。")
