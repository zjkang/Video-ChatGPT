import torch
import argparse
import os
import sys

# --- 引用路径修正 ---
from video_chatgpt.eval.model_utils import initialize_model, load_video
# 修正点：去掉了 .model 中间层
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria

def main(args):
    # 1. 加载模型
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)

    # 2. 加载视频
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return

    print(f"🎬 Processing video: {args.video_path}")
    video_frames = load_video(args.video_path)
    
    if video_frames is None:
        print("❌ Error: Failed to load video frames.")
        return

    # 3. 图像预处理
    try:
        video_process = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
        video_process = video_process.half().to(model.device)
        tensor = video_process.unsqueeze(0) # [1, T, C, H, W]
    except Exception as e:
        print(f"❌ Error during video preprocessing: {e}")
        return

    # 4. 构建对话 Prompt
    conv_mode = "video-chatgpt_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    prompt_text = args.question
    if "<video>" not in prompt_text:
        prompt_text = f"<video>\n{prompt_text}"
        
    conv.append_message(roles[0], prompt_text)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    # 5. Tokenize & 推理
    print(f"🤖 Asking: {args.question}")
    
    inputs = tokenizer([prompt])
    
    # 这里的 input_ids 已经是 List，会在 model.forward 里被我们的补丁转为 Tensor
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor, 
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    # 6. 解码输出
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    
    print("\n" + "="*30)
    print(f"📝 Answer:\n{outputs}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./checkpoints/Llama-2-7b-chat-hf")
    parser.add_argument("--projection_path", type=str, default="./checkpoints/Video-ChatGPT-7B-Adapter/video_chatgpt-7B.bin")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--question", type=str, default="Describe the video in detail.")
    args = parser.parse_args()
    main(args)
