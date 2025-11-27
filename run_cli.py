import torch
import argparse
import os
import sys

# --- 引用路径修正 ---
from video_chatgpt.eval.model_utils import initialize_model, load_video
# 修正点：去掉了 .model 中间层
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.inference import get_spatio_temporal_features_torch

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

    # 3. 图像预处理和特征提取
    try:
        # 预处理视频帧
        image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.half().to(model.device)
        
        # 使用 vision_tower 提取视频特征
        with torch.no_grad():
            image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # Use second to last layer as in LLaVA
        
        # 生成时空特征
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)
        
    except Exception as e:
        print(f"❌ Error during video preprocessing: {e}")
        return

    # 4. 构建对话 Prompt
    conv_mode = "video-chatgpt_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    # 准备问题字符串（包含 video tokens）
    DEFAULT_VID_START_TOKEN = "<vid_start>"
    DEFAULT_VID_END_TOKEN = "<vid_end>"
    DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
    
    if model.get_model().vision_config.use_vid_start_end:
        qs = args.question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = args.question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
        
    conv.append_message(roles[0], qs)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    # 5. Tokenize & 推理
    print(f"🤖 Asking: {args.question}")
    
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 6. 运行模型推理
    # 注意：PEFT 包装后的模型要求所有参数都是关键字参数
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,  # 改为关键字参数
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
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
