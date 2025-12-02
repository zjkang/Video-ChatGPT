import torch
import argparse
import os
import sys

# --- 引用路径修正 ---
from video_chatgpt.eval.model_utils import initialize_model, load_video
# 修正点：去掉了 .model 中间层
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.inference import get_temporal_features_torch

def main(args):
    # 1. 加载模型
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)

    # 2. 加载视频
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return

    print(f"🎬 Processing video: {args.video_path}")
    
    # 确定要加载的帧数（与训练时一致）
    # 如果指定了 num_frames，直接从视频采样该数量的帧
    # 如果不指定，使用默认值16（与训练时的默认值一致）
    num_frames_to_load = args.num_frames if args.num_frames is not None else 16
    print(f"📹 Loading {num_frames_to_load} frames from video (consistent with training)")
    
    video_frames = load_video(args.video_path, num_frames=num_frames_to_load)
    
    if video_frames is None:
        print("❌ Error: Failed to load video frames.")
        return

    # 3. 图像预处理和特征提取（与训练时完全一致的流程）
    try:
        # 预处理视频帧
        image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.half().to(model.device)
        
        # 使用 vision_tower 提取视频特征（与训练时一致）
        with torch.no_grad():
            image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
            # 使用倒数第二层（与训练时一致，LLaVA的做法）
            frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # [num_frames, 256, 1024]
            # 去掉CLS token，保留256个空间patch
        
        # 转换为temporal features（与训练时完全一致的处理方式）
        # 对空间维度（256个patch）做平均，得到每帧的全局特征
        temporal_features = torch.mean(frame_features, dim=1)  # [num_frames, 1024]
        
        # 如果指定了target_frames且与当前帧数不同，进行采样/填充
        # 否则直接使用（已经与训练时一致）
        if args.num_frames is not None and temporal_features.shape[0] != args.num_frames:
            # 这种情况应该很少发生，因为我们已经从视频采样了指定数量的帧
            video_spatio_temporal_features = get_temporal_features_torch(
                frame_features, 
                target_frames=args.num_frames
            )
        else:
            video_spatio_temporal_features = temporal_features.half()
        
        # 使用实际的特征长度（与训练时的逻辑一致）
        actual_video_token_len = video_spatio_temporal_features.shape[0]
        print(f"📊 Generated {actual_video_token_len} temporal tokens (shape: {video_spatio_temporal_features.shape}, consistent with training)")
        
    except Exception as e:
        print(f"❌ Error during video preprocessing: {e}")
        return

    # 4. 构建对话 Prompt（与训练时完全一致的格式）
    # 训练时格式: "Human: <video> {video_tokens} {q}\nAssistant: {a}</s>"
    # 推理时使用相同的格式，但不包含答案部分
    DEFAULT_VID_START_TOKEN = "<vid_start>"
    DEFAULT_VID_END_TOKEN = "<vid_end>"
    DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
    DEFAULT_VIDEO_TOKEN = "<video>"
    
    # 使用空格分隔，确保 tokenizer 能正确识别为多个 token（与训练一致）
    video_tokens = " ".join([DEFAULT_VIDEO_PATCH_TOKEN] * actual_video_token_len)
    
    # 构建与训练时完全一致的 prompt 格式
    if model.get_model().vision_config.use_vid_start_end:
        # 如果使用 vid_start_end，格式略有不同
        qs = args.question + '\n' + DEFAULT_VID_START_TOKEN + video_tokens + DEFAULT_VID_END_TOKEN
        prompt = f"Human: <video> {qs}\nAssistant:"
    else:
        # 与训练时完全一致的格式: "Human: <video> {video_tokens} {q}\nAssistant:"
        prompt = f"Human: <video> {video_tokens} {args.question}\nAssistant:"

    # 5. Tokenize & 推理
    print(f"🤖 Asking: {args.question}")
    
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
    # 调试信息：检查 prompt 中的 vid_patch token 数量
    vid_patch_token_id = tokenizer.convert_tokens_to_ids("<vid_patch>")
    num_vid_patch_tokens = (input_ids == vid_patch_token_id).sum().item()
    print(f"🔍 Debug: Found {num_vid_patch_tokens} <vid_patch> tokens in prompt (expected {actual_video_token_len})")
    
    # 使用与训练时一致的停止标记 "</s>"
    stop_str = "</s>"
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 6. 运行模型推理
    # 注意：PEFT 包装后的模型要求所有参数都是关键字参数
    print("🚀 Generating response...")
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

    # 7. 解码输出
    input_token_len = input_ids.shape[1]
    output_token_len = output_ids.shape[1] - input_token_len
    print(f"🔍 Debug: Generated {output_token_len} new tokens")
    
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
    parser.add_argument("--num_frames", type=int, default=None, 
                        help="Number of frames to use (must match training). If None, will use actual frame count from video.")
    args = parser.parse_args()
    main(args)
