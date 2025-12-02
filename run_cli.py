import torch
import argparse
import os
import sys
import re
import logging
from datetime import datetime

# --- 引用路径修正 ---
from video_chatgpt.eval.model_utils import initialize_model, load_video
# 修正点：去掉了 .model 中间层
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.inference import get_temporal_features_torch

# 设置日志
def setup_logging(log_file=None):
    """设置日志，同时输出到控制台和文件"""
    if log_file is None:
        # 默认日志文件名：使用时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"inference_{timestamp}.log"
    
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置logging：同时输出到控制台和文件
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),  # 文件输出
            logging.StreamHandler(sys.stdout)  # 控制台输出
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging to file: {log_path}")
    return logger, log_path

def main(args):
    # 0. 设置日志
    logger, log_path = setup_logging(args.log_file)
    logger.info("="*60)
    logger.info("🚀 Starting Video-ChatGPT Inference")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Projection: {args.projection_path}")
    logger.info(f"Video: {args.video_path}")
    logger.info(f"Question: {args.question}")
    logger.info(f"Num frames: {args.num_frames}")
    logger.info("="*60)
    
    # 1. 加载模型
    logger.info("📦 Loading model...")
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)
    logger.info("✅ Model loaded successfully")

    # 2. 加载视频
    if not os.path.exists(args.video_path):
        logger.error(f"❌ Error: Video file not found: {args.video_path}")
        return

    logger.info(f"🎬 Processing video: {args.video_path}")
    
    # 确定要加载的帧数（与训练时一致）
    # 如果指定了 num_frames，直接从视频采样该数量的帧
    # 如果不指定，使用默认值16（与训练时的默认值一致）
    num_frames_to_load = args.num_frames if args.num_frames is not None else 16
    logger.info(f"📹 Loading {num_frames_to_load} frames from video (consistent with training)")
    
    video_frames = load_video(args.video_path, num_frames=num_frames_to_load)
    
    if video_frames is None:
        logger.error("❌ Error: Failed to load video frames.")
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
        logger.info(f"📊 Generated {actual_video_token_len} temporal tokens (shape: {video_spatio_temporal_features.shape}, consistent with training)")
        
    except Exception as e:
        logger.error(f"❌ Error during video preprocessing: {e}", exc_info=True)
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
    logger.info(f"🤖 Asking: {args.question}")
    
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
    # 调试信息：检查 prompt 中的 vid_patch token 数量
    vid_patch_token_id = tokenizer.convert_tokens_to_ids("<vid_patch>")
    num_vid_patch_tokens = (input_ids == vid_patch_token_id).sum().item()
    logger.info(f"🔍 Debug: Found {num_vid_patch_tokens} <vid_patch> tokens in prompt (expected {actual_video_token_len})")
    
    # 【关键修复】验证 token 数量与特征数量必须一致
    if num_vid_patch_tokens != actual_video_token_len:
        logger.error(f"❌ CRITICAL ERROR: Token count mismatch!")
        logger.error(f"   - <vid_patch> tokens in prompt: {num_vid_patch_tokens}")
        logger.error(f"   - Video features count: {actual_video_token_len}")
        logger.error(f"   - This will cause the model to ignore video features!")
        logger.error(f"   - Model will fallback to text-only generation!")
        raise ValueError(
            f"Token count mismatch: {num_vid_patch_tokens} <vid_patch> tokens in prompt "
            f"but {actual_video_token_len} video features. "
            f"They must be equal for video features to be processed correctly."
        )
    else:
        logger.info(f"✅ Token count matches: {num_vid_patch_tokens} tokens == {actual_video_token_len} features")
    
    # 获取 EOS token ID（更可靠的停止方式）
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # 如果 eos_token_id 不存在，尝试从 "</s>" 获取
        stop_str = "</s>"
        eos_tokens = tokenizer(stop_str, add_special_tokens=False, return_tensors=None)
        if isinstance(eos_tokens, dict):
            eos_token_list = eos_tokens.get('input_ids', [])
        else:
            eos_token_list = eos_tokens
        if len(eos_token_list) == 1:
            eos_token_id = eos_token_list[0] if isinstance(eos_token_list[0], int) else eos_token_list[0][0]
            logger.info(f"✅ Found eos_token_id from '</s>': {eos_token_id}")
        else:
            eos_token_id = None
            logger.warning("⚠️  Warning: Could not determine eos_token_id")
    else:
        logger.info(f"✅ Using tokenizer.eos_token_id: {eos_token_id}")
    
    # 使用与训练时一致的停止标记 "</s>"
    stop_str = "</s>"
    keywords = [stop_str]
    
    # 修复 KeywordsStoppingCriteria 的初始化：手动设置 keyword_ids
    # 因为原始的 KeywordsStoppingCriteria 可能无法正确解析 "</s>"
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # 如果 keyword_ids 为空，手动设置（修复 bug）
    if (not hasattr(stopping_criteria, 'keyword_ids') or 
        len(stopping_criteria.keyword_ids) == 0) and eos_token_id is not None:
        # 手动添加 eos_token_id 到 keyword_ids
        stopping_criteria.keyword_ids = [eos_token_id]
        logger.info(f"✅ Fixed stopping_criteria.keyword_ids: {stopping_criteria.keyword_ids}")
    
    # 调试信息：检查停止条件
    logger.info(f"🔍 Debug: eos_token_id={eos_token_id}, stop_str='{stop_str}'")
    if hasattr(stopping_criteria, 'keyword_ids') and len(stopping_criteria.keyword_ids) > 0:
        logger.info(f"🔍 Debug: stopping_criteria.keyword_ids={stopping_criteria.keyword_ids}")
    else:
        logger.warning("⚠️  Warning: stopping_criteria.keyword_ids is empty, token ID detection may fail")

    # 6. 运行模型推理
    # 注意：PEFT 包装后的模型要求所有参数都是关键字参数
    logger.info("🚀 Generating response...")
    
    # 准备生成参数
    # 【关键修复】确保 video_spatio_temporal_features 的形状正确
    # 形状应该是 [batch_size, num_frames, feature_dim]，即 [1, actual_video_token_len, 1024]
    video_features_for_model = video_spatio_temporal_features.unsqueeze(0)  # [1, 16, 1024]
    logger.info(f"✅ Video features shape for model: {video_features_for_model.shape}")
    logger.info(f"   - Batch size: {video_features_for_model.shape[0]}")
    logger.info(f"   - Num frames/tokens: {video_features_for_model.shape[1]} (must match {num_vid_patch_tokens} <vid_patch> tokens)")
    logger.info(f"   - Feature dim: {video_features_for_model.shape[2]}")
    
    generation_kwargs = {
        "input_ids": input_ids,
        "video_spatio_temporal_features": video_features_for_model,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 256,
        "use_cache": True,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "stopping_criteria": [stopping_criteria],
    }
    
    # 确保 eos_token_id 正确设置（这是最关键的停止条件）
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
        logger.info(f"✅ Using eos_token_id={eos_token_id} for generation")
    else:
        logger.warning("⚠️  Warning: eos_token_id is None, generation may not stop properly")
    
    # 设置 pad_token_id
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif eos_token_id is not None:
        generation_kwargs["pad_token_id"] = eos_token_id
    else:
        generation_kwargs["pad_token_id"] = 0
    
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    # 7. 解码输出（不截断，用于分析问题）
    input_token_len = input_ids.shape[1]
    output_token_len = output_ids.shape[1] - input_token_len
    logger.info(f"🔍 Debug: Generated {output_token_len} new tokens")
    
    # 检查是否生成了 EOS token（用于分析问题）
    eos_token_id_check = tokenizer.eos_token_id
    eos_positions = None
    if eos_token_id_check is not None:
        eos_positions = (output_ids[0] == eos_token_id_check).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos_pos = eos_positions[0].item()
            logger.info(f"✅ Found </s> token at position {first_eos_pos} (relative to input start: {first_eos_pos - input_token_len})")
            logger.info(f"   → Model generated </s>, but stopping may have failed")
        else:
            logger.warning("❌ No </s> token found in generated sequence")
            logger.warning("   → This suggests the model may not have learned to use </s> properly")
            logger.warning("   → This could be a training issue")
    
    # 解码完整输出（不跳过特殊token，用于分析）
    outputs_full = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    
    # 清理停止标记
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    # 检测并移除重复文本（后处理）
    import re
    # 检测重复的句子模式（相同的句子重复多次）
    lines = outputs.split('\n')
    if len(lines) > 1:
        # 检查是否有重复的句子
        seen = set()
        unique_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                unique_lines.append(line)
            elif line_stripped in seen:
                # 发现重复，截断到这里
                logger.warning("⚠️  Warning: Detected repetitive sentences, truncating output")
                break
        if len(unique_lines) < len(lines):
            outputs = '\n'.join(unique_lines).strip()
    
    # 如果输出仍然包含大量重复，尝试更激进的截断
    # 检测重复的短语（至少10个字符，重复3次以上）
    if len(outputs) > 100:
        for pattern_len in range(20, 5, -1):  # 从长到短检测
            pattern = outputs[:pattern_len]
            if outputs.count(pattern) >= 3:
                # 找到第一个重复模式的位置
                first_repeat = outputs.find(pattern, pattern_len)
                if first_repeat > 0:
                    outputs = outputs[:first_repeat].strip()
                    logger.warning(f"⚠️  Warning: Detected repetitive pattern (length {pattern_len}), truncated at position {first_repeat}")
                    break
    
    # 显示完整信息用于分析（不截断）
    logger.info("\n" + "="*60)
    logger.info("📊 Analysis Information:")
    logger.info("="*60)
    logger.info(f"Generated tokens: {output_token_len}")
    if eos_positions is not None and len(eos_positions) > 0:
        logger.info(f"EOS token position: {eos_positions[0].item() - input_token_len}")
        logger.warning("⚠️  Warning: Model generated </s> but continued generating (stopping failed)")
    else:
        logger.info("EOS token: Not found")
        logger.warning("⚠️  Warning: Model did not generate </s> token (training issue?)")
    logger.info(f"Output length: {len(outputs)} characters (after deduplication)")
    logger.info(f"Full output (with special tokens, first 500 chars):\n{outputs_full[:500]}")
    if len(outputs_full) > 500:
        logger.info(f"... (truncated for display, total {len(outputs_full)} chars)")
    logger.info("="*60)
    
    # 保存最终答案到日志
    logger.info("\n" + "="*30)
    logger.info(f"📝 Answer:\n{outputs}")
    logger.info("="*30)
    logger.info(f"✅ Inference completed. Log saved to: {log_path}")
    logger.info("="*60 + "\n")
    
    # 同时打印到控制台（保持原有行为）
    print("\n" + "="*30)
    print(f"📝 Answer:\n{outputs}")
    print("="*30 + "\n")
    print(f"💾 Log saved to: {log_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="./checkpoints/Llama-2-7b-chat-hf")
    parser.add_argument("--projection_path", type=str, default="./checkpoints/Video-ChatGPT-7B-Adapter/video_chatgpt-7B.bin")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--question", type=str, default="Describe the video in detail.")
    parser.add_argument("--num_frames", type=int, default=None, 
                        help="Number of frames to use (must match training). If None, will use actual frame count from video.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file name (optional). If not specified, will use timestamp-based name.")
    args = parser.parse_args()
    main(args)
