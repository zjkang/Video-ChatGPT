import argparse
import json
import os
from typing import Optional, List

# Set HuggingFace mirror for faster download in China
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor

try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️  Warning: decord not available. Install with: pip install decord")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VideoInstruct100K videos into frame-level CLIP features [T, 1024]."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store extracted features (.pt files).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (train/validation/test).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to sample per video.",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="MBZUAI/VideoInstruct-100K",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP vision tower to use.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="If set, only process N samples (useful for debugging).",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start processing from this index (useful for batch processing).",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Precision used when saving features.",
    )
    parser.add_argument(
        "--batch_frames",
        type=int,
        default=16,
        help="Number of frames to encode per CLIP forward pass (trade-off memory vs speed).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if the .pt file already exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for CLIP inference (default: "cuda" if available else "cpu").',
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to local JSON file (VideoInstruct100K.json). If provided, will use local videos instead of HF dataset.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing video files (e.g., data/VideoInstruct-100K/videos). Required if --json_path is provided.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sample_frames(video_array: np.ndarray, num_frames: int) -> np.ndarray:
    total_frames = video_array.shape[0]
    if total_frames <= 0:
        raise ValueError("Video has no frames.")
    T = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, num=T, dtype=int)
    return video_array[indices]


@torch.no_grad()
def encode_frames(
    frames: np.ndarray,
    clip_model: CLIPVisionModel,
    clip_processor: CLIPImageProcessor,
    device: torch.device,
    batch_frames: int,
) -> torch.Tensor:
    """
    Args:
        frames: [T, H, W, C] uint8
    Returns:
        Tensor [T, 1024] (float32)
    """
    tensors: List[torch.Tensor] = []
    total = frames.shape[0]
    for start in range(0, total, batch_frames):
        end = min(total, start + batch_frames)
        chunk = frames[start:end]
        inputs = clip_processor(images=list(chunk), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        outputs = clip_model(pixel_values=pixel_values)
        cls_embeds = outputs.last_hidden_state[:, 0, :]  # [chunk, hidden]
        tensors.append(cls_embeds.cpu())
    return torch.cat(tensors, dim=0)


def load_video_from_file(video_path: str, num_frames: int) -> np.ndarray:
    """
    Load video frames from a local video file using decord.
    Returns: [T, H, W, C] uint8 array
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required to load video files. Install with: pip install decord")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Sample frames uniformly
    T = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, num=T, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    
    return frames.astype(np.uint8)


def resolve_video_array(sample_video) -> np.ndarray:
    """
    Handles different HF Video formats.
    """
    if isinstance(sample_video, dict):
        if "frames" in sample_video:
            return sample_video["frames"].astype(np.uint8)
        if "array" in sample_video:
            return sample_video["array"].astype(np.uint8)
        if "bytes" in sample_video:
            raise ValueError("Video sample contains raw bytes; decoding not implemented.")
    return np.array(sample_video, dtype=np.uint8)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    print(f"📁 Output directory: {args.output_dir}")

    # Mode 1: Load from local JSON + video files
    if args.json_path is not None:
        if args.video_dir is None:
            raise ValueError("--video_dir is required when using --json_path")
        if not os.path.exists(args.json_path):
            raise FileNotFoundError(f"JSON file not found: {args.json_path}")
        if not os.path.exists(args.video_dir):
            raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
        
        print(f"📖 Loading from local JSON: {args.json_path}")
        with open(args.json_path, 'r') as f:
            data_list = json.load(f)
        
        total_size = len(data_list)
        start_idx = max(0, args.start_index)
        if start_idx >= total_size:
            print(f"❌ start_index ({start_idx}) >= dataset size ({total_size}). Nothing to process.")
            return
        
        if args.subset_size is not None:
            end_idx = min(start_idx + args.subset_size, total_size)
            data_list = data_list[start_idx:end_idx]
            print(f"✅ Processing samples {start_idx} to {end_idx-1} (total: {len(data_list)} samples)")
        else:
            data_list = data_list[start_idx:]
            print(f"✅ Processing samples {start_idx} to {total_size-1} (total: {len(data_list)} samples)")
        
        use_local_mode = True
    else:
        # Mode 2: Load from HuggingFace dataset
        print(f"📦 Loading from HuggingFace dataset: {args.hf_dataset_name}")
        dataset = load_dataset(args.hf_dataset_name, split=args.split)
        total_size = len(dataset)
        
        start_idx = max(0, args.start_index)
        if start_idx >= total_size:
            print(f"❌ start_index ({start_idx}) >= dataset size ({total_size}). Nothing to process.")
            return
        
        if args.subset_size is not None:
            end_idx = min(start_idx + args.subset_size, total_size)
            dataset = dataset.select(range(start_idx, end_idx))
            print(f"✅ Processing samples {start_idx} to {end_idx-1} (total: {len(dataset)} samples)")
        else:
            dataset = dataset.select(range(start_idx, total_size))
            print(f"✅ Processing samples {start_idx} to {total_size-1} (total: {len(dataset)} samples)")
        
        use_local_mode = False

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"🚀 Loading CLIP vision model ({args.clip_model_name}) on {device} ...")
    clip_model = CLIPVisionModel.from_pretrained(args.clip_model_name).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model_name)

    dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    # Process samples
    if use_local_mode:
        # Mode 1: Process from local JSON + video files
        data_to_process = data_list
    else:
        # Mode 2: Process from HuggingFace dataset
        # Check first sample structure
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"🔍 Dataset sample keys: {list(first_sample.keys())}")
            print(f"🔍 First sample preview: {str(first_sample)[:500]}...")
            print()
        data_to_process = dataset
    
    for idx in tqdm(range(len(data_to_process)), desc="Exporting features"):
        if use_local_mode:
            # Local mode: data_list contains dicts with video_id
            sample = data_to_process[idx]
            video_id = sample.get("video_id") or sample.get("id") or f"sample_{idx}"
            
            # Try to find video file
            video_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                candidate = os.path.join(args.video_dir, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if video_path is None:
                # Try without extension
                candidate = os.path.join(args.video_dir, video_id)
                if os.path.exists(candidate):
                    video_path = candidate
            
            if video_path is None:
                print(f"[Skip] Video file not found for {video_id} in {args.video_dir}")
                continue
            
            save_path = os.path.join(args.output_dir, f"{video_id}.pt")
            if os.path.exists(save_path) and not args.overwrite:
                continue
            
            try:
                # Load video from file
                video_array = load_video_from_file(video_path, args.num_frames)
                frames = sample_frames(video_array, args.num_frames)
            except Exception as e:
                print(f"[Skip] Could not load video {video_path}: {e}")
                continue
        else:
            # HuggingFace mode: dataset contains video arrays
            sample = data_to_process[idx]
            
            # Try to get video data
            video = None
            if "video" in sample:
                video = sample["video"]
            elif "video_path" in sample or "video_url" in sample:
                if idx == 0:
                    print(f"\n❌ Sample {idx} has video_path/video_url but not video array.")
                    print(f"   This dataset likely only contains metadata, not actual video files.")
                    print(f"   Use --json_path and --video_dir to process local videos instead.")
                    print(f"   Available keys: {list(sample.keys())}")
                continue
            else:
                if idx == 0:
                    print(f"\n❌ Dataset does not contain 'video' field.")
                    print(f"   Available keys: {list(sample.keys())}")
                    print(f"   Use --json_path and --video_dir to process local videos instead.")
                continue
            
            try:
                video_array = resolve_video_array(video)
            except Exception as e:
                print(f"[Skip] Could not resolve video for sample {idx}: {e}")
                continue

            video_id = sample.get("video_id") or sample.get("id") or f"{args.split}_{idx}"
            save_path = os.path.join(args.output_dir, f"{video_id}.pt")
            if os.path.exists(save_path) and not args.overwrite:
                continue

            try:
                frames = sample_frames(video_array, args.num_frames)
            except Exception as e:
                print(f"[Skip] Failed to sample frames for {video_id}: {e}")
                continue
        
        # Encode frames with CLIP
        try:
            frame_embeds = encode_frames(
                frames, clip_model, clip_processor, device, args.batch_frames
            ).to(dtype)
        except Exception as e:
            print(f"[Skip] Failed to encode frames for {video_id}: {e}")
            continue

        torch.save(frame_embeds, save_path)
        if (idx + 1) % 100 == 0:  # Print every 100 samples
            print(f"💾 Saved {idx + 1}/{len(data_to_process)} features to {args.output_dir}")

    # Count saved files
    saved_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pt')]
    print(f"✅ Feature export finished. Total saved: {len(saved_files)} .pt files in {args.output_dir}")


if __name__ == "__main__":
    main()

