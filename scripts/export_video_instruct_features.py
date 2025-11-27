import argparse
import os
from typing import Optional, List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor


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

    dataset = load_dataset(args.hf_dataset_name, split=args.split)
    total_size = len(dataset)
    
    # Handle start_index and subset_size
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

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"🚀 Loading CLIP vision model ({args.clip_model_name}) on {device} ...")
    clip_model = CLIPVisionModel.from_pretrained(args.clip_model_name).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model_name)

    dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    for idx in tqdm(range(len(dataset)), desc="Exporting features"):
        sample = dataset[idx]
        video = sample["video"]
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
            frame_embeds = encode_frames(
                frames, clip_model, clip_processor, device, args.batch_frames
            ).to(dtype)
        except Exception as e:
            print(f"[Skip] Failed to process {video_id}: {e}")
            continue

        torch.save(frame_embeds, save_path)

    print("✅ Feature export finished.")


if __name__ == "__main__":
    main()

