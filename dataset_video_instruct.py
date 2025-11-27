import os
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import CLIPVisionModel, CLIPImageProcessor, PreTrainedTokenizer
from dataclasses import dataclass


class VideoInstruct100KDataset(Dataset):
    """
    VideoInstruct100K + CLIP 抽帧 + 帧级特征 [T, 1024] 的 Dataset。

    - 每个样本：
        video_spatio_temporal_features: [T, 1024]，T = num_frames
        prompt: "Human: <video> <vid_patch> x T {Q}\\nAssistant: {A}</s>"
    - 与 VideoChatGPTLlamaForCausalLM 完全兼容
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        num_frames: int = 16,
        hf_dataset_name: str = "vidore/video-instruct-100k",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        subset_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.num_frames = num_frames

        self.dataset = load_dataset(hf_dataset_name, split=split)
        if subset_size is not None:
            self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
            print(f"✅ Using subset of size {len(self.dataset)} from {hf_dataset_name}/{split}")
        else:
            print(f"✅ Loaded {len(self.dataset)} samples from {hf_dataset_name}/{split}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"🚀 Loading CLIP vision model ({clip_model_name}) on {self.device} ...")
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        vocab = tokenizer.get_vocab()
        if "<video>" not in vocab or "<vid_patch>" not in vocab:
            raise ValueError("Tokenizer must already contain <video> and <vid_patch> tokens.")
        self.video_token = "<video>"
        self.vid_patch_token = "<vid_patch>"

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.ignore_index = -100

    def __len__(self):
        return len(self.dataset)

    def _sample_frames(self, video_array: np.ndarray) -> np.ndarray:
        total_frames = video_array.shape[0]
        if total_frames <= 0:
            raise ValueError("Video has no frames.")
        T = min(self.num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=T, dtype=int)
        frames = video_array[indices]
        return frames

    @torch.no_grad()
    def _encode_frames_with_clip(self, frames: np.ndarray) -> torch.Tensor:
        images = [frame for frame in frames]
        inputs = self.clip_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.clip_model(pixel_values=pixel_values)
        frame_embeds = outputs.last_hidden_state[:, 0, :]
        return frame_embeds.cpu()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        video = sample["video"]
        if isinstance(video, dict) and "frames" in video:
            video_array = video["frames"].astype(np.uint8)
        else:
            video_array = np.array(video, dtype=np.uint8)

        frames = self._sample_frames(video_array)
        frame_embeds = self._encode_frames_with_clip(frames)
        T = frame_embeds.shape[0]

        conv = sample["conversations"]
        humans = [c for c in conv if c.get("from") == "human"]
        assists = [c for c in conv if c.get("from") == "assistant"]
        if len(humans) == 0 or len(assists) == 0:
            q = "Please describe this video."
            a = "The video shows some generic content."
        else:
            q = humans[0]["value"]
            a = assists[0]["value"]

        video_tokens = " ".join([self.vid_patch_token] * T)
        prompt = f"Human: {self.video_token} {video_tokens} {q}\nAssistant: {a}</s>"

        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]

        labels = input_ids.clone()
        prompt_before_answer = f"Human: {self.video_token} {video_tokens} {q}\nAssistant:"
        before_tokenized = self.tokenizer(
            prompt_before_answer,
            return_tensors="pt",
            add_special_tokens=True
        )
        before_len = before_tokenized.input_ids[0].shape[0]
        labels[:before_len] = self.ignore_index
        labels[input_ids == self.pad_token_id] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "video": frame_embeds.to(dtype=torch.float16),
        }


@dataclass
class DataCollatorForVideoInstruct:
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([ins["input_ids"] for ins in instances])
        attention_mask = torch.stack([ins["attention_mask"] for ins in instances])
        labels = torch.stack([ins["labels"] for ins in instances])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "video" in instances[0]:
            feats = [ins["video"] for ins in instances]
            if all(f.shape == feats[0].shape for f in feats):
                batch["video_spatio_temporal_features"] = torch.stack(feats)
            else:
                max_T = max(f.shape[0] for f in feats)
                padded = []
                for f in feats:
                    T, D = f.shape
                    if T < max_T:
                        pad = torch.zeros(max_T - T, D, dtype=f.dtype)
                        f = torch.cat([f, pad], dim=0)
                    padded.append(f)
                batch["video_spatio_temporal_features"] = torch.stack(padded)

        return batch

