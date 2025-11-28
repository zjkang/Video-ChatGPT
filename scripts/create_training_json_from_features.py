#!/usr/bin/env python3
"""
从特征文件创建训练用的 JSON 标注文件
1. 扫描特征文件目录，获取所有 video_id
2. 从 HuggingFace 下载或加载 JSON 标注文件
3. 过滤出匹配的样本
4. 保存为训练格式
"""
import os
import json
import argparse
from pathlib import Path

# Set HuggingFace mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  Warning: datasets library not available. Will try to load from local JSON file.")


def get_video_ids_from_features(features_dir):
    """从特征文件目录获取所有 video_id"""
    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    video_ids = []
    for pkl_file in features_path.glob("*.pkl"):
        # 去掉 .pkl 后缀
        video_id = pkl_file.stem
        video_ids.append(video_id)
    
    print(f"✅ Found {len(video_ids)} feature files in {features_dir}")
    return set(video_ids)


def load_annotations_from_hf():
    """从 HuggingFace 加载标注数据"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    print("📦 Loading annotations from HuggingFace...")
    dataset = load_dataset("MBZUAI/VideoInstruct-100K", split="train")
    print(f"✅ Loaded {len(dataset)} samples from HuggingFace")
    
    # 转换为列表格式
    annotations = []
    for sample in dataset:
        # 检查数据格式
        if "conversations" in sample:
            # 提取 conversations 格式
            conv = sample["conversations"]
            humans = [c for c in conv if c.get("from") == "human"]
            assists = [c for c in conv if c.get("from") == "assistant"]
            
            if len(humans) > 0 and len(assists) > 0:
                annotations.append({
                    "video_id": sample.get("video_id", sample.get("id", "")),
                    "q": humans[0]["value"],
                    "a": assists[0]["value"]
                })
        elif "q" in sample and "a" in sample:
            # 已经是 q/a 格式
            annotations.append({
                "video_id": sample.get("video_id", sample.get("id", "")),
                "q": sample["q"],
                "a": sample["a"]
            })
    
    print(f"✅ Extracted {len(annotations)} valid annotations")
    return annotations


def load_annotations_from_json(json_path):
    """从本地 JSON 文件加载标注数据"""
    print(f"📖 Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 转换为统一格式
    annotations = []
    for item in data:
        if "conversations" in item:
            conv = item["conversations"]
            humans = [c for c in conv if c.get("from") == "human"]
            assists = [c for c in conv if c.get("from") == "assistant"]
            
            if len(humans) > 0 and len(assists) > 0:
                annotations.append({
                    "video_id": item.get("video_id", item.get("id", "")),
                    "q": humans[0]["value"],
                    "a": assists[0]["value"]
                })
        elif "q" in item and "a" in item:
            annotations.append({
                "video_id": item.get("video_id", item.get("id", "")),
                "q": item["q"],
                "a": item["a"]
            })
    
    print(f"✅ Loaded {len(annotations)} annotations from JSON")
    return annotations


def filter_annotations(annotations, video_ids):
    """过滤出匹配的标注"""
    matched = []
    for ann in annotations:
        vid = ann["video_id"]
        # 尝试不同的匹配方式
        if vid in video_ids:
            matched.append(ann)
        elif vid.startswith("v_") and vid[2:] in video_ids:
            # 如果标注有 v_ 前缀，但特征文件没有
            ann["video_id"] = vid[2:]
            matched.append(ann)
        elif f"v_{vid}" in video_ids:
            # 如果特征文件有 v_ 前缀，但标注没有
            ann["video_id"] = f"v_{vid}"
            matched.append(ann)
    
    print(f"✅ Matched {len(matched)} annotations with feature files")
    return matched


def main():
    parser = argparse.ArgumentParser(description="Create training JSON from feature files")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing feature .pkl files")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--annotation_json", type=str, default=None,
                        help="Local annotation JSON file (if not provided, will download from HF)")
    parser.add_argument("--hf_dataset", type=str, default="MBZUAI/VideoInstruct-100K",
                        help="HuggingFace dataset name")
    
    args = parser.parse_args()
    
    # 1. 获取特征文件中的 video_id
    video_ids = get_video_ids_from_features(args.features_dir)
    
    # 2. 加载标注数据
    if args.annotation_json and os.path.exists(args.annotation_json):
        annotations = load_annotations_from_json(args.annotation_json)
    else:
        try:
            annotations = load_annotations_from_hf()
        except Exception as e:
            print(f"❌ Failed to load from HuggingFace: {e}")
            print("💡 Please provide --annotation_json with a local JSON file")
            return
    
    # 3. 过滤匹配的标注
    matched = filter_annotations(annotations, video_ids)
    
    if len(matched) == 0:
        print("❌ No matching annotations found!")
        print("💡 Check if video_id format matches between features and annotations")
        return
    
    # 4. 保存结果
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(matched, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Training JSON created: {args.output_json}")
    print(f"   - Total samples: {len(matched)}")
    print(f"   - Coverage: {len(matched)}/{len(video_ids)} feature files ({100*len(matched)/len(video_ids):.1f}%)")
    
    print(f"\n💡 Use this JSON for training:")
    print(f"   --data_path {args.output_json} \\")
    print(f"   --features_folder {args.features_dir}")

if __name__ == "__main__":
    main()

