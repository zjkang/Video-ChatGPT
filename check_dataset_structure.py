#!/usr/bin/env python3
"""
快速检查 VideoInstruct-100K 数据集结构
"""
import os

# Set HuggingFace mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

print("🔍 正在加载数据集...")
try:
    dataset = load_dataset("MBZUAI/VideoInstruct-100K", split="train")
    print(f"✅ 数据集加载成功，共 {len(dataset)} 个样本\n")
    
    # 检查第一个样本
    if len(dataset) > 0:
        first_sample = dataset[0]
        print("=" * 60)
        print("📋 第一个样本的键 (Keys):")
        print("=" * 60)
        for key in first_sample.keys():
            value = first_sample[key]
            value_type = type(value).__name__
            if isinstance(value, (list, dict)):
                value_preview = f"{value_type} (length: {len(value)})"
            elif isinstance(value, str):
                value_preview = f"{value_type}: {value[:100]}..." if len(value) > 100 else f"{value_type}: {value}"
            else:
                value_preview = f"{value_type}: {str(value)[:100]}"
            print(f"  - {key}: {value_preview}")
        
        print("\n" + "=" * 60)
        print("🔍 检查是否包含视频数据:")
        print("=" * 60)
        
        has_video = "video" in first_sample
        has_video_path = "video_path" in first_sample
        has_video_url = "video_url" in first_sample
        has_video_id = "video_id" in first_sample
        
        print(f"  - 'video' 字段: {'✅ 存在' if has_video else '❌ 不存在'}")
        if has_video:
            video = first_sample["video"]
            print(f"    └─ 类型: {type(video).__name__}")
            if isinstance(video, dict):
                print(f"    └─ 包含的键: {list(video.keys())}")
                if "frames" in video:
                    frames = video["frames"]
                    print(f"    └─ frames 形状: {frames.shape if hasattr(frames, 'shape') else type(frames)}")
        
        print(f"  - 'video_path' 字段: {'✅ 存在' if has_video_path else '❌ 不存在'}")
        print(f"  - 'video_url' 字段: {'✅ 存在' if has_video_url else '❌ 不存在'}")
        print(f"  - 'video_id' 字段: {'✅ 存在' if has_video_id else '❌ 不存在'}")
        
        if has_video_id:
            print(f"    └─ video_id 值: {first_sample['video_id']}")
        
        print("\n" + "=" * 60)
        print("💡 结论:")
        print("=" * 60)
        if has_video:
            print("✅ 数据集包含视频数据，可以直接提取特征！")
            print("   使用命令:")
            print("   python scripts/export_video_instruct_features.py \\")
            print("       --output_dir data/VideoInstruct-100K/features_v100k_16f \\")
            print("       --start_index 0 --subset_size 2000")
        elif has_video_path or has_video_url:
            print("⚠️  数据集只包含视频路径/URL，不包含实际视频文件")
            print("   需要:")
            print("   1. 下载视频文件到本地")
            print("   2. 使用 --json_path 和 --video_dir 参数")
        else:
            print("❌ 数据集不包含视频数据")
            print("   需要:")
            print("   1. 下载 VideoInstruct100K.json 文件")
            print("   2. 下载对应的视频文件")
            print("   3. 使用 --json_path 和 --video_dir 参数")
        
        print("\n" + "=" * 60)
        print("📄 完整样本预览 (前500字符):")
        print("=" * 60)
        print(str(first_sample)[:500])
        if len(str(first_sample)) > 500:
            print("...")
        
except Exception as e:
    print(f"❌ 加载数据集失败: {e}")
    print("\n💡 可能的原因:")
    print("   1. 网络连接问题（尝试设置镜像）")
    print("   2. 数据集不存在或无法访问")
    print("\n建议:")
    print("   1. 检查网络连接")
    print("   2. 尝试直接下载 JSON 文件:")
    print("      python download_all.py")
    import traceback
    traceback.print_exc()

