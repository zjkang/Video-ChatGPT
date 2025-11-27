#!/usr/bin/env python3
"""
检查并转换预提取的特征文件格式
支持从 zip 文件解压，并检查格式是否匹配训练代码
"""
import os
import pickle
import zipfile
import numpy as np
import argparse
from pathlib import Path


def check_pkl_file(pkl_path):
    """检查单个 .pkl 文件的格式"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, np.ndarray):
            shape = data.shape
            dtype = data.dtype
            return {
                'valid': True,
                'shape': shape,
                'dtype': str(dtype),
                'size_mb': data.nbytes / (1024 * 1024)
            }
        else:
            return {
                'valid': False,
                'error': f'Not a numpy array, type: {type(data)}'
            }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def extract_zip(zip_path, output_dir):
    """解压 zip 文件"""
    print(f"📦 正在解压: {zip_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"✅ 解压完成: {output_dir}")
    return output_dir


def check_features_directory(features_dir):
    """检查特征目录中的所有 .pkl 文件"""
    pkl_files = list(Path(features_dir).glob("*.pkl"))
    
    if len(pkl_files) == 0:
        print(f"❌ 在 {features_dir} 中没有找到 .pkl 文件")
        return None
    
    print(f"✅ 找到 {len(pkl_files)} 个 .pkl 文件")
    
    # 检查前几个文件
    sample_files = pkl_files[:min(5, len(pkl_files))]
    print("\n🔍 检查样本文件格式:")
    print("=" * 60)
    
    shapes = []
    for pkl_file in sample_files:
        result = check_pkl_file(pkl_file)
        if result['valid']:
            print(f"  ✅ {pkl_file.name}")
            print(f"     Shape: {result['shape']}")
            print(f"     Dtype: {result['dtype']}")
            print(f"     Size: {result['size_mb']:.2f} MB")
            shapes.append(result['shape'])
        else:
            print(f"  ❌ {pkl_file.name}: {result.get('error', 'Unknown error')}")
    
    # 检查格式一致性
    if len(set(shapes)) == 1:
        print(f"\n✅ 所有文件格式一致: {shapes[0]}")
        expected_shape = shapes[0]
    elif len(set([s[:2] if len(s) >= 2 else s for s in shapes])) == 1:
        # 如果前两个维度一致（忽略第三个维度）
        print(f"\n⚠️  文件格式基本一致（可能有维度差异）")
        expected_shape = shapes[0]
    else:
        print(f"\n⚠️  文件格式不一致，请检查")
        expected_shape = shapes[0] if shapes else None
    
    return {
        'total_files': len(pkl_files),
        'sample_shape': expected_shape,
        'directory': features_dir
    }


def convert_to_training_format(features_dir, output_dir, target_frames=100):
    """
    转换特征文件为训练代码期望的格式
    如果特征文件是 (T, 256, 1024) 格式，需要转换为 (T, 1024)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_files = list(Path(features_dir).glob("*.pkl"))
    print(f"\n🔄 开始转换 {len(pkl_files)} 个特征文件...")
    
    converted = 0
    skipped = 0
    errors = 0
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 检查是否需要转换
            if len(data.shape) == 3 and data.shape[1] == 256:
                # (T, 256, 1024) -> (T, 1024) 对空间维度平均
                data = np.mean(data, axis=1)
                converted += 1
            elif len(data.shape) == 2:
                if data.shape[0] == 356 and data.shape[1] == 1024:
                    # (356, 1024) = (100 temporal + 256 spatial, 1024)
                    # 提取前 100 个 temporal tokens
                    data = data[:100]  # 取前 100 个 temporal tokens
                    converted += 1
                
                # 调整帧数到目标值
                if data.shape[0] != target_frames:
                    if data.shape[0] < target_frames:
                        # Padding
                        padding = np.zeros((target_frames - data.shape[0], data.shape[1]), dtype=data.dtype)
                        data = np.concatenate([data, padding], axis=0)
                    else:
                        # 均匀采样到目标帧数
                        indices = np.linspace(0, data.shape[0] - 1, num=target_frames, dtype=int)
                        data = data[indices]
                skipped += 1
            else:
                print(f"⚠️  跳过 {pkl_file.name}: 未知格式 {data.shape}")
                errors += 1
                continue
            
            # 确保是 float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # 保存转换后的文件
            output_path = os.path.join(output_dir, pkl_file.name)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            print(f"❌ 处理 {pkl_file.name} 失败: {e}")
            errors += 1
    
    print(f"\n✅ 转换完成:")
    print(f"   - 转换: {converted}")
    print(f"   - 跳过（已正确格式）: {skipped}")
    print(f"   - 错误: {errors}")
    print(f"   - 输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="检查并转换特征文件格式")
    parser.add_argument("--features_path", type=str, required=True,
                        help="特征文件路径（可以是 zip 文件或目录）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="转换后的输出目录（如果需要转换）")
    parser.add_argument("--convert", action="store_true",
                        help="是否转换格式（如果需要）")
    parser.add_argument("--target_frames", type=int, default=16,
                        help="目标帧数（默认 16，也可以是 32 或其他值）")
    
    args = parser.parse_args()
    
    features_path = Path(args.features_path)
    
    # 如果是 zip 文件，先解压
    if features_path.suffix == '.zip':
        print(f"📦 检测到 zip 文件: {features_path}")
        extract_dir = features_path.parent / features_path.stem
        if not extract_dir.exists():
            extract_zip(str(features_path), str(extract_dir))
        features_dir = extract_dir
    else:
        features_dir = features_path
    
    # 检查格式
    result = check_features_directory(features_dir)
    
    if result is None:
        return
    
    # 判断是否需要转换
    sample_shape = result['sample_shape']
    needs_conversion = False
    
    if sample_shape is not None:
        print(f"\n📊 特征格式分析:")
        print(f"   - 当前格式: {sample_shape}")
        
        if len(sample_shape) == 3:
            print(f"   ⚠️  需要转换: (T, 256, 1024) -> (T, 1024)")
            needs_conversion = True
        elif len(sample_shape) == 2:
            if sample_shape == (356, 1024):
                print(f"   ⚠️  需要转换: (356, 1024) -> ({args.target_frames}, 1024)")
                print(f"      (356 = 100 temporal + 256 spatial, 提取前 100 个 temporal tokens)")
                needs_conversion = True
            elif sample_shape[0] != args.target_frames:
                print(f"   ⚠️  需要调整帧数: {sample_shape[0]} -> {args.target_frames}")
                needs_conversion = True
            else:
                print(f"   ✅ 格式正确，可以直接使用！")
        else:
            print(f"   ❌ 未知格式，无法自动转换")
    
    # 如果需要转换
    if needs_conversion and args.convert:
        if args.output_dir is None:
            args.output_dir = str(features_dir.parent / f"{features_dir.name}_converted")
        
        convert_to_training_format(features_dir, args.output_dir, args.target_frames)
        print(f"\n💡 使用转换后的特征:")
        print(f"   --features_folder {args.output_dir}")
    elif needs_conversion:
        print(f"\n💡 需要转换格式，运行:")
        print(f"   python scripts/check_and_convert_features.py \\")
        print(f"       --features_path {features_path} \\")
        print(f"       --output_dir data/VideoInstruct-100K/features_converted \\")
        print(f"       --target_frames {args.target_frames} \\")
        print(f"       --convert")
    else:
        print(f"\n💡 可以直接使用:")
        print(f"   --features_folder {features_dir}")


if __name__ == "__main__":
    main()

