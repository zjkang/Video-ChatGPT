#!/usr/bin/env python3
"""
快速检查特征文件维度（只使用标准库）
"""
import pickle
import os
import sys

def check_pkl_file(pkl_path):
    """检查单个 .pkl 文件的格式"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 尝试获取 shape（如果是 numpy array）
        if hasattr(data, 'shape'):
            shape = data.shape
            dtype = str(data.dtype) if hasattr(data, 'dtype') else 'unknown'
            size_bytes = data.nbytes if hasattr(data, 'nbytes') else 0
            return {
                'type': 'numpy_array',
                'shape': shape,
                'dtype': dtype,
                'size_mb': size_bytes / (1024 * 1024) if size_bytes > 0 else 0
            }
        elif isinstance(data, (list, tuple)):
            # 如果是列表或元组，尝试推断形状
            if len(data) > 0:
                first = data[0]
                if hasattr(first, 'shape'):
                    return {
                        'type': 'list_of_arrays',
                        'length': len(data),
                        'first_element_shape': first.shape,
                        'first_element_dtype': str(first.dtype) if hasattr(first, 'dtype') else 'unknown'
                    }
            return {
                'type': 'list/tuple',
                'length': len(data)
            }
        else:
            return {
                'type': type(data).__name__,
                'info': str(data)[:200]
            }
    except Exception as e:
        return {
            'error': str(e)
        }

def main():
    features_dir = os.path.expanduser("~/Downloads/activity_clip-14L_spatio_temporal_356")
    
    if not os.path.exists(features_dir):
        print(f"❌ 目录不存在: {features_dir}")
        return
    
    # 查找第一个 .pkl 文件
    pkl_files = [f for f in os.listdir(features_dir) if f.endswith('.pkl')]
    
    if len(pkl_files) == 0:
        print(f"❌ 在 {features_dir} 中没有找到 .pkl 文件")
        return
    
    print(f"✅ 找到 {len(pkl_files)} 个 .pkl 文件")
    print(f"📁 目录: {features_dir}\n")
    
    # 检查前几个文件
    sample_files = pkl_files[:min(3, len(pkl_files))]
    
    print("=" * 60)
    print("🔍 检查样本文件:")
    print("=" * 60)
    
    for pkl_file in sample_files:
        pkl_path = os.path.join(features_dir, pkl_file)
        result = check_pkl_file(pkl_path)
        
        print(f"\n📄 {pkl_file}:")
        if 'error' in result:
            print(f"   ❌ 错误: {result['error']}")
        elif result.get('type') == 'numpy_array':
            print(f"   ✅ 类型: NumPy Array")
            print(f"   📊 Shape: {result['shape']}")
            print(f"   🔢 Dtype: {result['dtype']}")
            print(f"   💾 Size: {result['size_mb']:.2f} MB")
            
            # 根据形状推断格式
            shape = result['shape']
            if len(shape) == 2:
                if shape[0] == 356 and shape[1] == 1024:
                    print(f"   💡 格式: Spatio-temporal (356 = 100 temporal + 256 spatial)")
                elif shape[0] == 100 and shape[1] == 1024:
                    print(f"   💡 格式: Temporal only (100 frames)")
                elif shape[0] == 16 and shape[1] == 1024:
                    print(f"   💡 格式: Temporal only (16 frames)")
                elif shape[0] == 32 and shape[1] == 1024:
                    print(f"   💡 格式: Temporal only (32 frames)")
            elif len(shape) == 3:
                print(f"   💡 格式: (T, 256, 1024) - 需要转换为 (T, 1024)")
        else:
            print(f"   ⚠️  类型: {result.get('type', 'unknown')}")
            for key, value in result.items():
                if key != 'type':
                    print(f"   - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("💡 建议:")
    print("=" * 60)
    
    # 检查第一个文件的结果
    first_result = check_pkl_file(os.path.join(features_dir, pkl_files[0]))
    if 'shape' in first_result:
        shape = first_result['shape']
        if len(shape) == 2 and shape[0] == 356:
            print("⚠️  特征格式是 (356, 1024) - Spatio-temporal")
            print("   需要转换为 temporal-only 格式 (16/32/100, 1024)")
            print("\n   运行转换命令:")
            print("   python3 scripts/check_and_convert_features.py \\")
            print("       --features_path ~/Downloads/activity_clip-14L_spatio_temporal_356 \\")
            print("       --output_dir ~/Downloads/activity_clip-14L_temporal_16f \\")
            print("       --target_frames 16 \\")
            print("       --convert")
        elif len(shape) == 2 and shape[0] in [16, 32, 100]:
            print(f"✅ 特征格式是 ({shape[0]}, 1024) - 可以直接使用！")
            print(f"   训练时使用:")
            print(f"   --features_folder ~/Downloads/activity_clip-14L_spatio_temporal_356")
            print(f"   --num_frames {shape[0]}")

if __name__ == "__main__":
    main()

