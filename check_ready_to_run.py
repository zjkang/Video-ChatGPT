#!/usr/bin/env python3
"""
运行前检查脚本：检查代码是否可以直接运行
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (不存在)")
        return False

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    if os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"⚠️  {description}: {dirpath} (不存在，运行时会自动创建)")
        return True  # 目录不存在不是致命错误

def check_python_packages():
    """检查必要的 Python 包"""
    required_packages = [
        'torch',
        'transformers',
        'peft',
        'bitsandbytes',
        'numpy',
        'datasets',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package: {package}")
        except ImportError:
            print(f"❌ Package: {package} (未安装)")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_cuda():
    """检查 CUDA 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA 不可用（将使用 CPU，速度会很慢）")
            return False
    except:
        print("⚠️  无法检查 CUDA")
        return False

def main():
    print("=" * 60)
    print("🚀 运行前检查")
    print("=" * 60)
    
    all_ok = True
    
    # 1. 检查代码文件
    print("\n📁 检查代码文件:")
    code_files = [
        ("train_mem.py", "训练脚本"),
        ("video_chatgpt/model/video_chatgpt.py", "模型定义"),
        ("video_chatgpt/model/temporal_transformer.py", "TemporalTransformer"),
    ]
    for filepath, desc in code_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    # 2. 检查模型路径
    print("\n🤖 检查模型路径:")
    model_path = "./checkpoints/Llama-2-7b-chat-hf"
    if not check_directory_exists(model_path, "Llama-2 模型路径"):
        print("   💡 提示: 需要先下载 Llama-2-7B-Chat 模型")
        print("   可以使用: python download_all.py")
        all_ok = False
    
    # 3. 检查数据路径
    print("\n📊 检查数据路径:")
    data_path = "data/mini_dataset/mini_train.json"
    features_path = "data/mini_dataset/features"
    
    data_ok = check_file_exists(data_path, "训练数据 JSON")
    features_ok = check_directory_exists(features_path, "视频特征文件夹")
    
    if not data_ok or not features_ok:
        print("   💡 提示: 需要准备训练数据")
        print("   数据格式: JSON 文件包含 {'video_id', 'q', 'a'} 字段")
        print("   特征格式: features/{video_id}.pkl 文件，形状 (100, 1024)")
        all_ok = False
    
    # 4. 检查 Python 包
    print("\n📦 检查 Python 包:")
    if not check_python_packages():
        print("   💡 提示: 安装缺失的包")
        print("   pip install torch transformers peft bitsandbytes numpy datasets")
        all_ok = False
    
    # 5. 检查 CUDA
    print("\n🖥️  检查硬件:")
    cuda_ok = check_cuda()
    if not cuda_ok:
        print("   ⚠️  警告: 没有 GPU，训练会很慢")
    
    # 6. 检查输出目录
    print("\n💾 检查输出目录:")
    output_dir = "./outputs/baseline"
    check_directory_exists(output_dir, "输出目录")
    
    # 总结
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 所有检查通过！代码应该可以直接运行。")
        print("\n💡 运行命令:")
        print("   python train_mem.py \\")
        print("     --output_dir ./outputs/baseline \\")
        print("     --per_device_train_batch_size 2 \\")
        print("     --gradient_accumulation_steps 4 \\")
        print("     --learning_rate 2e-4 \\")
        print("     --max_steps 10 \\")
        print("     --logging_steps 1")
    else:
        print("❌ 发现一些问题，请先解决后再运行。")
        print("\n💡 常见问题:")
        print("   1. 模型未下载: 运行 python download_all.py")
        print("   2. 数据未准备: 准备 mini_train.json 和 features/ 文件夹")
        print("   3. 包未安装: pip install -r requirements.txt")
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

