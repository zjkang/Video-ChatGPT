#!/usr/bin/env python3
"""
验证 TemporalTransformer 集成是否正确
"""

import sys
import torch
import traceback

def test_temporal_transformer_import():
    """测试 TemporalTransformer 能否正确导入"""
    print("=" * 60)
    print("测试 1: TemporalTransformer 导入")
    print("=" * 60)
    try:
        from video_chatgpt.model.temporal_transformer import TemporalTransformer
        print("✅ TemporalTransformer 导入成功")
        return True, TemporalTransformer
    except Exception as e:
        print(f"❌ TemporalTransformer 导入失败: {e}")
        traceback.print_exc()
        return False, None

def test_temporal_transformer_basic(TemporalTransformer):
    """测试 TemporalTransformer 基本功能"""
    print("\n" + "=" * 60)
    print("测试 2: TemporalTransformer 基本功能")
    print("=" * 60)
    try:
        # 创建模型
        transformer = TemporalTransformer(
            input_dim=1024,
            output_dim=4096,
            num_layers=2,
            num_heads=8,
            max_seq_len=100
        )
        print("✅ TemporalTransformer 创建成功")
        
        # 测试前向传播
        batch_size = 2
        time_steps = 100
        input_dim = 1024
        x = torch.randn(batch_size, time_steps, input_dim)
        
        transformer.eval()
        with torch.no_grad():
            y = transformer(x)
        
        expected_shape = (batch_size, time_steps, 4096)
        if y.shape == expected_shape:
            print(f"✅ 输出形状正确: {y.shape} == {expected_shape}")
        else:
            print(f"❌ 输出形状错误: {y.shape} != {expected_shape}")
            return False
        
        # 测试参数数量
        num_params = sum(p.numel() for p in transformer.parameters())
        print(f"✅ 参数量: {num_params:,} (约 {num_params/1e6:.2f}M)")
        
        return True
    except Exception as e:
        print(f"❌ TemporalTransformer 基本功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_video_chatgpt_import():
    """测试 VideoChatGPT 模型能否正确导入"""
    print("\n" + "=" * 60)
    print("测试 3: VideoChatGPT 模型导入")
    print("=" * 60)
    try:
        from video_chatgpt.model.video_chatgpt import (
            VideoChatGPTLlamaForCausalLM,
            VideoChatGPTLlamaModel,
            VisionConfig
        )
        print("✅ VideoChatGPT 模型导入成功")
        return True
    except Exception as e:
        print(f"❌ VideoChatGPT 模型导入失败: {e}")
        traceback.print_exc()
        return False

def test_mm_projector_type():
    """测试 mm_projector 是否为 TemporalTransformer"""
    print("\n" + "=" * 60)
    print("测试 4: mm_projector 类型检查")
    print("=" * 60)
    try:
        from video_chatgpt.model.video_chatgpt import VideoChatGPTLlamaModel, VisionConfig
        from video_chatgpt.model.temporal_transformer import TemporalTransformer
        from transformers import LlamaConfig
        
        # 创建配置
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            mm_vision_tower="test",
            use_mm_proj=True,
            mm_hidden_size=1024
        )
        
        # 创建模型
        model = VideoChatGPTLlamaModel(config)
        
        # 检查 mm_projector 是否存在
        if not hasattr(model, 'mm_projector'):
            print("❌ mm_projector 不存在")
            return False
        
        # 检查类型
        if isinstance(model.mm_projector, TemporalTransformer):
            print("✅ mm_projector 是 TemporalTransformer 类型")
        else:
            print(f"❌ mm_projector 类型错误: {type(model.mm_projector)}")
            print(f"   期望: TemporalTransformer")
            return False
        
        # 测试 initialize_vision_modules
        model.vision_config = VisionConfig()
        result = model.initialize_vision_modules()
        
        if isinstance(model.mm_projector, TemporalTransformer):
            print("✅ initialize_vision_modules 后，mm_projector 仍然是 TemporalTransformer")
        else:
            print(f"❌ initialize_vision_modules 后，mm_projector 类型错误: {type(model.mm_projector)}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ mm_projector 类型检查失败: {e}")
        traceback.print_exc()
        return False

def test_forward_shape():
    """测试 forward 方法的形状兼容性"""
    print("\n" + "=" * 60)
    print("测试 5: Forward 方法形状兼容性")
    print("=" * 60)
    try:
        from video_chatgpt.model.video_chatgpt import VideoChatGPTLlamaModel, VisionConfig
        from transformers import LlamaConfig
        
        # 创建配置
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=2,  # 使用较小的层数以加快测试
            num_attention_heads=32,
            mm_vision_tower="test",
            use_mm_proj=True,
            mm_hidden_size=1024
        )
        
        # 创建模型
        model = VideoChatGPTLlamaModel(config)
        model.vision_config = VisionConfig()
        model.vision_config.vid_patch_token = 32001  # 设置一个假的 token id
        
        # 初始化视觉模块
        model.initialize_vision_modules()
        
        # 准备输入
        batch_size = 2
        time_steps = 100
        video_features = torch.randn(batch_size, time_steps, 1024)
        
        # 创建假的 input_ids（包含 vid_patch_token）
        num_video_tokens = 100
        input_ids = torch.full((batch_size, 200), 0)  # 填充 token
        # 在中间插入 vid_patch_token
        input_ids[:, 50:50+num_video_tokens] = model.vision_config.vid_patch_token
        
        # 测试 forward（不完整，但可以测试形状）
        model.eval()
        with torch.no_grad():
            # 只测试 mm_projector 部分
            output = model.mm_projector(video_features)
        
        expected_shape = (batch_size, time_steps, 4096)
        if output.shape == expected_shape:
            print(f"✅ mm_projector 输出形状正确: {output.shape} == {expected_shape}")
        else:
            print(f"❌ mm_projector 输出形状错误: {output.shape} != {expected_shape}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Forward 方法形状兼容性测试失败: {e}")
        traceback.print_exc()
        return False

def test_model_utils():
    """测试 model_utils.py 中的修改"""
    print("\n" + "=" * 60)
    print("测试 6: model_utils.py 中的 TemporalTransformer 导入")
    print("=" * 60)
    try:
        # 检查文件内容
        import os
        file_path = "video_chatgpt/eval/model_utils.py"
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        if "TemporalTransformer" in content:
            print("✅ model_utils.py 包含 TemporalTransformer")
        else:
            print("❌ model_utils.py 不包含 TemporalTransformer")
            return False
        
        if "from video_chatgpt.model.temporal_transformer import TemporalTransformer" in content:
            print("✅ model_utils.py 包含正确的导入语句")
        else:
            print("❌ model_utils.py 缺少导入语句")
            return False
        
        return True
    except Exception as e:
        print(f"❌ model_utils.py 检查失败: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TemporalTransformer 集成验证")
    print("=" * 60)
    
    results = []
    
    # 测试 1: 导入
    success, TemporalTransformer = test_temporal_transformer_import()
    results.append(("导入测试", success))
    
    if not success:
        print("\n❌ 导入失败，无法继续测试")
        return
    
    # 测试 2: 基本功能
    success = test_temporal_transformer_basic(TemporalTransformer)
    results.append(("基本功能测试", success))
    
    # 测试 3: VideoChatGPT 导入
    success = test_video_chatgpt_import()
    results.append(("VideoChatGPT 导入", success))
    
    # 测试 4: mm_projector 类型
    success = test_mm_projector_type()
    results.append(("mm_projector 类型", success))
    
    # 测试 5: Forward 形状
    success = test_forward_shape()
    results.append(("Forward 形状兼容性", success))
    
    # 测试 6: model_utils
    success = test_model_utils()
    results.append(("model_utils.py 检查", success))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！TemporalTransformer 集成成功！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())

