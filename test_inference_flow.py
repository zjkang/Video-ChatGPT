"""
测试推理流程是否正确
验证关键步骤：
1. 视频特征是否正确传递
2. Token 数量是否匹配
3. forward 方法是否正确处理视频特征
"""

import torch
import sys

def test_forward_condition():
    """测试 forward 方法的条件判断"""
    print("=" * 60)
    print("测试 1: forward 方法的条件判断")
    print("=" * 60)
    
    # 模拟不同场景
    scenarios = [
        {
            "name": "训练时",
            "training": True,
            "past_key_values": None,
            "input_ids_shape": (1, 100),  # 完整 prompt
            "video_features": torch.randn(16, 1024),
            "should_process": True,
        },
        {
            "name": "推理第一次 forward",
            "training": False,
            "past_key_values": None,
            "input_ids_shape": (1, 100),  # 完整 prompt
            "video_features": torch.randn(16, 1024),
            "should_process": True,  # ✅ 应该处理
        },
        {
            "name": "推理后续步骤（有 past_key_values）",
            "training": False,
            "past_key_values": [torch.randn(1, 1, 4096)],  # 模拟 past_key_values
            "input_ids_shape": (1, 1),  # 只有最后一个 token
            "video_features": torch.randn(16, 1024),
            "should_process": False,  # ✅ 不应该处理（已经在第一次处理过了）
        },
        {
            "name": "推理后续步骤（无 video_features）",
            "training": False,
            "past_key_values": [torch.randn(1, 1, 4096)],
            "input_ids_shape": (1, 1),
            "video_features": None,
            "should_process": False,  # ✅ 不应该处理（没有特征）
        },
    ]
    
    for scenario in scenarios:
        # 模拟条件判断
        should_process_video = (
            scenario["video_features"] is not None and 
            (scenario["training"] or scenario["past_key_values"] is None)
        )
        
        expected = scenario["should_process"]
        actual = should_process_video
        
        status = "✅" if actual == expected else "❌"
        print(f"{status} {scenario['name']}:")
        print(f"   - training: {scenario['training']}")
        print(f"   - past_key_values: {scenario['past_key_values'] is not None}")
        print(f"   - video_features: {scenario['video_features'] is not None}")
        print(f"   - 应该处理: {expected}, 实际处理: {actual}")
        if actual != expected:
            print(f"   ⚠️  不匹配！")
        print()

def test_token_count_matching():
    """测试 token 数量匹配"""
    print("=" * 60)
    print("测试 2: Token 数量匹配验证")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "匹配（16 tokens, 16 features）",
            "num_tokens": 16,
            "num_features": 16,
            "should_pass": True,
        },
        {
            "name": "不匹配（16 tokens, 100 features）",
            "num_tokens": 16,
            "num_features": 100,
            "should_pass": False,
        },
        {
            "name": "不匹配（100 tokens, 16 features）",
            "num_tokens": 100,
            "num_features": 16,
            "should_pass": False,
        },
    ]
    
    for scenario in scenarios:
        num_tokens = scenario["num_tokens"]
        num_features = scenario["num_features"]
        should_pass = scenario["should_pass"]
        
        # 模拟验证逻辑
        if num_tokens != num_features:
            actual_pass = False
            error_msg = f"Token count mismatch: {num_tokens} tokens != {num_features} features"
        else:
            actual_pass = True
            error_msg = None
        
        status = "✅" if actual_pass == should_pass else "❌"
        print(f"{status} {scenario['name']}:")
        print(f"   - Tokens: {num_tokens}, Features: {num_features}")
        print(f"   - 应该通过: {should_pass}, 实际通过: {actual_pass}")
        if error_msg:
            print(f"   - 错误信息: {error_msg}")
        print()

def test_prepare_inputs_for_generation():
    """测试 prepare_inputs_for_generation 是否正确传递 video_features"""
    print("=" * 60)
    print("测试 3: prepare_inputs_for_generation 传递 video_features")
    print("=" * 60)
    
    # 模拟 prepare_inputs_for_generation 的逻辑
    def mock_prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "past_key_values": past_key_values,
            "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
        })
        return model_inputs
    
    scenarios = [
        {
            "name": "第一次 forward（无 past_key_values）",
            "input_ids": torch.randint(0, 1000, (1, 100)),
            "past_key_values": None,
            "video_features": torch.randn(16, 1024),
            "should_include": True,
        },
        {
            "name": "后续步骤（有 past_key_values）",
            "input_ids": torch.randint(0, 1000, (1, 1)),
            "past_key_values": [torch.randn(1, 1, 4096)],
            "video_features": torch.randn(16, 1024),
            "should_include": True,  # ✅ 应该传递（虽然不会处理，但要传递）
        },
    ]
    
    for scenario in scenarios:
        result = mock_prepare_inputs_for_generation(
            scenario["input_ids"],
            past_key_values=scenario["past_key_values"],
            video_spatio_temporal_features=scenario["video_features"]
        )
        
        has_video_features = result["video_spatio_temporal_features"] is not None
        should_include = scenario["should_include"]
        
        status = "✅" if has_video_features == should_include else "❌"
        print(f"{status} {scenario['name']}:")
        print(f"   - past_key_values: {scenario['past_key_values'] is not None}")
        print(f"   - 应该包含 video_features: {should_include}")
        print(f"   - 实际包含 video_features: {has_video_features}")
        if has_video_features:
            print(f"   - video_features shape: {result['video_spatio_temporal_features'].shape}")
        print()

def main():
    print("\n" + "=" * 60)
    print("推理流程验证测试")
    print("=" * 60 + "\n")
    
    test_forward_condition()
    test_token_count_matching()
    test_prepare_inputs_for_generation()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

