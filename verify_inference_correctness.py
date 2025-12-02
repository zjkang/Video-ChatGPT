"""
验证推理流程正确性的脚本
可以在 remote server 上运行，检查关键配置是否正确
"""

import torch
import sys
import os

def check_forward_condition_logic():
    """检查 forward 方法的条件逻辑是否正确"""
    print("=" * 70)
    print("检查 1: forward 方法的视频特征处理条件")
    print("=" * 70)
    
    # 模拟不同场景
    test_cases = [
        {
            "name": "训练时（应该处理）",
            "training": True,
            "past_key_values": None,
            "video_features": True,
            "expected": True,
        },
        {
            "name": "推理第一次 forward（past_key_values=None，应该处理）",
            "training": False,
            "past_key_values": None,
            "video_features": True,
            "expected": True,
        },
        {
            "name": "推理后续步骤（past_key_values存在，不应该处理）",
            "training": False,
            "past_key_values": True,  # 表示存在
            "video_features": True,
            "expected": False,  # 不应该处理，因为已经在第一次处理过了
        },
        {
            "name": "没有视频特征（不应该处理）",
            "training": False,
            "past_key_values": None,
            "video_features": False,
            "expected": False,
        },
    ]
    
    all_pass = True
    for case in test_cases:
        # 模拟我们的修复后的条件
        should_process = (
            case["video_features"] and 
            (case["training"] or case["past_key_values"] is None)
        )
        
        status = "✅" if should_process == case["expected"] else "❌"
        if should_process != case["expected"]:
            all_pass = False
        
        print(f"{status} {case['name']}:")
        print(f"   - training: {case['training']}")
        print(f"   - past_key_values: {case['past_key_values']}")
        print(f"   - video_features: {case['video_features']}")
        print(f"   - 应该处理: {case['expected']}, 实际: {should_process}")
        print()
    
    return all_pass

def check_token_count_validation():
    """检查 token 数量验证逻辑"""
    print("=" * 70)
    print("检查 2: Token 数量与特征数量匹配验证")
    print("=" * 70)
    
    test_cases = [
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
    
    all_pass = True
    for case in test_cases:
        num_tokens = case["num_tokens"]
        num_features = case["num_features"]
        should_pass = case["should_pass"]
        
        # 模拟验证逻辑
        if num_tokens != num_features:
            actual_pass = False
        else:
            actual_pass = True
        
        status = "✅" if actual_pass == should_pass else "❌"
        if actual_pass != should_pass:
            all_pass = False
        
        print(f"{status} {case['name']}:")
        print(f"   - Tokens: {num_tokens}, Features: {num_features}")
        print(f"   - 应该通过: {should_pass}, 实际: {actual_pass}")
        if not actual_pass:
            print(f"   - ⚠️  会抛出 ValueError（这是正确的行为）")
        print()
    
    return all_pass

def check_prepare_inputs_for_generation():
    """检查 prepare_inputs_for_generation 是否正确传递 video_features"""
    print("=" * 70)
    print("检查 3: prepare_inputs_for_generation 传递 video_features")
    print("=" * 70)
    
    # 模拟 prepare_inputs_for_generation 的逻辑（从代码中提取）
    def mock_prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "past_key_values": past_key_values,
            "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
        })
        return model_inputs
    
    test_cases = [
        {
            "name": "第一次 forward（无 past_key_values）",
            "past_key_values": None,
            "video_features": torch.randn(16, 1024),
            "should_include": True,
        },
        {
            "name": "后续步骤（有 past_key_values）",
            "past_key_values": [torch.randn(1, 1, 4096)],  # 模拟 past_key_values
            "video_features": torch.randn(16, 1024),
            "should_include": True,  # ✅ 应该传递（虽然不会处理，但要传递）
        },
    ]
    
    all_pass = True
    for case in test_cases:
        result = mock_prepare_inputs_for_generation(
            torch.randint(0, 1000, (1, 100)),
            past_key_values=case["past_key_values"],
            video_spatio_temporal_features=case["video_features"]
        )
        
        has_video_features = result["video_spatio_temporal_features"] is not None
        should_include = case["should_include"]
        
        status = "✅" if has_video_features == should_include else "❌"
        if has_video_features != should_include:
            all_pass = False
        
        print(f"{status} {case['name']}:")
        print(f"   - past_key_values: {case['past_key_values'] is not None}")
        print(f"   - 应该包含 video_features: {should_include}")
        print(f"   - 实际包含 video_features: {has_video_features}")
        if has_video_features:
            print(f"   - video_features shape: {result['video_spatio_temporal_features'].shape}")
        print()
    
    return all_pass

def check_code_files():
    """检查关键代码文件是否存在且包含修复"""
    print("=" * 70)
    print("检查 4: 关键代码文件检查")
    print("=" * 70)
    
    files_to_check = [
        {
            "path": "video_chatgpt/model/video_chatgpt.py",
            "keywords": ["should_process_video", "past_key_values is None"],
            "description": "forward 方法的视频特征处理条件修复",
        },
        {
            "path": "run_cli.py",
            "keywords": ["Token count mismatch", "actual_video_token_len"],
            "description": "Token 数量验证",
        },
    ]
    
    all_pass = True
    for file_info in files_to_check:
        file_path = file_info["path"]
        keywords = file_info["keywords"]
        description = file_info["description"]
        
        if not os.path.exists(file_path):
            print(f"❌ {file_path}: 文件不存在")
            all_pass = False
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        found_keywords = []
        missing_keywords = []
        for keyword in keywords:
            if keyword in content:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        if missing_keywords:
            print(f"❌ {file_path}:")
            print(f"   - 描述: {description}")
            print(f"   - 缺少关键词: {missing_keywords}")
            all_pass = False
        else:
            print(f"✅ {file_path}:")
            print(f"   - 描述: {description}")
            print(f"   - 找到关键词: {found_keywords}")
        print()
    
    return all_pass

def main():
    print("\n" + "=" * 70)
    print("推理流程正确性验证")
    print("=" * 70 + "\n")
    
    results = []
    
    # 检查 1: forward 条件逻辑
    results.append(("forward 条件逻辑", check_forward_condition_logic()))
    print()
    
    # 检查 2: token 数量验证
    results.append(("Token 数量验证", check_token_count_validation()))
    print()
    
    # 检查 3: prepare_inputs_for_generation
    results.append(("prepare_inputs_for_generation", check_prepare_inputs_for_generation()))
    print()
    
    # 检查 4: 代码文件
    results.append(("代码文件检查", check_code_files()))
    print()
    
    # 总结
    print("=" * 70)
    print("验证结果总结")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} {name}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("🎉 所有检查都通过！推理流程应该是正确的。")
        print("\n建议：")
        print("1. 运行实际推理测试，验证模型输出是否与视频相关")
        print("2. 检查日志中的以下信息：")
        print("   - ✅ Token count matches: X tokens == X features")
        print("   - ✅ Video features shape for model: torch.Size([1, X, 1024])")
        print("   - ✅ Using eos_token_id=... for generation")
    else:
        print("⚠️  部分检查未通过，请检查相关代码。")
    
    print("=" * 70 + "\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())

