#!/usr/bin/env python3
"""
静态验证 TemporalTransformer 集成（不依赖运行环境）
只检查代码结构和语法
"""

import os
import re
import ast

def check_file_exists(filepath):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ 文件存在: {filepath}")
        return True
    else:
        print(f"❌ 文件不存在: {filepath}")
        return False

def check_syntax(filepath):
    """检查 Python 语法"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"✅ 语法正确: {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误: {filepath}")
        print(f"   行 {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"⚠️  无法解析: {filepath} - {e}")
        return False

def check_imports(filepath, expected_imports):
    """检查是否包含预期的导入"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        all_found = True
        for imp in expected_imports:
            if imp in content:
                print(f"✅ 找到导入: {imp}")
            else:
                print(f"❌ 缺少导入: {imp}")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"⚠️  无法读取文件: {filepath} - {e}")
        return False

def check_patterns(filepath, patterns, description):
    """检查代码模式"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        all_found = True
        for pattern, name in patterns:
            if re.search(pattern, content):
                print(f"✅ 找到 {name}")
            else:
                print(f"❌ 未找到 {name}")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"⚠️  无法读取文件: {filepath} - {e}")
        return False

def check_no_linear_mm_projector(filepath):
    """检查是否还有 nn.Linear 用于 mm_projector"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        found_linear = False
        for i, line in enumerate(lines, 1):
            # 查找 mm_projector = nn.Linear 的模式
            if 'mm_projector' in line and 'nn.Linear' in line:
                print(f"⚠️  第 {i} 行可能仍有 nn.Linear: {line.strip()}")
                found_linear = True
        
        if not found_linear:
            print(f"✅ 未发现 mm_projector = nn.Linear 的模式")
        
        return not found_linear
    except Exception as e:
        print(f"⚠️  无法读取文件: {filepath} - {e}")
        return False

def main():
    """运行静态验证"""
    print("=" * 60)
    print("TemporalTransformer 集成静态验证")
    print("=" * 60)
    
    results = []
    
    # 1. 检查 TemporalTransformer 文件
    print("\n" + "-" * 60)
    print("检查 1: temporal_transformer.py")
    print("-" * 60)
    filepath = "video_chatgpt/model/temporal_transformer.py"
    exists = check_file_exists(filepath)
    results.append(("temporal_transformer.py 存在", exists))
    
    if exists:
        syntax_ok = check_syntax(filepath)
        results.append(("temporal_transformer.py 语法", syntax_ok))
        
        # 检查关键组件
        patterns = [
            (r'class TemporalTransformer', 'TemporalTransformer 类'),
            (r'def __init__', '__init__ 方法'),
            (r'def forward', 'forward 方法'),
            (r'temporal_pos_embed', '位置编码'),
            (r'TransformerEncoder', 'TransformerEncoder'),
            (r'output_proj', '输出投影层'),
        ]
        patterns_ok = check_patterns(filepath, patterns, "关键组件")
        results.append(("temporal_transformer.py 组件", patterns_ok))
    
    # 2. 检查 video_chatgpt.py
    print("\n" + "-" * 60)
    print("检查 2: video_chatgpt.py")
    print("-" * 60)
    filepath = "video_chatgpt/model/video_chatgpt.py"
    exists = check_file_exists(filepath)
    results.append(("video_chatgpt.py 存在", exists))
    
    if exists:
        syntax_ok = check_syntax(filepath)
        results.append(("video_chatgpt.py 语法", syntax_ok))
        
        # 检查导入
        expected_imports = [
            "from video_chatgpt.model.temporal_transformer import TemporalTransformer"
        ]
        imports_ok = check_imports(filepath, expected_imports)
        results.append(("video_chatgpt.py 导入", imports_ok))
        
        # 检查是否使用了 TemporalTransformer
        patterns = [
            (r'TemporalTransformer\(', 'TemporalTransformer 实例化'),
        ]
        patterns_ok = check_patterns(filepath, patterns, "TemporalTransformer 使用")
        results.append(("video_chatgpt.py 使用 TemporalTransformer", patterns_ok))
        
        # 检查是否还有 nn.Linear
        no_linear = check_no_linear_mm_projector(filepath)
        results.append(("video_chatgpt.py 无 nn.Linear", no_linear))
    
    # 3. 检查 model_utils.py
    print("\n" + "-" * 60)
    print("检查 3: model_utils.py")
    print("-" * 60)
    filepath = "video_chatgpt/eval/model_utils.py"
    exists = check_file_exists(filepath)
    results.append(("model_utils.py 存在", exists))
    
    if exists:
        syntax_ok = check_syntax(filepath)
        results.append(("model_utils.py 语法", syntax_ok))
        
        # 检查导入
        expected_imports = [
            "from video_chatgpt.model.temporal_transformer import TemporalTransformer"
        ]
        imports_ok = check_imports(filepath, expected_imports)
        results.append(("model_utils.py 导入", imports_ok))
        
        # 检查是否使用了 TemporalTransformer
        patterns = [
            (r'TemporalTransformer\(', 'TemporalTransformer 实例化'),
        ]
        patterns_ok = check_patterns(filepath, patterns, "TemporalTransformer 使用")
        results.append(("model_utils.py 使用 TemporalTransformer", patterns_ok))
        
        # 检查是否还有 nn.Linear
        no_linear = check_no_linear_mm_projector(filepath)
        results.append(("model_utils.py 无 nn.Linear", no_linear))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n总计: {passed}/{total} 检查通过")
    
    if passed == total:
        print("\n🎉 所有静态检查通过！")
        print("\n注意：这只是静态检查，实际运行还需要：")
        print("  1. 安装依赖包（torch, transformers 等）")
        print("  2. 运行 verify_temporal_transformer.py 进行运行时测试")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个检查失败")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

