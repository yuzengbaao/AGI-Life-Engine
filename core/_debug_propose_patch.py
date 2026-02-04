#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""详细调试SelfModifyingEngine.propose_patch流程"""
import sys
import logging
sys.path.insert(0, 'd:/TRAE_PROJECT/AGI')

# 启用详细日志
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from core.self_modifying_engine import SelfModifyingEngine
from pathlib import Path

engine = SelfModifyingEngine()

# 测试模块
target_module = "core.math_utils"
file_path = Path("d:/TRAE_PROJECT/AGI/core/math_utils.py")

print(f"Testing module: {target_module}")
print(f"File exists: {file_path.exists()}")

# 读取原始代码
with open(file_path, 'r', encoding='utf-8') as f:
    original_code = f.read()
print(f"Original code lines: {len(original_code.splitlines())}")

# 直接测试patch_generator
print("\n=== Testing patch_generator directly ===")
modified = engine.patch_generator.generate_patch(
    old_code=original_code,
    target_desc="添加性能监控装饰器或增强日志, goal=performance",
    strategy="auto"
)
print(f"patch_generator result: {'Yes' if modified else 'No'}")
if modified:
    print(f"Modified code lines: {len(modified.splitlines())}")
    print(f"Same as original: {modified == original_code}")
    
    # 计算diff
    from difflib import unified_diff
    diff = list(unified_diff(
        original_code.splitlines(keepends=True),
        modified.splitlines(keepends=True)
    ))
    print(f"Diff lines: {len(diff)}")

# 调用完整的 propose_patch
print("\n=== Testing propose_patch ===")
record = engine.propose_patch(
    target_module=target_module,
    issue_description="[Test] 添加性能监控",
    use_llm=False,
    patch_strategy="auto"
)
print(f"propose_patch result: {record}")
if record:
    print(f"Record ID: {getattr(record, 'id', 'N/A')}")
    print(f"Diff preview: {record.diff[:200] if record.diff else 'None'}...")
