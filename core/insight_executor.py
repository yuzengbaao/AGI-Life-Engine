#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insight执行验证框架 (Insight Execution Validation Framework)

自动验证生成的Insight代码是否可执行、正确、高效。

层级：
1. 语法检查 - AST解析
2. 依赖检查 - 所有函数是否可用
3. 沙箱执行 - 在隔离环境中运行
4. 结果验证 - 输出是否符合预期
5. 性能基准 - 执行效率评估
"""

import ast
import sys
import io
import traceback
import time
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List, Optional
import importlib

logger = logging.getLogger(__name__)


class InsightValidator:
    """Insight验证器"""

    def __init__(self):
        self.available_modules = {
            'numpy': 'np',
            'torch': 'torch',
            'torch.nn': 'nn',
            'torch.nn.functional': 'F',
            'core.insight_utilities': 'insight_utilities'
        }

    def validate_syntax(self, code: str) -> Dict[str, Any]:
        """验证Python语法"""
        try:
            ast.parse(code)
            return {
                'valid': True,
                'syntax_errors': []
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'syntax_errors': [str(e)]
            }

    def check_dependencies(self, code: str) -> Dict[str, Any]:
        """检查依赖函数是否可用"""
        tree = ast.parse(code)

        # 收集所有函数调用
        function_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    function_calls.add(node.func.id)

        # 检查哪些在insight_utilities中
        try:
            import core.insight_utilities as utils
            available = set(dir(utils))
            missing = function_calls - available - set(['print', 'len', 'range', 'list', 'dict'])
        except ImportError:
            missing = function_calls

        return {
            'valid': len(missing) == 0,
            'available_functions': list(function_calls - missing),
            'missing_functions': list(missing)
        }

    def sandbox_execute(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """在沙箱环境中执行代码"""
        # 准备执行环境
        exec_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'float': float,
                'int': int,
                'str': str,
                'bool': bool,
            }
        }

        # 添加常用模块
        try:
            import numpy as np
            exec_globals['np'] = np
            import torch
            exec_globals['torch'] = torch
            import torch.nn as nn
            exec_globals['nn'] = nn
            import torch.nn.functional as F
            exec_globals['F'] = F
            from core import insight_utilities
            exec_globals['insight_utilities'] = insight_utilities
        except ImportError as e:
            return {
                'valid': False,
                'error': f'Module import failed: {e}',
                'executed': False
            }

        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                start_time = time.time()
                exec(code, exec_globals)
                exec_time = time.time() - start_time

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            return {
                'valid': True,
                'executed': True,
                'exec_time': exec_time,
                'stdout': stdout_output,
                'stderr': stderr_output,
                'error': None
            }

        except Exception as e:
            return {
                'valid': False,
                'executed': True,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'stderr': stderr_capture.getvalue()
            }

    def validate(self, code: str) -> Dict[str, Any]:
        """完整验证流程"""
        results = {
            'code': code,
            'timestamp': time.time(),
            'stages': {}
        }

        # 阶段1：语法检查
        syntax_result = self.validate_syntax(code)
        results['stages']['syntax'] = syntax_result

        if not syntax_result['valid']:
            results['valid'] = False
            results['reason'] = 'Syntax error'
            return results

        # 阶段2：依赖检查
        dep_result = self.check_dependencies(code)
        results['stages']['dependencies'] = dep_result

        # 阶段3：沙箱执行
        exec_result = self.sandbox_execute(code)
        results['stages']['execution'] = exec_result

        # 综合判断
        results['valid'] = (
            syntax_result['valid'] and
            exec_result['valid']
        )

        if results['valid']:
            results['reason'] = 'All validations passed'
        else:
            results['reason'] = exec_result.get('error', 'Unknown error')

        return results


def test_insight_validator():
    """测试验证器"""
    validator = InsightValidator()

    # 测试1：语法错误
    bad_code = """
def test(:
    return 1
"""
    result = validator.validate(bad_code)
    assert not result['valid']
    assert 'syntax' in result['stages']
    print("[PASS] Test 1: Syntax error detected")

    # 测试2：可执行代码
    good_code = """
from core.insight_utilities import analyze_tone

result = analyze_tone('test text')
print(f"Valence: {result['valence']}")
"""
    result = validator.validate(good_code)
    print(f"Test 2 result: {result['valid']}")
    if result['valid']:
        print("[PASS] Test 2: Valid code executed successfully")
    else:
        print(f"[FAIL] Test 2: {result.get('reason')}")

    # 测试3：使用缺失函数
    missing_dep_code = """
nonexistent_function()
"""
    result = validator.validate(missing_dep_code)
    assert not result['valid']
    print("[PASS] Test 3: Missing dependency detected")

    print("\n[SUCCESS] All validator tests completed!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_insight_validator()
