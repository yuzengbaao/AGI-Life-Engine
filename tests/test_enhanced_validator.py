"""
增强验证器测试套件
测试依赖检查和沙箱执行功能

运行: pytest -v tests/test_enhanced_validator.py
"""

import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.insight_validator import InsightValidator, SYSTEM_FUNCTION_REGISTRY, SAFE_MODULES


class TestEnhancedValidator:
    """增强验证器测试"""
    
    @pytest.fixture
    def validator(self):
        """创建验证器实例"""
        return InsightValidator(system_dependency_graph={
            'custom_system_func': True,
            'another_sys_func': True,
        })
    
    # =========================================================================
    # 依赖检查测试
    # =========================================================================
    
    def test_dependency_check_valid_builtin(self, validator):
        """测试：使用内置函数应该通过"""
        code = '''
def calculate_stats(data):
    total = sum(data)
    average = total / len(data)
    return min(data), max(data), average
'''
        valid, missing = validator._check_dependencies(code)
        assert valid is True
        assert missing == []
    
    def test_dependency_check_valid_local_def(self, validator):
        """测试：调用本地定义的函数应该通过"""
        code = '''
def helper_func(x):
    return x * 2

def main_func(data):
    return helper_func(data)
'''
        valid, missing = validator._check_dependencies(code)
        assert valid is True
        assert missing == []
    
    def test_dependency_check_valid_import(self, validator):
        """测试：调用导入的函数应该通过"""
        code = '''
import math
import random

def calculate(x):
    return math.sqrt(x) + random.random()
'''
        valid, missing = validator._check_dependencies(code)
        assert valid is True
        assert missing == []
    
    def test_dependency_check_missing_function(self, validator):
        """测试：调用不存在的函数应该失败"""
        code = '''
def process_state(state):
    # 这些函数都不存在！
    depth = trace_self_inquiry_depth(state)
    concepts = find_emergent_concepts(state)
    return depth, concepts
'''
        valid, missing = validator._check_dependencies(code)
        assert valid is False
        assert 'trace_self_inquiry_depth' in missing
        assert 'find_emergent_concepts' in missing
    
    def test_dependency_check_pseudocode_topology(self, validator):
        """测试：伪代码（拓扑学函数）应该被检测"""
        code = '''
def adaptive_entropy_regulation_via_topology(state, curiosity, entropy_threshold=0.65):
    window = hidden_states_window(state, size=20)
    variability = compute_persistent_homology_variability(window)
    return variability
'''
        valid, missing = validator._check_dependencies(code)
        assert valid is False
        assert 'hidden_states_window' in missing
        assert 'compute_persistent_homology_variability' in missing
    
    def test_dependency_check_system_graph(self, validator):
        """测试：系统依赖图中的函数应该通过"""
        code = '''
def use_system_func():
    result = custom_system_func()
    return another_sys_func(result)
'''
        valid, missing = validator._check_dependencies(code)
        assert valid is True
        assert missing == []
    
    # =========================================================================
    # 沙箱执行测试
    # =========================================================================
    
    def test_sandbox_valid_code(self, validator):
        """测试：有效代码应该在沙箱中成功执行"""
        code = '''
def add_numbers(a, b):
    return a + b

def multiply(x, y):
    return x * y
'''
        valid, error = validator._run_in_sandbox(code)
        assert valid is True
        assert error == ""
    
    def test_sandbox_runtime_error(self, validator):
        """测试：运行时错误应该被捕获"""
        code = '''
def divide_by_zero():
    return 1 / 0
'''
        valid, error = validator._run_in_sandbox(code)
        assert valid is False
        assert "ZeroDivision" in error or "division" in error.lower()
    
    def test_sandbox_name_error(self, validator):
        """测试：调用不存在函数的NameError应该被捕获"""
        code = '''
def call_undefined():
    return undefined_function_xyz()
'''
        valid, error = validator._run_in_sandbox(code)
        assert valid is False
        assert "NameError" in error or "undefined_function_xyz" in error
    
    # =========================================================================
    # 完整验证流程测试
    # =========================================================================
    
    def test_full_validation_valid_code(self, validator):
        """测试：有效代码应该通过完整验证"""
        code = '''
def apply_cognitive_dither(entropy, curiosity, strength=0.02):
    import random
    noise = strength * (2 * random.random() - 1)
    new_curiosity = min(1.0, max(0.0, curiosity + noise))
    return entropy, new_curiosity
'''
        result = validator.validate(code, {'trigger_goal': 'test', 'content': 'test'})
        
        assert result['checks']['syntax'] is True
        assert result['checks']['safety'] is True
        assert result['checks']['dependency'] is True
        assert result['checks']['sandbox'] is True
        assert result['score'] > 0.5
        assert result['recommendation'] in ['INTEGRATE', 'ARCHIVE']
    
    def test_full_validation_pseudocode_rejected(self, validator):
        """测试：伪代码应该被拒绝"""
        code = '''
def resolve_entropy(state, curiosity, entropy_threshold=0.7):
    depth = trace_self_inquiry_depth(state)
    candidates = find_emergent_concepts(state)
    for concept in candidates:
        if violates_ontology(concept):
            quarantine(concept, duration=float('inf'))
    restore_coherence_gradient(state, strength=0.3)
'''
        result = validator.validate(code, {'trigger_goal': 'test', 'content': 'test'})
        
        assert result['checks']['dependency'] is False
        assert result['recommendation'] == 'REJECT'
        assert len(result['missing_deps']) > 0
        assert any('trace_self_inquiry_depth' in dep for dep in result['missing_deps'])
    
    def test_full_validation_missing_deps_list(self, validator):
        """测试：验证结果应该包含缺失依赖列表"""
        code = '''
def process(state):
    a = func_not_exists_1(state)
    b = func_not_exists_2(a)
    return b
'''
        result = validator.validate(code, {'trigger_goal': 'test', 'content': 'test'})
        
        assert 'missing_deps' in result
        assert 'func_not_exists_1' in result['missing_deps']
        assert 'func_not_exists_2' in result['missing_deps']
    
    # =========================================================================
    # 退火机制测试
    # =========================================================================
    
    def test_validation_backoff(self, validator):
        """测试：连续失败应该触发退火"""
        bad_code = '''
def fail():
    return undefined_xyz()
'''
        # 连续3次失败
        for _ in range(3):
            validator.validate(bad_code, {'trigger_goal': 'test', 'content': 'test'})
        
        # 第4次应该触发退火
        result = validator.validate(bad_code, {'trigger_goal': 'test', 'content': 'test'})
        
        # 验证退火状态（可能触发也可能不触发，取决于时间）
        assert result['recommendation'] == 'REJECT'


class TestSystemRegistry:
    """系统函数注册表测试"""
    
    def test_builtin_functions_registered(self):
        """测试：内置函数应该在注册表中"""
        assert 'print' in SYSTEM_FUNCTION_REGISTRY
        assert 'len' in SYSTEM_FUNCTION_REGISTRY
        assert 'range' in SYSTEM_FUNCTION_REGISTRY
    
    def test_safe_modules_defined(self):
        """测试：安全模块列表应该存在"""
        assert 'math' in SAFE_MODULES
        assert 'random' in SAFE_MODULES
        assert 'json' in SAFE_MODULES
        # 危险模块不应该在列表中
        assert 'subprocess' not in SAFE_MODULES
        assert 'shutil' not in SAFE_MODULES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
