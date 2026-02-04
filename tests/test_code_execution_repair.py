"""
测试 code_execution 工具修复效果

验证:
1. validate_python_code() 函数正确性
2. 原失败用例（引号不匹配）现在被拦截
3. 各种语法错误正确检测
4. 正确的代码正常执行
5. 边界条件和错误处理
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_tools_collection import validate_python_code, CodeExecutionTool


class TestValidatePythonCode:
    """测试 validate_python_code() 函数"""
    
    def test_valid_simple_code(self):
        """测试简单有效代码"""
        valid_codes = [
            "print('hello')",
            "x = 1 + 2",
            "for i in range(10): print(i)",
            "def func(): pass",
            "import math\nprint(math.pi)",
        ]
        
        for code in valid_codes:
            is_valid, error = validate_python_code(code)
            assert is_valid, f"代码应该有效: {code}\n错误: {error}"
            assert error == ""
    
    def test_invalid_unclosed_string(self):
        """测试未闭合字符串"""
        # 单引号未闭合
        is_valid, error = validate_python_code("print('hello")
        assert not is_valid
        assert "字符串不完整" in error or "引号" in error
        
        # 双引号未闭合
        is_valid, error = validate_python_code('print("world')
        assert not is_valid
        assert "字符串不完整" in error or "引号" in error
    
    def test_invalid_bracket_mismatch(self):
        """测试括号不匹配"""
        # 缺少右括号
        is_valid, error = validate_python_code("print('hello'")
        assert not is_valid
        assert "括号不匹配" in error
        
        # 缺少左括号
        is_valid, error = validate_python_code("print 'hello')")
        assert not is_valid
        assert "括号不匹配" in error or "语法错误" in error
    
    def test_invalid_syntax_error(self):
        """测试语法错误"""
        invalid_codes = [
            "if True print('hello')",  # 缺少冒号
            "def func( pass",  # 括号不匹配
            "for i in range(10) print(i)",  # 缺少冒号
            "x = ",  # 不完整的赋值
        ]
        
        for code in invalid_codes:
            is_valid, error = validate_python_code(code)
            assert not is_valid, f"代码应该无效: {code}"
            assert len(error) > 0
    
    def test_multiline_code(self):
        """测试多行代码"""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(5))
"""
        is_valid, error = validate_python_code(code)
        assert is_valid, f"多行代码应该有效\n错误: {error}"
    
    def test_code_with_fstring(self):
        """测试 f-string"""
        # 正确的 f-string
        is_valid, error = validate_python_code("import math\nprint(f'圆周率: {math.pi}')")
        assert is_valid, f"f-string 应该有效\n错误: {error}"
        
        # 未闭合的 f-string
        is_valid, error = validate_python_code("print(f'圆周率: {math.pi}'")
        assert not is_valid
        assert "括号不匹配" in error or "引号" in error
    
    def test_nested_structures(self):
        """测试嵌套结构"""
        code = "data = {'key': [1, 2, {'nested': (3, 4)}]}"
        is_valid, error = validate_python_code(code)
        assert is_valid, f"嵌套结构应该有效\n错误: {error}"


class TestCodeExecutionToolIntegration:
    """测试 CodeExecutionTool 工具集成"""
    
    def setup_method(self):
        """每个测试前初始化工具"""
        self.tool = CodeExecutionTool()
    
    def test_execute_valid_code(self):
        """测试执行有效代码"""
        code = "print('Hello, World!')"
        result = self.tool.execute(code=code)
        
        assert result.success is True
        assert 'Hello, World!' in result.data['output']
        assert result.data['code'] == code
    
    def test_original_failure_case_unclosed_string(self):
        """测试原失败用例 - 未闭合字符串"""
        # 原失败: print(f'圆周率: {math.pi}'  (缺少右括号和引号)
        code = "print(f'圆周率: {math.pi}'"
        result = self.tool.execute(code=code)
        
        # 现在应该被验证拦截
        assert result.success is False
        assert "代码验证失败" in result.error
        assert "括号" in result.error or "引号" in result.error
    
    def test_correct_fstring_executes(self):
        """测试正确的 f-string 能执行"""
        code = "import math\nprint(f'圆周率: {math.pi}')"
        result = self.tool.execute(code=code)
        
        assert result.success is True
        assert '圆周率' in result.data['output']
        assert '3.14' in result.data['output']
    
    def test_syntax_error_caught_before_execution(self):
        """测试语法错误在执行前被捕获"""
        code = "if True print('hello')"  # 缺少冒号
        result = self.tool.execute(code=code)
        
        assert result.success is False
        assert "代码验证失败" in result.error
        assert "语法错误" in result.error
    
    def test_bracket_mismatch_caught(self):
        """测试括号不匹配被捕获"""
        code = "print('test'"  # 缺少右括号
        result = self.tool.execute(code=code)
        
        assert result.success is False
        assert "括号不匹配" in result.error
    
    def test_execute_with_variables(self):
        """测试执行带变量的代码"""
        code = """
x = 10
y = 20
result = x + y
print(f'Result: {result}')
"""
        result = self.tool.execute(code=code)
        
        assert result.success is True
        assert 'Result: 30' in result.data['output']
        assert 'x' in result.data['variables']
        assert 'result' in result.data['variables']
    
    def test_execute_with_loop(self):
        """测试执行循环"""
        code = "for i in range(3): print(i)"
        result = self.tool.execute(code=code)
        
        assert result.success is True
        assert '0' in result.data['output']
        assert '1' in result.data['output']
        assert '2' in result.data['output']
    
    def test_execute_with_function(self):
        """测试执行函数定义"""
        code = """
def greet(name):
    return f'Hello, {name}!'

print(greet('AGI'))
"""
        result = self.tool.execute(code=code)
        
        assert result.success is True
        assert 'Hello, AGI!' in result.data['output']
    
    def test_runtime_error_after_validation(self):
        """测试运行时错误（语法正确但运行失败）"""
        code = "x = 1 / 0"  # 除零错误
        result = self.tool.execute(code=code)
        
        # 语法验证应该通过
        # 但执行应该失败
        assert result.success is False
        assert "division by zero" in result.error.lower()
    
    def test_execution_time_recorded(self):
        """测试执行时间被记录"""
        code = "print('test')"
        result = self.tool.execute(code=code)
        
        assert result.execution_time is not None
        assert result.execution_time >= 0
    
    def test_missing_code_parameter(self):
        """测试缺少 code 参数"""
        result = self.tool.execute()
        
        assert result.success is False
        assert "缺少必需参数" in result.error


class TestEdgeCases:
    """边界条件测试"""
    
    def test_empty_code(self):
        """测试空代码"""
        is_valid, error = validate_python_code("")
        # 空代码在语法上是有效的
        assert is_valid
    
    def test_whitespace_only(self):
        """测试仅空白字符"""
        is_valid, error = validate_python_code("   \n\t  ")
        assert is_valid
    
    def test_comment_only(self):
        """测试仅注释"""
        is_valid, error = validate_python_code("# This is a comment")
        assert is_valid
    
    def test_very_long_code(self):
        """测试很长的代码"""
        code = "\n".join([f"x{i} = {i}" for i in range(100)])
        is_valid, error = validate_python_code(code)
        assert is_valid
    
    def test_unicode_characters(self):
        """测试 Unicode 字符"""
        code = "变量 = '中文字符串'\nprint(变量)"
        is_valid, error = validate_python_code(code)
        assert is_valid


class TestComparison:
    """对比测试 - 验证功能完整性"""
    
    def test_feature_parity(self):
        """确保新实现支持所有原有功能"""
        tool = CodeExecutionTool()
        
        test_cases = [
            ("print('hello')", True),
            ("x = 1 + 1\nprint(x)", True),
            ("import math\nprint(math.sqrt(16))", True),
            ("for i in range(5): print(i)", True),
            ("[x*2 for x in range(3)]", True),
        ]
        
        passed = 0
        for code, should_succeed in test_cases:
            result = tool.execute(code=code)
            if result.success == should_succeed:
                passed += 1
        
        assert passed == len(test_cases), f"功能对比: {passed}/{len(test_cases)}"


if __name__ == "__main__":
    print("=" * 60)
    print("Code Execution 修复测试")
    print("=" * 60)
    
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-p", "no:warnings"
    ])
