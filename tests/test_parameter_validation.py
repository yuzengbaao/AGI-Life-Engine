"""
Unit Tests for Tool Parameter Validator - 参数验证器单元测试

测试覆盖：
1. 引号配对测试（10个用例）
2. 括号匹配测试（10个用例）
3. 序列化测试（5个用例）
4. 边界条件测试（5个用例）
5. 工具特定验证（5个用例）

总计：35+测试用例，覆盖率目标>=90%

Author: AGI System Developer
Created: 2025-11-16
"""

import pytest
import json
from tool_parameter_validator import (
    check_string_completeness,
    check_bracket_matching,
    validate_tool_params,
    safe_serialize_params,
    validate_and_serialize,
    validate_math_expression,
    validate_python_code_string,
    validate_file_path,
    batch_validate_params
)


# ==================== 引号配对测试（10个用例） ====================

class TestStringCompleteness:
    """测试字符串完整性检查"""
    
    def test_valid_no_quotes(self):
        """测试：没有引号的普通字符串"""
        result, msg = check_string_completeness("hello world")
        assert result == True
        assert msg == ""
    
    def test_valid_double_quotes_paired(self):
        """测试：正确配对的双引号"""
        result, msg = check_string_completeness('say "hello"')
        assert result == True
    
    def test_valid_single_quotes_paired(self):
        """测试：正确配对的单引号"""
        result, msg = check_string_completeness("say 'hello'")
        assert result == True
    
    def test_valid_mixed_quotes_paired(self):
        """测试：混合引号正确配对"""
        result, msg = check_string_completeness('''say "hello" and 'world' ''')
        assert result == True
    
    def test_invalid_single_quote_unclosed(self):
        """测试：单引号未闭合"""
        result, msg = check_string_completeness("hello 'world")
        assert result == False
        assert "单引号" in msg
    
    def test_invalid_double_quote_unclosed(self):
        """测试：双引号未闭合"""
        result, msg = check_string_completeness('say "hello')
        assert result == False
        assert "双引号" in msg
    
    def test_invalid_multiple_single_quotes_odd(self):
        """测试：奇数个单引号"""
        result, msg = check_string_completeness("it's a 'nice' day")
        assert result == False  # 3个单引号
    
    def test_invalid_multiple_double_quotes_odd(self):
        """测试：奇数个双引号"""
        result, msg = check_string_completeness('"hello" world "test')
        assert result == False  # 3个双引号
    
    def test_valid_empty_string(self):
        """测试：空字符串"""
        result, msg = check_string_completeness("")
        assert result == True
    
    def test_valid_escaped_quotes_even(self):
        """测试：转义引号（偶数个）"""
        result, msg = check_string_completeness('say "hello \\"world\\""')
        assert result == True  # 4个双引号（包含转义）


# ==================== 括号匹配测试（10个用例） ====================

class TestBracketMatching:
    """测试括号匹配检查"""
    
    def test_valid_parentheses_matched(self):
        """测试：圆括号正确匹配"""
        result, msg = check_bracket_matching("func(a, b)")
        assert result == True
    
    def test_valid_square_brackets_matched(self):
        """测试：方括号正确匹配"""
        result, msg = check_bracket_matching("array[0][1]")
        assert result == True
    
    def test_valid_curly_braces_matched(self):
        """测试：花括号正确匹配"""
        result, msg = check_bracket_matching("dict{key: value}")
        assert result == True
    
    def test_valid_nested_brackets(self):
        """测试：嵌套括号正确匹配"""
        result, msg = check_bracket_matching("func(array[dict{key}])")
        assert result == True
    
    def test_invalid_parentheses_unclosed(self):
        """测试：圆括号未闭合"""
        result, msg = check_bracket_matching("func(a, b")
        assert result == False
        assert "圆括号" in msg or "括号顺序" in msg
    
    def test_invalid_square_bracket_extra_close(self):
        """测试：多余的方括号闭合"""
        result, msg = check_bracket_matching("array[0]]")
        assert result == False
        assert "方括号" in msg or "多余" in msg
    
    def test_invalid_curly_brace_mismatched_count(self):
        """测试：花括号数量不匹配"""
        result, msg = check_bracket_matching("dict{key: value")
        assert result == False
        assert "花括号" in msg or "括号顺序" in msg
    
    def test_invalid_wrong_bracket_type_order(self):
        """测试：括号类型顺序错误"""
        result, msg = check_bracket_matching("func(array[test)")
        assert result == False
        assert "不匹配" in msg or "类型" in msg
    
    def test_valid_no_brackets(self):
        """测试：无括号的字符串"""
        result, msg = check_bracket_matching("hello world")
        assert result == True
    
    def test_invalid_nested_brackets_wrong_order(self):
        """测试：嵌套括号顺序错误"""
        result, msg = check_bracket_matching("func([)]")
        assert result == False
        assert "不匹配" in msg or "类型" in msg


# ==================== 参数验证测试（10个用例） ====================

class TestValidateToolParams:
    """测试工具参数验证"""
    
    def test_valid_math_expression(self):
        """测试：有效的数学表达式"""
        result, msg = validate_tool_params('math', {'expression': '2^10 + sqrt(144)'})
        assert result == True
        assert msg == ""
    
    def test_invalid_math_expression_unclosed_paren(self):
        """测试：数学表达式括号未闭合（原失败用例）"""
        result, msg = validate_tool_params('math', {'expression': '2^10 + sqrt(144'})
        assert result == False
        assert '圆括号' in msg
    
    def test_valid_python_code(self):
        """测试：有效的Python代码"""
        result, msg = validate_tool_params('code', {'code': "import math; print(f'圆周率: {math.pi}')"})
        assert result == True
    
    def test_invalid_python_code_unclosed_quote(self):
        """测试：Python代码引号未闭合（原失败用例）"""
        result, msg = validate_tool_params('code', {'code': "print('hello"})
        assert result == False
        assert '引号' in msg
    
    def test_valid_file_path(self):
        """测试：有效的文件路径"""
        result, msg = validate_tool_params('file', {'file_path': 'D:\\TRAE_PROJECT\\AGI\\test.py'})
        assert result == True
    
    def test_valid_multiple_params(self):
        """测试：多个参数都有效"""
        params = {
            'operation': 'calculate',
            'expression': 'sin(pi/2)',
            'precision': '10'
        }
        result, msg = validate_tool_params('math', params)
        assert result == True
    
    def test_invalid_one_param_fails(self):
        """测试：多个参数中有一个失败"""
        params = {
            'operation': 'calculate',
            'expression': 'sqrt(144',  # 错误
            'precision': '10'
        }
        result, msg = validate_tool_params('math', params)
        assert result == False
    
    def test_valid_non_string_params_ignored(self):
        """测试：非字符串参数被忽略"""
        params = {
            'expression': '2+2',
            'precision': 10,  # int类型
            'use_cache': True,  # bool类型
            'constants': ['pi', 'e']  # list类型
        }
        result, msg = validate_tool_params('math', params)
        assert result == True
    
    def test_valid_empty_params_dict(self):
        """测试：空参数字典"""
        result, msg = validate_tool_params('test', {})
        assert result == True
    
    def test_invalid_nested_quote_and_bracket(self):
        """测试：嵌套引号和括号都有问题"""
        result, msg = validate_tool_params('code', {'code': "print(\"hello world')"})
        assert result == False


# ==================== 序列化测试（5个用例） ====================

class TestSafeSerializeParams:
    """测试参数序列化"""
    
    def test_serialize_simple_string(self):
        """测试：序列化简单字符串"""
        params = {'name': 'Alice'}
        json_str = safe_serialize_params(params)
        parsed = json.loads(json_str)
        assert parsed['name'] == 'Alice'
    
    def test_serialize_nested_quotes(self):
        """测试：序列化嵌套引号"""
        params = {'code': "print('hello')"}
        json_str = safe_serialize_params(params)
        parsed = json.loads(json_str)
        assert parsed['code'] == "print('hello')"
    
    def test_serialize_complex_expression(self):
        """测试：序列化复杂表达式"""
        params = {'expression': '2^10 + sqrt(144)'}
        json_str = safe_serialize_params(params)
        parsed = json.loads(json_str)
        assert parsed['expression'] == '2^10 + sqrt(144)'
    
    def test_serialize_chinese_characters(self):
        """测试：序列化中文字符"""
        params = {'message': '你好世界'}
        json_str = safe_serialize_params(params)
        parsed = json.loads(json_str)
        assert parsed['message'] == '你好世界'
    
    def test_serialize_nested_dict(self):
        """测试：序列化嵌套字典"""
        params = {
            'config': {
                'name': 'test',
                'values': [1, 2, 3],
                'options': {'debug': True}
            }
        }
        json_str = safe_serialize_params(params)
        parsed = json.loads(json_str)
        assert parsed['config']['name'] == 'test'
        assert parsed['config']['values'] == [1, 2, 3]


# ==================== 边界条件测试（5个用例） ====================

class TestBoundaryConditions:
    """测试边界条件"""
    
    def test_empty_string_param(self):
        """测试：空字符串参数"""
        result, msg = validate_tool_params('test', {'value': ''})
        assert result == True
    
    def test_very_long_string(self):
        """测试：超长字符串（1000字符）"""
        long_string = 'a' * 1000
        result, msg = validate_tool_params('test', {'data': long_string})
        assert result == True
    
    def test_unicode_characters(self):
        """测试：Unicode字符（emoji等）"""
        result, msg = validate_tool_params('test', {'msg': '🚀 AGI系统 ✅'})
        assert result == True
    
    def test_special_characters(self):
        """测试：特殊字符"""
        result, msg = validate_tool_params('test', {'text': '!@#$%^&*()_+-=[]{}|;:,.<>?/~`'})
        assert result == True
    
    def test_newlines_and_tabs(self):
        """测试：换行符和制表符"""
        result, msg = validate_tool_params('test', {'code': 'def func():\n\tprint("hello")'})
        assert result == True


# ==================== 组合功能测试（5个用例） ====================

class TestValidateAndSerialize:
    """测试组合验证和序列化"""
    
    def test_valid_params_full_flow(self):
        """测试：有效参数完整流程"""
        is_valid, json_str, error = validate_and_serialize('math', {'expression': '2+2'})
        assert is_valid == True
        assert json_str is not None
        assert error == ""
        parsed = json.loads(json_str)
        assert parsed['expression'] == '2+2'
    
    def test_invalid_params_validation_fails(self):
        """测试：无效参数验证失败"""
        is_valid, json_str, error = validate_and_serialize('math', {'expression': 'sqrt(144'})
        assert is_valid == False
        assert json_str is None
        assert '圆括号' in error
    
    def test_valid_multiple_tools_batch(self):
        """测试：批量验证多个工具（原失败用例）"""
        tool_calls = [
            {'tool': 'math', 'params': {'expression': '2^10 + sqrt(144)'}},  # 原失败
            {'tool': 'code', 'params': {'code': "import math; print(math.pi)"}},
            {'tool': 'file', 'params': {'path': 'test.py'}}
        ]
        result = batch_validate_params(tool_calls)
        assert result['passed'] == 3
        assert result['failed'] == 0
    
    def test_mixed_valid_invalid_batch(self):
        """测试：批量验证混合有效和无效"""
        tool_calls = [
            {'tool': 'math', 'params': {'expression': '2+2'}},  # 有效
            {'tool': 'math', 'params': {'expression': 'sqrt(144'}},  # 无效
            {'tool': 'code', 'params': {'code': "print('hello')"}},  # 有效
            {'tool': 'code', 'params': {'code': "print('world"}},  # 无效
        ]
        result = batch_validate_params(tool_calls)
        assert result['passed'] == 2
        assert result['failed'] == 2
        assert len(result['failures']) == 2
    
    def test_empty_tool_calls_batch(self):
        """测试：空工具调用列表"""
        result = batch_validate_params([])
        assert result['total'] == 0
        assert result['passed'] == 0
        assert result['failed'] == 0


# ==================== 工具特定验证测试（5个用例） ====================

class TestToolSpecificValidators:
    """测试工具特定的验证器"""
    
    def test_validate_math_expression_valid(self):
        """测试：验证有效数学表达式"""
        result, msg = validate_math_expression('2^10 + sqrt(144)')
        assert result == True
    
    def test_validate_math_expression_empty(self):
        """测试：验证空数学表达式"""
        result, msg = validate_math_expression('')
        assert result == False
        assert '不能为空' in msg
    
    def test_validate_python_code_valid(self):
        """测试：验证有效Python代码字符串"""
        result, msg = validate_python_code_string("import math; print(math.pi)")
        assert result == True
    
    def test_validate_python_code_empty(self):
        """测试：验证空Python代码"""
        result, msg = validate_python_code_string('   ')
        assert result == False
        assert '不能为空' in msg
    
    def test_validate_file_path_valid(self):
        """测试：验证有效文件路径"""
        result, msg = validate_file_path('D:\\TRAE_PROJECT\\AGI\\test.py')
        assert result == True


# ==================== 性能测试（可选） ====================

class TestPerformance:
    """性能测试（确保验证不会太慢）"""
    
    def test_validation_performance_100_calls(self):
        """测试：100次验证调用的性能"""
        import time
        
        params = {'expression': '2^10 + sqrt(144)'}
        
        start = time.time()
        for _ in range(100):
            validate_tool_params('math', params)
        elapsed = time.time() - start
        
        # 期望100次验证在1秒内完成
        assert elapsed < 1.0, f"100次验证耗时{elapsed:.3f}秒，超过1秒阈值"
    
    def test_serialization_performance_100_calls(self):
        """测试：100次序列化调用的性能"""
        import time
        
        params = {
            'expression': '2^10 + sqrt(144)',
            'precision': 10,
            'use_cache': True
        }
        
        start = time.time()
        for _ in range(100):
            safe_serialize_params(params)
        elapsed = time.time() - start
        
        # 期望100次序列化在0.5秒内完成
        assert elapsed < 0.5, f"100次序列化耗时{elapsed:.3f}秒，超过0.5秒阈值"


# ==================== 运行测试 ====================

if __name__ == "__main__":
    # 使用pytest运行所有测试
    pytest.main([__file__, '-v', '--tb=short'])
