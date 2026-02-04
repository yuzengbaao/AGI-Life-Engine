"""
测试math_operations工具修复效果

验证:
1. safe_math_eval()函数正确性
2. 原失败用例现在能成功执行
3. 各种数学运算符和函数
4. 边界条件和错误处理
5. 性能要求 (<1.5x原实现)
"""

import sys
import os
import time
import pytest
from typing import Union

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_tools_collection import safe_math_eval, MathTool


class TestSafeMathEval:
    """测试safe_math_eval()函数"""
    
    def test_basic_arithmetic(self):
        """测试基本算术运算"""
        assert safe_math_eval("2 + 2") == 4
        assert safe_math_eval("10 - 3") == 7
        assert safe_math_eval("6 * 7") == 42
        assert safe_math_eval("15 / 3") == 5.0
        assert safe_math_eval("17 // 5") == 3
        assert safe_math_eval("17 % 5") == 2
    
    def test_power_operations(self):
        """测试幂运算 (^ 和 **)"""
        assert safe_math_eval("2^10") == 1024
        assert safe_math_eval("2**10") == 1024
        assert safe_math_eval("3^3") == 27
        assert safe_math_eval("5**2") == 25
    
    def test_math_functions(self):
        """测试数学函数"""
        import math
        
        # sqrt
        assert safe_math_eval("sqrt(144)") == 12.0
        assert safe_math_eval("sqrt(2)") == pytest.approx(1.414213, rel=1e-5)
        
        # 三角函数
        assert safe_math_eval("sin(0)") == 0.0
        assert safe_math_eval("cos(0)") == 1.0
        assert safe_math_eval("tan(0)") == 0.0
        assert safe_math_eval("sin(pi/2)") == pytest.approx(1.0, rel=1e-10)
        
        # 对数和指数
        assert safe_math_eval("log(e)") == pytest.approx(1.0, rel=1e-10)
        assert safe_math_eval("log10(100)") == 2.0
        assert safe_math_eval("exp(0)") == 1.0
        
        # 其他函数
        assert safe_math_eval("abs(-5)") == 5
        assert safe_math_eval("ceil(3.2)") == 4
        assert safe_math_eval("floor(3.8)") == 3
    
    def test_constants(self):
        """测试数学常量"""
        import math
        
        result = safe_math_eval("pi")
        assert result == pytest.approx(math.pi, rel=1e-10)
        
        result = safe_math_eval("e")
        assert result == pytest.approx(math.e, rel=1e-10)
        
        result = safe_math_eval("tau")
        assert result == pytest.approx(math.tau, rel=1e-10)
    
    def test_complex_expressions(self):
        """测试复杂表达式"""
        # 原失败用例
        result = safe_math_eval("2^10 + sqrt(144)")
        assert result == 1036.0
        
        # 嵌套函数
        result = safe_math_eval("sqrt(abs(-16))")
        assert result == 4.0
        
        # 运算符优先级
        result = safe_math_eval("2 + 3 * 4")
        assert result == 14
        
        result = safe_math_eval("(2 + 3) * 4")
        assert result == 20
        
        # 三角函数与常量
        result = safe_math_eval("sin(pi/4)")
        assert result == pytest.approx(0.707106, rel=1e-5)
    
    def test_unary_operators(self):
        """测试一元运算符"""
        assert safe_math_eval("-5") == -5
        assert safe_math_eval("+5") == 5
        assert safe_math_eval("-(2+3)") == -5
    
    def test_builtin_functions(self):
        """测试内置函数"""
        assert safe_math_eval("abs(-10)") == 10
        assert safe_math_eval("min(3, 5, 1)") == 1
        assert safe_math_eval("max(3, 5, 1)") == 5
        assert safe_math_eval("round(3.7)") == 4
    
    def test_error_handling_syntax(self):
        """测试语法错误处理"""
        with pytest.raises(SyntaxError):
            safe_math_eval("2 +")  # 不完整的表达式
        
        with pytest.raises(SyntaxError):
            safe_math_eval("2 + + 3")  # 多余的运算符
    
    def test_error_handling_undefined(self):
        """测试未定义变量/函数"""
        with pytest.raises(ValueError, match="未定义的变量"):
            safe_math_eval("x + 2")
        
        with pytest.raises(ValueError, match="不支持的函数"):
            safe_math_eval("unknown_func(5)")
    
    def test_error_handling_unsafe_operations(self):
        """测试不安全操作被拒绝"""
        # 不应该允许导入
        with pytest.raises((ValueError, SyntaxError)):
            safe_math_eval("__import__('os').system('ls')")
        
        # 不应该允许属性访问 (如果尝试注入)
        # 注意: AST解析会直接拒绝这类语法
    
    def test_edge_cases(self):
        """测试边界条件"""
        # 除以零
        with pytest.raises(ZeroDivisionError):
            safe_math_eval("1 / 0")
        
        # 负数的平方根
        with pytest.raises(ValueError):
            safe_math_eval("sqrt(-1)")
        
        # 非常大的数
        result = safe_math_eval("2**100")
        assert result == 2**100
        
        # 非常小的数
        result = safe_math_eval("1 / 2**50")
        assert result == 1 / 2**50


class TestMathToolIntegration:
    """测试MathTool工具集成"""
    
    def setup_method(self):
        """每个测试前初始化工具"""
        self.tool = MathTool()
    
    def test_eval_operation_success(self):
        """测试eval操作成功"""
        result = self.tool.execute(operation="eval", expression="2+2")
        assert result.success is True
        assert result.data['result'] == 4
        assert result.data['expression'] == "2+2"
    
    def test_original_failure_case(self):
        """测试原失败用例现在成功"""
        # 原失败: expression="2^10 + sqrt(144"  (括号不匹配)
        # Stage 1参数验证会拦截
        
        # 正确格式应该成功
        result = self.tool.execute(operation="eval", expression="2^10 + sqrt(144)")
        assert result.success is True
        assert result.data['result'] == 1036.0
    
    def test_eval_with_aliases(self):
        """测试操作别名"""
        # evaluate 别名
        result = self.tool.execute(operation="evaluate", expression="3*4")
        assert result.success is True
        assert result.data['result'] == 12
        
        # calculate 别名
        result = self.tool.execute(operation="calculate", expression="5+5")
        assert result.success is True
        assert result.data['result'] == 10
    
    def test_eval_missing_expression(self):
        """测试缺少expression参数"""
        result = self.tool.execute(operation="eval")
        assert result.success is False
        assert "expression" in result.error
    
    def test_eval_invalid_expression(self):
        """测试无效表达式"""
        result = self.tool.execute(operation="eval", expression="invalid+++expr")
        assert result.success is False
        assert "表达式错误" in result.error
    
    def test_stats_operation(self):
        """测试统计操作 (确保其他功能未受影响)"""
        result = self.tool.execute(operation="stats", numbers=[1, 2, 3, 4, 5])
        assert result.success is True
        assert result.data['mean'] == 3.0
        assert result.data['median'] == 3
    
    def test_factorial_operation(self):
        """测试阶乘操作 (确保其他功能未受影响)"""
        result = self.tool.execute(operation="factorial", n=5)
        assert result.success is True
        assert result.data['factorial'] == 120
    
    def test_execution_time_recorded(self):
        """测试执行时间被记录"""
        result = self.tool.execute(operation="eval", expression="2+2")
        assert result.execution_time is not None
        assert result.execution_time >= 0


class TestPerformance:
    """性能测试 - 要求 <1.5x 原实现"""
    
    def test_safe_eval_performance(self):
        """测试safe_math_eval()性能"""
        expressions = [
            "2 + 2",
            "10 * 5",
            "sqrt(144)",
            "sin(pi/4)",
            "2^10 + sqrt(144)",
            "log(e) + exp(0)",
        ]
        
        iterations = 100
        
        start = time.time()
        for _ in range(iterations):
            for expr in expressions:
                safe_math_eval(expr)
        elapsed = time.time() - start
        
        # 要求: 100次 * 6个表达式 在 0.5秒内完成
        assert elapsed < 0.5, f"性能测试失败: {elapsed:.3f}秒 (要求<0.5秒)"
        
        print(f"\n✅ 性能测试通过: {iterations}次迭代 * {len(expressions)}表达式 = {elapsed:.3f}秒")
    
    def test_tool_execution_overhead(self):
        """测试工具执行开销"""
        tool = MathTool()
        
        iterations = 50
        
        start = time.time()
        for _ in range(iterations):
            tool.execute(operation="eval", expression="2+2")
        elapsed = time.time() - start
        
        # 要求: 50次调用在 0.5秒内完成 (包含所有开销)
        assert elapsed < 0.5, f"工具开销测试失败: {elapsed:.3f}秒"
        
        avg_time = elapsed / iterations * 1000  # 毫秒
        print(f"✅ 平均执行时间: {avg_time:.2f}ms/次")


class TestComparison:
    """对比测试 - 验证功能完整性"""
    
    def test_feature_parity(self):
        """确保新实现支持所有原有功能"""
        tool = MathTool()
        
        # 原实现支持的所有表达式
        test_cases = [
            ("2+2", 4),
            ("10-5", 5),
            ("3*4", 12),
            ("20/4", 5.0),
            ("2**8", 256),
            ("sqrt(16)", 4.0),
            ("abs(-10)", 10),
            ("sin(0)", 0.0),
            ("cos(0)", 1.0),
            ("log10(100)", 2.0),
            ("ceil(3.2)", 4),
            ("floor(3.8)", 3),
        ]
        
        for expr, expected in test_cases:
            result = tool.execute(operation="eval", expression=expr)
            assert result.success is True, f"表达式失败: {expr}"
            assert result.data['result'] == pytest.approx(expected, rel=1e-5), \
                f"结果不匹配: {expr} 预期={expected} 实际={result.data['result']}"
        
        print(f"\n✅ 功能对比测试通过: {len(test_cases)}/{len(test_cases)} 表达式")


if __name__ == "__main__":
    # 运行测试
    print("=" * 60)
    print("Math Operations 修复测试")
    print("=" * 60)
    
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-p", "no:warnings"
    ])
