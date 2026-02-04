"""
高级数学工具测试套件
测试 AdvancedMathematicsTool 的所有功能
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enhanced_tools_collection import AdvancedMathematicsTool, ToolResult


class TestAdvancedMathematicsTool(unittest.TestCase):
    """高级数学工具测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.tool = AdvancedMathematicsTool()
    
    def test_tool_metadata(self):
        """测试工具元数据"""
        self.assertEqual(self.tool.name, "advanced_mathematics")
        self.assertIn("求导", self.tool.description)
        self.assertEqual(self.tool.category, "计算")
    
    # ==================== 求导测试 ====================
    
    def test_differentiate_basic(self):
        """测试基本求导: d/dx(x^2) = 2x"""
        result = self.tool.execute(
            operation="differentiate",
            expression="x**2",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "2*x")
    
    def test_differentiate_polynomial(self):
        """测试多项式求导: d/dx(x^3 + 3x^2 + 2x + 1)"""
        result = self.tool.execute(
            operation="differentiate",
            expression="x**3 + 3*x**2 + 2*x + 1",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "3*x**2 + 6*x + 2")
    
    def test_differentiate_trigonometric(self):
        """测试三角函数求导: d/dx(sin(x)) = cos(x)"""
        result = self.tool.execute(
            operation="differentiate",
            expression="sin(x)",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "cos(x)")
    
    def test_differentiate_higher_order(self):
        """测试高阶导数: d^2/dx^2(x^3)"""
        result = self.tool.execute(
            operation="differentiate",
            expression="x**3",
            variable="x",
            order=2
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "6*x")
    
    def test_differentiate_alias_derivative(self):
        """测试求导别名: derivative"""
        result = self.tool.execute(
            operation="derivative",
            expression="x**2",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "2*x")
    
    def test_differentiate_alias_diff(self):
        """测试求导别名: diff"""
        result = self.tool.execute(
            operation="diff",
            expression="x**2",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "2*x")
    
    # ==================== 积分测试 ====================
    
    def test_integrate_basic(self):
        """测试基本积分: ∫x dx = x^2/2"""
        result = self.tool.execute(
            operation="integrate",
            expression="x",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "x**2/2")
    
    def test_integrate_polynomial(self):
        """测试多项式积分: ∫(2x + 1) dx"""
        result = self.tool.execute(
            operation="integrate",
            expression="2*x + 1",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "x**2 + x")
    
    def test_integrate_trigonometric(self):
        """测试三角函数积分: ∫cos(x) dx = sin(x)"""
        result = self.tool.execute(
            operation="integrate",
            expression="cos(x)",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "sin(x)")
    
    def test_integrate_definite(self):
        """测试定积分: ∫[0,1] x dx = 1/2"""
        result = self.tool.execute(
            operation="integrate",
            expression="x",
            variable="x",
            definite=True,
            lower=0,
            upper=1
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "1/2")
    
    # ==================== 解方程测试 ====================
    
    def test_solve_linear(self):
        """测试线性方程: x - 3 = 0"""
        result = self.tool.execute(
            operation="solve",
            equation="x - 3",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['solutions'], ["3"])
    
    def test_solve_quadratic(self):
        """测试二次方程: x^2 - 4 = 0"""
        result = self.tool.execute(
            operation="solve",
            equation="x**2 - 4",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.data['solutions']), 2)
        self.assertIn("-2", result.data['solutions'])
        self.assertIn("2", result.data['solutions'])
    
    def test_solve_no_solution(self):
        """测试无解方程: x^2 + 1 = 0 (在实数域)"""
        result = self.tool.execute(
            operation="solve",
            equation="x**2 + 1",
            variable="x"
        )
        self.assertTrue(result.success)
        # SymPy 会返回复数解
        self.assertEqual(len(result.data['solutions']), 2)
    
    # ==================== 符号操作测试 ====================
    
    def test_simplify_basic(self):
        """测试简化: (x^2 - 1)/(x - 1) = x + 1"""
        result = self.tool.execute(
            operation="simplify",
            expression="(x**2 - 1)/(x - 1)"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "x + 1")
    
    def test_expand_basic(self):
        """测试展开: (x + 1)^2 = x^2 + 2x + 1"""
        result = self.tool.execute(
            operation="expand",
            expression="(x + 1)**2"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "x**2 + 2*x + 1")
    
    def test_factor_basic(self):
        """测试因式分解: x^2 - 1 = (x - 1)(x + 1)"""
        result = self.tool.execute(
            operation="factor",
            expression="x**2 - 1"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "(x - 1)*(x + 1)")
    
    # ==================== 极限测试 ====================
    
    def test_limit_basic(self):
        """测试极限: lim(x->0) sin(x)/x = 1"""
        result = self.tool.execute(
            operation="limit",
            expression="sin(x)/x",
            variable="x",
            point=0
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "1")
    
    def test_limit_infinity(self):
        """测试无穷极限: lim(x->∞) 1/x = 0"""
        result = self.tool.execute(
            operation="limit",
            expression="1/x",
            variable="x",
            point="oo"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "0")
    
    # ==================== 级数展开测试 ====================
    
    def test_series_basic(self):
        """测试级数展开: e^x 在 x=0 处的泰勒展开"""
        result = self.tool.execute(
            operation="series",
            expression="exp(x)",
            variable="x",
            point=0,
            order=4
        )
        self.assertTrue(result.success)
        self.assertIn("1 + x", result.data['result'])
    
    # ==================== 矩阵运算测试 ====================
    
    def test_matrix_determinant(self):
        """测试矩阵行列式"""
        result = self.tool.execute(
            operation="matrix",
            matrix_operation="determinant",
            matrix=[[1, 2], [3, 4]]
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data['result'], "-2")
    
    def test_matrix_transpose(self):
        """测试矩阵转置"""
        result = self.tool.execute(
            operation="matrix",
            matrix_operation="transpose",
            matrix=[[1, 2, 3], [4, 5, 6]]
        )
        self.assertTrue(result.success)
        self.assertIn("Matrix", result.data['result'])
    
    def test_matrix_inverse(self):
        """测试矩阵求逆"""
        result = self.tool.execute(
            operation="matrix",
            matrix_operation="inverse",
            matrix=[[1, 2], [3, 4]]
        )
        self.assertTrue(result.success)
        self.assertIn("Matrix", result.data['result'])
    
    # ==================== 错误处理测试 ====================
    
    def test_missing_operation(self):
        """测试缺失 operation 参数"""
        result = self.tool.execute(expression="x**2")
        self.assertFalse(result.success)
        self.assertIn("缺少必需参数: operation", result.error)
    
    def test_missing_expression(self):
        """测试缺失 expression 参数"""
        result = self.tool.execute(operation="differentiate")
        self.assertFalse(result.success)
        self.assertIn("缺少必需参数: expression", result.error)
    
    def test_unsupported_operation(self):
        """测试不支持的操作"""
        result = self.tool.execute(
            operation="unsupported_op",
            expression="x**2"
        )
        self.assertFalse(result.success)
        self.assertIn("不支持的操作", result.error)
    
    def test_invalid_expression(self):
        """测试无效表达式"""
        result = self.tool.execute(
            operation="differentiate",
            expression="invalid syntax @#$",
            variable="x"
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
    
    def test_matrix_missing_operation(self):
        """测试矩阵运算缺失 matrix_operation"""
        result = self.tool.execute(
            operation="matrix",
            matrix=[[1, 2], [3, 4]]
        )
        self.assertFalse(result.success)
        self.assertIn("缺少参数: matrix_operation", result.error)
    
    def test_matrix_missing_data(self):
        """测试矩阵运算缺失矩阵数据"""
        result = self.tool.execute(
            operation="matrix",
            matrix_operation="determinant"
        )
        self.assertFalse(result.success)
        self.assertIn("缺少参数: matrix", result.error)
    
    def test_matrix_unsupported_operation(self):
        """测试矩阵不支持的操作"""
        result = self.tool.execute(
            operation="matrix",
            matrix_operation="unsupported",
            matrix=[[1, 2], [3, 4]]
        )
        self.assertFalse(result.success)
        self.assertIn("不支持的矩阵操作", result.error)
    
    # ==================== 性能测试 ====================
    
    def test_execution_time_recorded(self):
        """测试执行时间记录"""
        result = self.tool.execute(
            operation="differentiate",
            expression="x**2",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertIsNotNone(result.execution_time)
        self.assertGreater(result.execution_time, 0)
    
    # ==================== LaTeX 输出测试 ====================
    
    def test_latex_output(self):
        """测试 LaTeX 格式输出"""
        result = self.tool.execute(
            operation="differentiate",
            expression="x**2",
            variable="x"
        )
        self.assertTrue(result.success)
        self.assertIn("latex", result.data)
        self.assertIsInstance(result.data['latex'], str)


def run_tests():
    """运行测试套件"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdvancedMathematicsTool)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
