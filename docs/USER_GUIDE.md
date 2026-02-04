# Math Component 用户指南

**Version:** 1.0.0  
**Last Updated:** 2025年11月15日

欢迎使用Math Component！本指南将帮助您快速上手并充分利用组件的强大功能。

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Common Use Cases](#common-use-cases)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## Quick Start

### 5分钟快速开始

```python
# 1. 导入
from math_component.integration import create_agi_math_tool

# 2. 创建工具实例
tool = create_agi_math_tool()

# 3. 立即开始计算
# 求导
result = tool.differentiate("x**3 + 2*x**2", "x")
print(result['result'].expression)  # 3*x**2 + 4*x

# 积分
result = tool.integrate("x**2", "x")
print(result['result'].expression)  # x**3/3

# 解方程
result = tool.solve_equation("x**2 - 4", "x")
print(result['result'])  # [-2, 2]

# 统计分析
result = tool.statistical_analysis([1, 2, 3, 4, 5])
print(result['result']['result']['statistics']['mean'])  # 3.0
```

**就这么简单！** 🎉

---

## Installation

### 系统要求

- **Python:** 3.8+
- **操作系统:** Windows / Linux / macOS
- **内存:** 4GB+ 推荐
- **GPU:** 可选，用于加速张量计算

### 安装步骤

#### 方法1：使用pip（推荐）

```bash
# 核心依赖
pip install sympy numpy scipy matplotlib

# 可选：物理计算（GPU加速）
pip install torch torchvision

# 可选：高级可视化
pip install plotly seaborn
```

#### 方法2：从源码安装

```bash
git clone https://github.com/your-repo/math-component.git
cd math-component
pip install -r requirements.txt
```

### 验证安装

```python
# 运行验证脚本
python verify_installation.py
```

预期输出：
```
✓ Core system initialized
✓ All 5 engines operational
✓ 2 plugins loaded
✓ AGI integration ready
Installation successful!
```

---

## Basic Usage

### 1. 初始化组件

有三种方式初始化：

#### 方式A：快捷工具（推荐新手）

```python
from math_component.integration import create_agi_math_tool

tool = create_agi_math_tool()
```

**优点：** 简单、快捷方法、适合常见操作

#### 方式B：桥接器（推荐高级用户）

```python
from math_component.integration import create_agi_math_bridge

bridge = create_agi_math_bridge()
```

**优点：** 完整控制、性能追踪、自然语言查询

#### 方式C：直接引擎（专家模式）

```python
from math_component.core import MathCore
from math_component.engines import SymbolicEngine

core = MathCore()
engine = SymbolicEngine(core)
```

**优点：** 最大灵活性、底层控制

### 2. 符号计算

#### 求导

```python
# 一阶导数
result = tool.differentiate("sin(x)*cos(x)", "x")
print(result['result'].expression)
# -sin(x)**2 + cos(x)**2

# 高阶导数
result = tool.differentiate("x**4", "x", order=2)
print(result['result'].expression)
# 12*x**2

# 多变量
result = tool.differentiate("x**2 + y**2", "x")
# 2*x
```

#### 积分

```python
# 不定积分
result = tool.integrate("1/x", "x")
# log(x)

# 定积分
from math_component.engines import SymbolicEngine
engine = SymbolicEngine(MathCore())
result = engine.integrate("x**2", "x", bounds=(0, 1))
print(result.expression)
# 1/3

# 多重积分
result = engine.integrate("x*y", "x", bounds=(0, 1))
result = engine.integrate(str(result.expression), "y", bounds=(0, 1))
# 1/4
```

#### 解方程

```python
# 代数方程
result = tool.solve_equation("x**2 - 5*x + 6", "x")
# [2, 3]

# 三角方程
result = tool.solve_equation("sin(x) - 0.5", "x")
# [pi/6, 5*pi/6]

# 超越方程
result = tool.solve_equation("exp(x) - 2", "x")
# [log(2)]
```

### 3. 数值计算

#### 求解ODE

```python
from math_component.engines import NumericalEngine

engine = NumericalEngine(MathCore())

# 简单ODE: dy/dt = -k*y (指数衰减)
def decay(t, y):
    k = 0.5
    return -k * y

result = engine.solve_ode(
    func=decay,
    initial_conditions={"y0": [10.0], "t0": 0.0},
    t_span=(0.0, 10.0)
)

print(f"初始值: {result['y'][0]}")  # 10.0
print(f"最终值: {result['y'][-1]}")  # ~0.067
```

#### 优化问题

```python
# 最小化 (x-3)^2 + (y-4)^2
def objective(params):
    x, y = params
    return (x - 3)**2 + (y - 4)**2

result = engine.optimize(
    objective=objective,
    initial_guess=[0, 0],
    method="SLSQP"
)

print(result['x'])  # [3.0, 4.0]
print(result['fun'])  # 0.0
```

#### 数值积分

```python
import numpy as np

# 计算 ∫₀^π sin(x) dx
result = engine.numerical_integrate(
    func=np.sin,
    a=0,
    b=np.pi,
    method="quad"
)

print(result['value'])  # 2.0
print(result['error'])  # ~1e-14
```

### 4. 物理建模

#### 刚体动力学

```python
from math_component.engines import PhysicsMathEngine

engine = PhysicsMathEngine(MathCore())

# 模拟立方体受力
result = engine.rigid_body_dynamics(
    mass=5.0,  # 5kg
    inertia=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    forces=[10, 0, -50],  # 水平推力 + 重力
    torques=[0, 0, 5],  # 扭矩
    dt=0.01
)

print(f"线性加速度: {result['linear_acceleration']}")
print(f"角加速度: {result['angular_acceleration']}")
```

#### 碰撞模拟

```python
# 两球碰撞
ball1 = {
    "position": [0, 0, 0],
    "velocity": [2, 0, 0],
    "mass": 1.0
}

ball2 = {
    "position": [5, 0, 0],
    "velocity": [-1, 0, 0],
    "mass": 2.0
}

result = engine.collision_physics(
    ball1, ball2,
    restitution=0.9  # 弹性系数
)

print(f"碰撞后速度: {result['velocities']}")
print(f"冲量: {result['impulse']}")
```

### 5. 几何计算

#### 3D变换

```python
from math_component.engines import GeometryEngine
import numpy as np

engine = GeometryEngine(MathCore())

# 定义立方体顶点
cube = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
]

# 旋转90度
result = engine.transform_3d(
    points=cube,
    transformation={"type": "rotation", "axis": "z", "angle": np.pi/2}
)

print(result['transformed_points'])
```

#### 投影

```python
# 3D点投影到2D平面
points_3d = [[1, 2, 5], [3, 4, 8], [-1, 0, 3]]

result = engine.projection(
    points_3d=points_3d,
    projection_type="perspective",
    focal_length=100
)

print(result['projected_points_2d'])
```

#### 碰撞检测

```python
# 球与立方体碰撞
sphere = {
    "type": "sphere",
    "center": [0, 0, 0],
    "radius": 2.0
}

box = {
    "type": "box",
    "center": [3, 0, 0],
    "dimensions": [2, 2, 2]
}

result = engine.collision_detection(sphere, box)
print(f"碰撞: {result['collision']}")  # True/False
print(f"距离: {result['distance']}")
```

### 6. 统计分析

```python
# 使用插件
data = [23, 45, 67, 89, 12, 34, 56, 78, 90, 11]

result = tool.statistical_analysis(data)
stats = result['result']['result']['statistics']

print(f"均值: {stats['mean']}")
print(f"中位数: {stats['median']}")
print(f"标准差: {stats['std']}")
print(f"四分位数: Q1={stats['q25']}, Q3={stats['q75']}")
```

---

## Common Use Cases

### Use Case 1: 微积分课程辅助

```python
# 学生学习工具
from math_component.integration import create_agi_math_tool

tool = create_agi_math_tool()

# 验证手工计算
problem = "x**3 - 3*x**2 + 2*x"

# 求导
derivative = tool.differentiate(problem, "x")
print(f"导数: {derivative['result'].expression}")

# 求极值点
critical_points = tool.solve_equation(str(derivative['result'].expression), "x")
print(f"极值点: {critical_points['result']}")

# 二阶导数判定
second_derivative = tool.differentiate(str(derivative['result'].expression), "x")
print(f"二阶导数: {second_derivative['result'].expression}")
```

### Use Case 2: 工程优化设计

```python
# 最小化材料成本
from math_component.engines import NumericalEngine

engine = NumericalEngine(MathCore())

def cost_function(dimensions):
    """计算圆柱体材料成本（固定体积）"""
    r, h = dimensions
    volume = 3.14159 * r**2 * h
    surface_area = 2 * 3.14159 * r * (r + h)
    
    # 约束：体积必须为1000
    if abs(volume - 1000) > 0.1:
        return 1e10  # 惩罚
    
    return surface_area  # 最小化表面积

result = engine.optimize(
    objective=cost_function,
    initial_guess=[5.0, 12.7],
    method="SLSQP"
)

print(f"最优尺寸 - 半径: {result['x'][0]:.2f}, 高度: {result['x'][1]:.2f}")
print(f"最小表面积: {result['fun']:.2f}")
```

### Use Case 3: 游戏物理引擎

```python
# 弹跳球模拟
from math_component.engines import PhysicsMathEngine

engine = PhysicsMathEngine(MathCore())

class Ball:
    def __init__(self, pos, vel, mass):
        self.pos = pos
        self.vel = vel
        self.mass = mass

# 初始状态
ball = Ball(
    pos=[0, 10, 0],  # 10米高
    vel=[5, 0, 0],   # 水平速度5m/s
    mass=1.0
)

# 模拟1秒（重力作用）
dt = 0.01
for step in range(100):
    result = engine.rigid_body_dynamics(
        mass=ball.mass,
        inertia=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        forces=[0, 0, -9.8 * ball.mass],  # 重力
        torques=[0, 0, 0],
        dt=dt
    )
    
    # 更新位置和速度
    acc = result['linear_acceleration']
    ball.vel = [ball.vel[i] + acc[i] * dt for i in range(3)]
    ball.pos = [ball.pos[i] + ball.vel[i] * dt for i in range(3)]
    
    # 地面碰撞
    if ball.pos[2] <= 0:
        ball.vel[2] = -ball.vel[2] * 0.8  # 弹性碰撞
        ball.pos[2] = 0

print(f"最终位置: {ball.pos}")
print(f"最终速度: {ball.vel}")
```

### Use Case 4: 数据分析管道

```python
# 完整分析流程
import numpy as np
from math_component.integration import create_agi_math_bridge

bridge = create_agi_math_bridge()

# 1. 生成模拟数据
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100).tolist()

# 2. 描述性统计
stats_result = bridge.execute_math_operation(
    operation="descriptive_stats",
    category="statistics",
    data=data
)
stats = stats_result['result']['result']['statistics']
print(f"数据摘要: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")

# 3. 拟合分布
# (假设添加了distribution_fitting操作)

# 4. 异常值检测
q1, q3 = stats['q25'], stats['q75']
iqr = q3 - q1
outliers = [x for x in data if x < q1 - 1.5*iqr or x > q3 + 1.5*iqr]
print(f"检测到 {len(outliers)} 个异常值")

# 5. 性能见解
insights = bridge.get_learning_insights("statistics.descriptive_stats")
print(f"平均执行时间: {insights.get('average_time', 0)*1000:.2f}ms")
```

### Use Case 5: 机器人路径规划

```python
# 机器人避障路径
from math_component.engines import GeometryEngine

engine = GeometryEngine(MathCore())

# 定义障碍物
obstacles = [
    {"type": "sphere", "center": [5, 5, 0], "radius": 2},
    {"type": "box", "center": [10, 8, 0], "dimensions": [3, 3, 3]}
]

# 测试路径点
path_points = [[0, 0, 0], [3, 3, 0], [7, 7, 0], [12, 12, 0]]

# 碰撞检测
safe_path = []
robot = {"type": "sphere", "center": [0, 0, 0], "radius": 0.5}

for point in path_points:
    robot["center"] = point
    collision_free = True
    
    for obstacle in obstacles:
        result = engine.collision_detection(robot, obstacle)
        if result['collision']:
            collision_free = False
            print(f"点 {point} 与障碍物碰撞！")
            break
    
    if collision_free:
        safe_path.append(point)

print(f"安全路径: {safe_path}")
```

---

## Advanced Features

### 1. 自然语言查询

```python
from math_component.integration import create_agi_math_bridge

bridge = create_agi_math_bridge()

# 用自然语言描述需求
queries = [
    "solve quadratic equation",
    "calculate derivative",
    "matrix eigenvalue",
    "statistical analysis"
]

for query in queries:
    result = bridge.query_natural_language(query)
    print(f"\n查询: '{query}'")
    for match in result['matches']:
        print(f"  匹配类别: {match['category']} (置信度: {match['confidence']})")
        print(f"  可用操作: {', '.join(match['operations'][:3])}...")
```

### 2. 性能追踪与优化

```python
# 自动性能记录
bridge = create_agi_math_bridge()

# 执行多次操作
for i in range(10):
    bridge.execute_math_operation(
        "differentiate",
        "symbolic",
        expression=f"x**{i+2}",
        variable="x"
    )

# 查看性能分析
insights = bridge.get_learning_insights("symbolic.differentiate")
print(f"总调用次数: {insights['record_count']}")
print(f"平均时间: {insights['average_time']*1000:.2f}ms")
print(f"最快: {insights['best_time']*1000:.2f}ms")
print(f"最慢: {insights['worst_time']*1000:.2f}ms")
print(f"成功率: {insights['success_rate']*100:.1f}%")

# 获取优化建议
optimization = bridge.optimize_for_problem_type("symbolic.differentiate")
print(f"优化级别: {optimization['optimization_applied']['optimization_level']}")
if optimization['optimization_applied']['recommendations']:
    print("建议:")
    for rec in optimization['optimization_applied']['recommendations']:
        print(f"  - {rec}")
```

### 3. 自定义插件

```python
# 创建自定义插件
from math_component.plugins import PluginBase

class StatisticsAdvancedPlugin(PluginBase):
    def __init__(self, math_core):
        super().__init__(
            name="statistics_advanced",
            version="1.0.0",
            description="Advanced statistical methods",
            math_core=math_core
        )
        
        # 注册能力
        self.register_capability("time_series_analysis", self.time_series)
        self.register_capability("correlation_matrix", self.correlation)
    
    def time_series(self, data, **kwargs):
        """时间序列分析"""
        import numpy as np
        
        # 简单移动平均
        window = kwargs.get('window', 5)
        ma = np.convolve(data, np.ones(window)/window, mode='valid')
        
        return {
            "moving_average": ma.tolist(),
            "window_size": window
        }
    
    def correlation(self, data_matrix, **kwargs):
        """相关系数矩阵"""
        import numpy as np
        
        corr = np.corrcoef(data_matrix)
        return {
            "correlation_matrix": corr.tolist()
        }

# 使用自定义插件
from math_component.plugins import PluginManager

manager = PluginManager(MathCore())
# 加载和激活你的插件...
```

### 4. 批量操作

```python
# 批量求导
expressions = [
    "x**2", "sin(x)", "exp(x)", "log(x)", "x**3 + 2*x"
]

results = []
for expr in expressions:
    result = tool.differentiate(expr, "x")
    results.append({
        "original": expr,
        "derivative": str(result['result'].expression)
    })

for r in results:
    print(f"d/dx({r['original']}) = {r['derivative']}")
```

### 5. 可视化集成

```python
# 绘制函数及其导数
import matplotlib.pyplot as plt
import numpy as np
from sympy import lambdify, symbols

# 符号计算
x = symbols('x')
from math_component.engines import SymbolicEngine

engine = SymbolicEngine(MathCore())

original = engine.simplify("x**3 - 3*x**2 + 2*x")
derivative = engine.differentiate(str(original.expression), "x")

# 转换为可绘图函数
x_vals = np.linspace(-1, 4, 100)
f = lambdify(x, original.expression, 'numpy')
f_prime = lambdify(x, derivative.expression, 'numpy')

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label='f(x)', linewidth=2)
plt.plot(x_vals, f_prime(x_vals), label="f'(x)", linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and its Derivative')
plt.savefig('function_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Best Practices

### 1. 性能优化

✅ **DO:**
```python
# 重用实例
tool = create_agi_math_tool()
for i in range(100):
    result = tool.differentiate(f"x**{i}", "x")
```

❌ **DON'T:**
```python
# 每次创建新实例（慢）
for i in range(100):
    tool = create_agi_math_tool()  # 浪费资源
    result = tool.differentiate(f"x**{i}", "x")
```

### 2. 错误处理

✅ **DO:**
```python
try:
    result = tool.solve_equation("complex_equation", "x")
    if result['success']:
        print(result['result'])
    else:
        print(f"错误: {result['error']}")
except Exception as e:
    print(f"异常: {e}")
```

❌ **DON'T:**
```python
# 不检查返回状态
result = tool.solve_equation("complex_equation", "x")
print(result['result'])  # 可能失败
```

### 3. 表达式处理

✅ **DO:**
```python
# 使用字符串表达式
expr = "x**2 + 2*x + 1"
result = tool.differentiate(expr, "x")
```

✅ **ALSO GOOD:**
```python
# 或使用SymPy对象
import sympy as sp
x = sp.Symbol('x')
expr = x**2 + 2*x + 1
result = engine.differentiate(str(expr), "x")
```

### 4. 数值稳定性

✅ **DO:**
```python
# 选择合适的方法
result = engine.solve_ode(
    func=stiff_equation,
    initial_conditions={"y0": [1.0], "t0": 0.0},
    t_span=(0, 10),
    method="BDF"  # 适合刚性方程
)
```

### 5. 内存管理

✅ **DO:**
```python
# 大规模计算后清理
import gc

for batch in large_dataset:
    results = process_batch(batch)
    # 使用结果...
    del results
    gc.collect()
```

---

## Troubleshooting

### 问题 1: 导入错误

**症状：**
```
ModuleNotFoundError: No module named 'math_component'
```

**解决方案：**
```bash
# 确保在正确的目录
cd /path/to/AGI

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/AGI"  # Linux/Mac
$env:PYTHONPATH = "D:\TRAE_PROJECT\AGI"  # Windows PowerShell
```

### 问题 2: 符号计算失败

**症状：**
```
SymbolicError: Unable to parse expression
```

**解决方案：**
```python
# 检查表达式语法
expressions = [
    "x**2",      # ✓ 正确
    "x^2",       # ✗ 错误（使用 ** 而非 ^）
    "sin(x)",    # ✓ 正确
    "Sin(x)",    # ✗ 错误（小写）
]

# 使用SymPy验证
import sympy as sp
try:
    expr = sp.sympify("your_expression")
    print("表达式有效")
except:
    print("表达式无效")
```

### 问题 3: 数值不收敛

**症状：**
```
NumericalError: Convergence failure
```

**解决方案：**
```python
# 1. 调整初始猜测
result = engine.optimize(
    objective=objective,
    initial_guess=[1.0, 1.0],  # 尝试不同的起点
    method="SLSQP"
)

# 2. 更改求解方法
result = engine.solve_ode(
    func=func,
    initial_conditions=initial,
    t_span=t_span,
    method="BDF"  # 尝试不同方法
)

# 3. 增加容差
core = MathCore(config={'tolerance': 1e-8})  # 放宽容差
```

### 问题 4: 插件未找到

**症状：**
```
PluginError: Plugin 'statistics' not found
```

**解决方案：**
```python
from math_component.plugins import PluginManager

manager = PluginManager(MathCore())

# 检查可用插件
results = manager.load_all_plugins()
print("加载结果:", results)

# 激活所有插件
manager.activate_all_plugins()

# 验证
for name, plugin in manager.plugins.items():
    print(f"{name}: {plugin.status.value}")
```

### 问题 5: 性能慢

**解决方案：**

1. **启用缓存**
```python
core = MathCore(config={'cache_enabled': True})
```

2. **简化表达式**
```python
# 先简化再计算
expr = engine.simplify("complex_expression")
result = engine.differentiate(str(expr.expression), "x")
```

3. **使用数值方法代替符号**
```python
# 符号积分可能很慢
# result = engine.integrate("very_complex", "x")

# 改用数值积分
result = numerical_engine.numerical_integrate(
    func=lambda x: eval_complex_expr(x),
    a=0, b=1
)
```

---

## FAQ

### Q1: Math Component支持哪些数学运算？

**A:** 支持5大类：
- **符号计算**：微积分、方程求解、代数运算
- **数值分析**：ODE/PDE求解、优化、线性代数
- **物理建模**：刚体动力学、碰撞、张量运算
- **几何计算**：3D变换、投影、碰撞检测
- **统计分析**：描述统计、假设检验、回归

### Q2: 如何提高计算精度？

**A:**
```python
# 方法1：配置高精度
core = MathCore(config={'precision': 'high'})

# 方法2：使用高精度计算
result = engine.high_precision_compute("pi", precision=100)

# 方法3：符号计算（精确）
result = symbolic_engine.integrate("1/x", "x")  # log(x)，精确
```

### Q3: 能否并行执行？

**A:** 目前版本不支持自动并行，但可以手动实现：

```python
from concurrent.futures import ThreadPoolExecutor

def compute(expr):
    return tool.differentiate(expr, "x")

expressions = ["x**2", "sin(x)", "exp(x)"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(compute, expressions))
```

### Q4: 如何保存和加载结果？

**A:**
```python
import json
import pickle

# 方法1：JSON（简单类型）
result = tool.differentiate("x**2", "x")
with open('result.json', 'w') as f:
    json.dump(result, f, default=str)

# 方法2：Pickle（复杂对象）
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)

# 加载
with open('result.pkl', 'rb') as f:
    loaded_result = pickle.load(f)
```

### Q5: 支持GPU加速吗？

**A:** 部分支持，主要在物理引擎的张量运算：

```python
# 确保已安装PyTorch
# pip install torch

from math_component.engines import PhysicsMathEngine

engine = PhysicsMathEngine(MathCore())

# 张量运算会自动使用GPU（如果可用）
result = engine.tensor_operations(
    operation="dot",
    tensor_a=[1, 2, 3],
    tensor_b=[4, 5, 6]
)
```

### Q6: 如何贡献代码或报告bug？

**A:**
- **GitHub**: https://github.com/your-repo/issues
- **文档**: https://github.com/your-repo/docs
- **Email**: support@example.com

### Q7: 许可证？

**A:** MIT License - 自由使用、修改和分发

---

## Next Steps

🎓 **学习更多：**
- [API Reference](./API_REFERENCE.md) - 完整API文档
- [Examples](./examples/) - 更多示例代码
- [Architecture Guide](./ARCHITECTURE.md) - 系统架构

🛠️ **开发：**
- [Plugin Development](./PLUGIN_DEVELOPMENT.md) - 创建自定义插件
- [Contributing Guide](./CONTRIBUTING.md) - 贡献指南

📊 **性能：**
- [Benchmarks](./BENCHMARKS.md) - 性能基准测试
- [Optimization Tips](./OPTIMIZATION.md) - 优化技巧

---

## Support

需要帮助？我们随时准备协助！

- 📚 **文档**: [完整文档](https://github.com/your-repo/docs)
- 💬 **社区**: [讨论区](https://github.com/your-repo/discussions)
- 🐛 **Bug报告**: [Issues](https://github.com/your-repo/issues)
- ✉️ **Email**: support@example.com

---

**Happy Computing!** 🚀

*Math Component Team*
