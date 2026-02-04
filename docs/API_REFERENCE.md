# Math Component API Reference

**Version:** 1.0.0  
**Date:** 2025年11月15日  
**Project:** Advanced Mathematical Computing Component for AGI Systems

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Engines API](#engines-api)
   - [Symbolic Engine](#symbolic-engine)
   - [Numerical Engine](#numerical-engine)
   - [Physics Math Engine](#physics-math-engine)
   - [Geometry Engine](#geometry-engine)
   - [Math Learning Engine](#math-learning-engine)
4. [Plugin System](#plugin-system)
5. [AGI Integration](#agi-integration)
6. [Data Structures](#data-structures)
7. [Error Handling](#error-handling)

---

## Overview

Math Component是一个为AGI系统设计的高性能数学计算组件，提供符号计算、数值分析、物理建模、几何处理和自适应学习能力。

### Key Features

- **5个专业引擎**：符号、数值、物理、几何、学习
- **插件扩展系统**：支持自定义数学能力
- **AGI集成接口**：自然语言查询、能力映射
- **性能优化**：自适应学习、缓存机制
- **类型安全**：完整的类型注解

### Installation

```bash
pip install sympy numpy scipy matplotlib torch
```

---

## Core Architecture

### MathCore

中心配置和资源管理器。

```python
from math_component.core import MathCore

# 初始化
core = MathCore(config={
    'precision': 'high',
    'cache_enabled': True,
    'parallel_enabled': False
})

# 访问配置
precision = core.config.get('precision')

# 访问引擎
symbolic = core.symbolic_engine
numerical = core.numerical_engine
```

#### Configuration Options

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `precision` | str | 'high' | 计算精度：'low', 'medium', 'high' |
| `cache_enabled` | bool | True | 启用结果缓存 |
| `parallel_enabled` | bool | False | 启用并行计算 |
| `max_iterations` | int | 1000 | 最大迭代次数 |
| `tolerance` | float | 1e-10 | 数值容差 |

---

## SymbolicEngine

## Engines API

### Symbolic Engine

符号数学计算引擎，基于SymPy。

#### 初始化

```python
from math_component.engines import SymbolicEngine

engine = SymbolicEngine(math_core)
```

#### Methods

##### `differentiate(expression: str, variable: str = "x", order: int = 1) -> MathExpression`

计算函数的导数。

**参数：**
- `expression` (str): 数学表达式，如 "x**2 + 3*x"
- `variable` (str): 求导变量，默认 "x"
- `order` (int): 导数阶数，默认 1

**返回：** `MathExpression` 对象

**示例：**
```python
# 一阶导数
result = engine.differentiate("x**3 + 2*x**2 + x", "x")
print(result.expression)  # 3*x**2 + 4*x + 1

# 二阶导数
result = engine.differentiate("sin(x)", "x", order=2)
print(result.expression)  # -sin(x)
```

##### `integrate(expression: str, variable: str = "x", bounds: Optional[Tuple] = None) -> MathExpression`

计算不定积分或定积分。

**参数：**
- `expression` (str): 被积函数
- `variable` (str): 积分变量
- `bounds` (Tuple[float, float], optional): 积分区间 (a, b)

**返回：** `MathExpression` 对象

**示例：**
```python
# 不定积分
result = engine.integrate("x**2", "x")
print(result.expression)  # x**3/3

# 定积分
result = engine.integrate("x**2", "x", bounds=(0, 1))
print(result.expression)  # 1/3
```

##### `solve(equation: str, variable: str = "x") -> List[Any]`

求解代数方程。

**参数：**
- `equation` (str): 方程表达式，如 "x**2 - 4"
- `variable` (str): 求解变量

**返回：** 解的列表

**示例：**
```python
# 二次方程
solutions = engine.solve("x**2 - 4", "x")
print(solutions)  # [-2, 2]

# 三角方程
solutions = engine.solve("sin(x)", "x")
print(solutions)  # [0, pi]
```

##### `simplify(expression: str) -> MathExpression`

化简数学表达式。

**示例：**
```python
result = engine.simplify("(x + 1)**2 - (x**2 + 2*x + 1)")
print(result.expression)  # 0
```

##### `expand(expression: str) -> MathExpression`

展开表达式。

**示例：**
```python
result = engine.expand("(x + 1)*(x + 2)")
print(result.expression)  # x**2 + 3*x + 2
```

##### `factor(expression: str) -> MathExpression`

因式分解。

**示例：**
```python
result = engine.factor("x**2 - 4")
print(result.expression)  # (x - 2)*(x + 2)
```

##### `limit(expression: str, variable: str, point: str, direction: str = "+-") -> Any`

计算极限。

**参数：**
- `direction`: "+", "-", 或 "+-" (双侧极限)

**示例：**
```python
result = engine.limit("sin(x)/x", "x", "0")
print(result)  # 1
```

##### `series(expression: str, variable: str, point: str = "0", order: int = 6) -> MathExpression`

泰勒级数展开。

**示例：**
```python
result = engine.series("sin(x)", "x", "0", order=5)
print(result.expression)  # x - x**3/6 + O(x**5)
```

---

## NumericalEngine

高精度数值计算引擎，基于NumPy/SciPy。

#### 初始化

```python
from math_component.engines import NumericalEngine

engine = NumericalEngine(math_core)
```

#### Methods

##### `solve_ode(func: Callable, initial_conditions: Dict, t_span: Tuple, method: str = "RK45") -> Dict`

求解常微分方程。

**参数：**
- `func` (Callable): 微分方程函数 dy/dt = f(t, y)
- `initial_conditions`: {"y0": [...], "t0": float}
- `t_span` (Tuple): 时间区间 (t_start, t_end)
- `method` (str): 求解方法 "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"

**返回：** 包含 t, y, success 的字典

**示例：**
```python
# 求解 dy/dt = -2*y
def ode_func(t, y):
    return -2 * y

result = engine.solve_ode(
    func=ode_func,
    initial_conditions={"y0": [1.0], "t0": 0.0},
    t_span=(0.0, 5.0),
    method="RK45"
)
print(result['y'][-1])  # 最终值
```

##### `solve_linear_system(A: List[List[float]], b: List[float]) -> Dict`

求解线性方程组 Ax = b。

**返回：** `{"x": solution, "residual": float, "method": str, "condition_number": float}`

**示例：**
```python
A = [[2, 1], [1, 3]]
b = [5, 5]
result = engine.solve_linear_system(A, b)
print(result['x'])  # [1.6, 1.8]
```

##### `optimize(objective: Callable, initial_guess: List[float], constraints: Optional[List] = None, method: str = "SLSQP") -> Dict`

数值优化问题。

**参数：**
- `objective` (Callable): 目标函数
- `initial_guess` (List): 初始点
- `constraints` (List, optional): 约束条件
- `method` (str): "SLSQP", "trust-constr", "COBYLA", 等

**返回：** `{"x": optimum, "fun": value, "success": bool, "message": str}`

**示例：**
```python
# 最小化 x^2 + y^2
def objective(x):
    return x[0]**2 + x[1]**2

result = engine.optimize(
    objective=objective,
    initial_guess=[1.0, 1.0],
    method="SLSQP"
)
print(result['x'])  # [0, 0]
```

##### `find_root(func: Callable, initial_guess: float, method: str = "hybr") -> Dict`

求函数的根。

**方法选项：** "hybr", "lm", "broyden1", "broyden2", "anderson", "krylov"

**示例：**
```python
def func(x):
    return x**2 - 4

result = engine.find_root(func, initial_guess=1.0)
print(result['root'])  # 2.0
```

##### `numerical_integrate(func: Callable, a: float, b: float, method: str = "quad") -> Dict`

数值积分。

**参数：**
- `method`: "quad" (自适应), "trapz" (梯形), "simps" (Simpson)

**返回：** `{"value": float, "error": float, "method": str}`

**示例：**
```python
import numpy as np

result = engine.numerical_integrate(
    func=lambda x: x**2,
    a=0,
    b=1,
    method="quad"
)
print(result['value'])  # 0.333...
```

##### `high_precision_compute(expression: str, precision: int = 50) -> str`

高精度计算（使用SymPy）。

**示例：**
```python
result = engine.high_precision_compute("pi", precision=100)
print(result)  # 3.141592653589793238462643383279502884197...
```

##### `matrix_operations(operation: str, **kwargs) -> Dict`

矩阵运算。

**操作类型：**
- `"eigenvalue"`: 特征值分解
- `"svd"`: 奇异值分解
- `"inverse"`: 矩阵求逆
- `"determinant"`: 行列式
- `"rank"`: 秩

**示例：**
```python
result = engine.matrix_operations(
    operation="eigenvalue",
    matrix=[[1, 2], [2, 1]]
)
print(result['eigenvalues'])  # [-1, 3]
```

---

## PhysicsMathEngine

物理建模和张量计算引擎。

#### 初始化

```python
from math_component.engines import PhysicsMathEngine

engine = PhysicsMathEngine(math_core)
```

#### Methods

##### `rigid_body_dynamics(mass: float, inertia: List[List[float]], forces: List[float], torques: List[float], dt: float) -> Dict`

刚体动力学模拟。

**参数：**
- `mass` (float): 质量 (kg)
- `inertia` (List[List]): 惯性张量 3×3
- `forces` (List): 力向量 [Fx, Fy, Fz]
- `torques` (List): 力矩向量 [Tx, Ty, Tz]
- `dt` (float): 时间步长

**返回：** 包含加速度、角加速度的字典

**示例：**
```python
result = engine.rigid_body_dynamics(
    mass=10.0,
    inertia=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    forces=[10, 0, 0],
    torques=[0, 5, 0],
    dt=0.01
)
print(result['linear_acceleration'])  # [1.0, 0, 0]
print(result['angular_acceleration'])  # [0, 5.0, 0]
```

##### `collision_physics(object1: Dict, object2: Dict, restitution: float = 0.8) -> Dict`

碰撞检测与响应。

**参数：**
- `object1/2`: `{"position": [x,y,z], "velocity": [vx,vy,vz], "mass": float}`
- `restitution` (float): 恢复系数 (0=完全非弹性, 1=完全弹性)

**返回：** 碰撞后的速度和冲量

**示例：**
```python
obj1 = {"position": [0, 0, 0], "velocity": [1, 0, 0], "mass": 1.0}
obj2 = {"position": [1, 0, 0], "velocity": [-1, 0, 0], "mass": 1.0}

result = engine.collision_physics(obj1, obj2, restitution=1.0)
print(result['velocities'])  # [[-1, 0, 0], [1, 0, 0]] (完全弹性碰撞)
```

##### `tensor_operations(operation: str, **kwargs) -> Dict`

张量运算（支持PyTorch）。

**操作类型：**
- `"dot"`: 点积
- `"cross"`: 叉积
- `"contraction"`: 张量收缩
- `"transformation"`: 坐标变换

**示例：**
```python
result = engine.tensor_operations(
    operation="dot",
    tensor_a=[1, 2, 3],
    tensor_b=[4, 5, 6]
)
print(result['result'])  # 32
```

##### `conservation_laws(system_state: Dict, law_type: str) -> Dict`

验证守恒定律。

**定律类型：** "energy", "momentum", "angular_momentum"

**示例：**
```python
state = {
    "kinetic_energy": 100,
    "potential_energy": 50,
    "momentum": [10, 0, 0]
}

result = engine.conservation_laws(state, law_type="energy")
print(result['conserved'])  # True/False
print(result['total_energy'])  # 150
```

##### `field_theory(field_type: str, **kwargs) -> Dict`

场论计算。

**场类型：** "electromagnetic", "gravitational", "scalar", "vector"

**示例：**
```python
# 电磁场
result = engine.field_theory(
    field_type="electromagnetic",
    charge=1.0,
    position=[0, 0, 0],
    observation_point=[1, 0, 0]
)
print(result['electric_field'])  # [E_x, E_y, E_z]
```

---

## GeometryEngine

几何变换和空间计算引擎。

#### 初始化

```python
from math_component.engines import GeometryEngine

engine = GeometryEngine(math_core)
```

#### Methods

##### `transform_3d(points: List[List[float]], transformation: Dict) -> Dict`

3D几何变换。

**变换类型：**
- `"translation"`: 平移，参数 `vector: [tx, ty, tz]`
- `"rotation"`: 旋转，参数 `axis: str ("x"/"y"/"z"), angle: float (弧度)`
- `"scaling"`: 缩放，参数 `factors: [sx, sy, sz]`

**示例：**
```python
points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# 平移
result = engine.transform_3d(
    points=points,
    transformation={"type": "translation", "vector": [1, 1, 1]}
)
print(result['transformed_points'])  # [[2,1,1], [1,2,1], [1,1,2]]

# 旋转
import numpy as np
result = engine.transform_3d(
    points=points,
    transformation={"type": "rotation", "axis": "z", "angle": np.pi/2}
)
```

##### `projection(points_3d: List[List[float]], projection_type: str = "perspective", **kwargs) -> Dict`

3D到2D投影。

**投影类型：**
- `"perspective"`: 透视投影，参数 `focal_length: float`
- `"orthographic"`: 正交投影
- `"stereographic"`: 立体投影

**示例：**
```python
points_3d = [[1, 2, 3], [4, 5, 6]]

result = engine.projection(
    points_3d=points_3d,
    projection_type="perspective",
    focal_length=100
)
print(result['projected_points_2d'])  # [[x1, y1], [x2, y2]]
```

##### `collision_detection(shape1: Dict, shape2: Dict) -> Dict`

几何体碰撞检测。

**支持形状：** "sphere", "box", "cylinder", "plane"

**示例：**
```python
sphere = {
    "type": "sphere",
    "center": [0, 0, 0],
    "radius": 1.0
}

box = {
    "type": "box",
    "center": [2, 0, 0],
    "dimensions": [1, 1, 1]
}

result = engine.collision_detection(sphere, box)
print(result['collision'])  # False
print(result['distance'])  # 1.0
```

##### `volume_calculation(shape: Dict) -> Dict`

计算几何体体积。

**示例：**
```python
sphere = {"type": "sphere", "radius": 1.0}
result = engine.volume_calculation(sphere)
print(result['volume'])  # 4.189 (4π/3)

cylinder = {"type": "cylinder", "radius": 1.0, "height": 2.0}
result = engine.volume_calculation(cylinder)
print(result['volume'])  # 6.283 (2π)
```

##### `curve_operations(curve_type: str, **kwargs) -> Dict`

曲线操作。

**操作类型：**
- `"interpolation"`: 插值，参数 `points, method`
- `"fitting"`: 拟合，参数 `data, degree`
- `"length"`: 弧长计算
- `"curvature"`: 曲率计算

**示例：**
```python
# 曲线插值
result = engine.curve_operations(
    curve_type="interpolation",
    points=[[0, 0], [1, 1], [2, 4]],
    method="cubic_spline"
)

# 曲线拟合
result = engine.curve_operations(
    curve_type="fitting",
    data=[[1, 2], [2, 4], [3, 6]],
    degree=1
)
print(result['coefficients'])  # [0, 2] (y = 2x)
```

---

## MathLearningEngine

自适应学习和性能优化引擎。

#### 初始化

```python
from math_component.engines import MathLearningEngine

engine = MathLearningEngine(math_core)
```

#### Methods

##### `record_performance(method: str, operation: str, parameters: Dict, execution_time: float, success: bool, **kwargs) -> None`

记录操作性能数据。

**参数：**
- `method` (str): 引擎名称 ("symbolic", "numerical", 等)
- `operation` (str): 操作名称 ("differentiate", "solve", 等)
- `parameters` (Dict): 操作参数
- `execution_time` (float): 执行时间（秒）
- `success` (bool): 是否成功
- `error` (str, optional): 错误信息
- `result_quality` (float, optional): 结果质量 (0-1)

**示例：**
```python
engine.record_performance(
    method="symbolic",
    operation="differentiate",
    parameters={"expression": "x**2", "variable": "x"},
    execution_time=0.005,
    success=True,
    result_quality=1.0
)
```

##### `get_learning_summary() -> Dict`

获取学习数据摘要。

**返回：**
```python
{
    "total_operations": int,
    "success_rate": float,
    "by_method": {
        "symbolic": {"count": int, "avg_time": float, ...},
        ...
    },
    "top_operations": List[Tuple[str, int]]
}
```

##### `adaptive_optimization(method: str, operation: str) -> Dict`

基于历史数据提供优化建议。

**返回：**
```python
{
    "method": str,
    "operation": str,
    "statistics": {
        "total_calls": int,
        "successful_calls": int,
        "avg_time": float,
        "success_rate": float
    },
    "cache_hit_rate": float,
    "optimization_level": str,  # "excellent", "good", "needs_improvement"
    "recommendations": List[str]
}
```

**示例：**
```python
optimization = engine.adaptive_optimization("symbolic", "differentiate")
print(optimization['optimization_level'])  # "excellent"
print(optimization['recommendations'])  # []
```

##### `predict_complexity(method: str, operation: str, parameters: Dict) -> Dict`

预测操作复杂度。

**返回：**
```python
{
    "estimated_time": float,
    "complexity_class": str,  # "O(1)", "O(n)", "O(n^2)", etc.
    "confidence": float
}
```

##### `suggest_alternative(method: str, operation: str, parameters: Dict) -> List[Dict]`

建议替代方案。

**返回：** 替代方法列表，按推荐度排序

---

## Plugin System

### Plugin Base Classes

所有插件继承自 `PluginBase`。

```python
from math_component.plugins import PluginBase, PluginCapability

class MyPlugin(PluginBase):
    def __init__(self, math_core):
        super().__init__(
            name="my_plugin",
            version="1.0.0",
            description="Custom math operations",
            math_core=math_core
        )
        self.register_capability("custom_op", self.custom_operation)
    
    def custom_operation(self, **kwargs):
        # 实现
        return {"result": value}
```

### Plugin Manager

插件加载和管理。

```python
from math_component.plugins import PluginManager

manager = PluginManager(math_core)

# 加载所有插件
results = manager.load_all_plugins()

# 激活插件
manager.activate_all_plugins()

# 获取插件
plugin = manager.get_plugin("matrix_operations")

# 执行插件操作
result = plugin.execute_capability("eigenvalue", matrix=[[1, 2], [2, 1]])
```

### Built-in Plugins

#### Matrix Operations Plugin

高级矩阵运算。

**能力：**
- `lu_decomposition`: LU分解
- `qr_decomposition`: QR分解
- `cholesky_decomposition`: Cholesky分解
- `eigenvalue`: 特征值和特征向量
- `matrix_exponential`: 矩阵指数
- `matrix_logarithm`: 矩阵对数

#### Statistics Plugin

统计分析工具。

**能力：**
- `descriptive_stats`: 描述性统计
- `hypothesis_tests`: 假设检验（t检验、卡方检验等）
- `distributions`: 概率分布（正态、泊松、二项等）
- `regression`: 回归分析（线性、多项式等）

**示例：**
```python
result = plugin.execute_capability(
    "descriptive_stats",
    data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)
print(result['statistics'])
# {
#   "mean": 5.5,
#   "median": 5.5,
#   "std": 2.87,
#   "min": 1.0,
#   "max": 10.0,
#   ...
# }
```

---

## AGI Integration

### AGI Math Bridge

统一接口，连接所有引擎。

```python
from math_component.integration import create_agi_math_bridge

bridge = create_agi_math_bridge()
```

#### Methods

##### `execute_math_operation(operation: str, category: str, **kwargs) -> Dict`

执行数学操作。

**类别：** "symbolic", "numerical", "physics", "geometry", "statistics", "matrix_advanced"

**返回格式：**
```python
{
    "success": bool,
    "result": Any,
    "error": Optional[str],
    "metadata": {
        "category": str,
        "operation": str,
        "engine": str,
        "execution_time": float,
        "timestamp": str
    }
}
```

**示例：**
```python
# 符号求导
result = bridge.execute_math_operation(
    operation="differentiate",
    category="symbolic",
    expression="x**3 + 2*x",
    variable="x"
)

# 数值积分
result = bridge.execute_math_operation(
    operation="numerical_integrate",
    category="numerical",
    func=lambda x: x**2,
    a=0,
    b=1
)

# 统计分析
result = bridge.execute_math_operation(
    operation="descriptive_stats",
    category="statistics",
    data=[1, 2, 3, 4, 5]
)
```

##### `get_capabilities() -> Dict[str, List[str]]`

获取所有能力。

**返回：**
```python
{
    "symbolic": ["differentiate", "integrate", "solve", ...],
    "numerical": ["solve_ode", "optimize", ...],
    ...
}
```

##### `query_natural_language(query: str) -> Dict`

自然语言查询匹配。

**示例：**
```python
result = bridge.query_natural_language("solve equation")
print(result['matches'])
# [
#   {
#     "category": "symbolic",
#     "confidence": 0.9,
#     "operations": ["solve", ...]
#   }
# ]
```

##### `get_learning_insights(problem_type: Optional[str] = None) -> Dict`

获取学习统计。

**示例：**
```python
insights = bridge.get_learning_insights("symbolic.differentiate")
print(insights)
# {
#   "problem_type": "symbolic.differentiate",
#   "record_count": 100,
#   "average_time": 0.005,
#   "best_time": 0.001,
#   "worst_time": 0.02,
#   "success_rate": 1.0
# }
```

##### `optimize_for_problem_type(problem_type: str) -> Dict`

针对特定问题类型优化。

**参数格式：** `"category.operation"`（如 "symbolic.differentiate"）

### AGI Math Tool

便捷快捷方法包装。

```python
from math_component.integration import create_agi_math_tool

tool = create_agi_math_tool()

# 快捷方法
result = tool.differentiate("x**2", "x")
result = tool.integrate("x**2", "x")
result = tool.solve_equation("x**2 - 4", "x")
result = tool.matrix_operation("eigenvalue", [[1, 2], [2, 1]])
result = tool.statistical_analysis([1, 2, 3, 4, 5])

# 工具信息
info = tool.get_tool_info()
print(info['name'])  # "math_component"
print(info['version'])  # "1.0.0"
```

---

## Data Structures

### MathExpression

```python
@dataclass
class MathExpression:
    expression: Any  # SymPy expression or value
    variables: List[sympy.Symbol]
    assumptions: Dict[str, Any]
    metadata: Dict[str, Any]
```

### PerformanceRecord

```python
@dataclass
class PerformanceRecord:
    timestamp: datetime
    method: str
    operation: str
    parameters: Dict
    execution_time: float
    success: bool
    error: Optional[str]
    result_quality: float
```

---

## Error Handling

### Exception Hierarchy

```
MathComponentError (base)
├── ConfigurationError
├── EngineError
│   ├── SymbolicError
│   ├── NumericalError
│   ├── PhysicsError
│   └── GeometryError
├── PluginError
│   ├── PluginLoadError
│   ├── PluginActivationError
│   └── PluginExecutionError
└── IntegrationError
```

### Error Handling Best Practices

```python
from math_component.core.exceptions import MathComponentError, EngineError

try:
    result = engine.differentiate("invalid expression", "x")
except SymbolicError as e:
    print(f"Symbolic computation failed: {e}")
except EngineError as e:
    print(f"Engine error: {e}")
except MathComponentError as e:
    print(f"General error: {e}")
```

### Common Error Codes

| 错误码 | 描述 | 解决方案 |
|--------|------|---------|
| `E001` | Invalid expression | 检查表达式语法 |
| `E002` | Unsupported operation | 查看支持的操作列表 |
| `E003` | Convergence failure | 调整参数或初始猜测 |
| `E004` | Singular matrix | 检查矩阵条件数 |
| `E005` | Plugin not found | 确认插件已加载 |

---

## Performance Tips

1. **启用缓存**：重复计算自动缓存
2. **批量操作**：使用批量接口减少开销
3. **并行计算**：适用于独立子问题
4. **精度控制**：根据需求选择精度级别
5. **学习优化**：利用自适应学习引擎的建议

---

## Version History

- **1.0.0** (2025-11-15): 初始发布
  - 5个核心引擎
  - 2个内置插件
  - AGI集成接口
  - 自适应学习系统

---

## Support

**Documentation:** https://github.com/your-repo/docs  
**Issues:** https://github.com/your-repo/issues  
**Email:** support@example.com

---

*Generated with Math Component API Documentation Generator*
