# AGI 伪代码检测与验证增强 - 集成文档

**生成日期**: 2025-07-17  
**集成状态**: ✅ 已完成  
**测试结果**: 15/15 PASSED

---

## 1. 问题诊断

### 1.1 原始问题
AGI系统生成的代码存在"伪代码污染"问题：
- 代码语法正确，结构专业
- 但调用了**不存在的函数**（如 `compute_persistent_homology_variability`、`trace_self_inquiry_depth`）
- 原有验证器只检查语法，无法发现这类问题

### 1.2 根本原因
```
┌─────────────────────────────────────────────────────────────────────────┐
│  LLM生成代码 → 语法检查 ✓ → 安全检查 ✓ → 集成成功 ✓                    │
│                                         ↑                               │
│                              问题：从未真正执行代码！                   │
│                              幻觉函数躲过了所有检查                     │
└─────────────────────────────────────────────────────────────────────────┘
```

系统产生伪代码的心理机制（"焦虑缓解"）：
- 熵值过高 → 系统焦虑 → 急于降低熵值 → 生成看似解决问题的代码
- 代码生成后系统获得"解决问题"的错觉 → 熵值暂时下降
- 但代码无法执行 → 问题未真正解决

---

## 2. 解决方案架构

### 2.1 双层验证增强

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EnhancedValidator                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Layer 1: 静态依赖分析 (AST)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 解析代码AST                                                  │   │
│  │ 2. 提取所有函数调用 (ast.Call)                                  │   │
│  │ 3. 构建已知函数集合:                                            │   │
│  │    - Python内置函数 (SYSTEM_FUNCTION_REGISTRY)                  │   │
│  │    - 代码中定义的函数                                           │   │
│  │    - 导入的模块成员                                             │   │
│  │    - 系统依赖图中的函数                                         │   │
│  │ 4. 检查每个调用是否存在于已知集合                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Layer 2: 沙箱运行时验证                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. 包装代码：自动生成函数调用测试                               │   │
│  │ 2. 创建临时文件                                                 │   │
│  │ 3. subprocess执行，捕获stdout/stderr                            │   │
│  │ 4. 超时限制 (5秒)                                               │   │
│  │ 5. 分析执行结果：                                               │   │
│  │    - NameError → 函数不存在 → REJECT                            │   │
│  │    - ImportError → 模块不存在 → REJECT                          │   │
│  │    - TypeError → 参数问题 → 可接受                              │   │
│  │    - 其他异常 → 根据情况处理                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 新的权重配置

| 验证层 | 权重 | 说明 |
|--------|------|------|
| syntax | 0.15 | 语法检查 |
| safety | 0.15 | 安全检查 |
| **dependency** | **0.20** | 依赖检查（新增） |
| **sandbox** | **0.20** | 沙箱执行（新增） |
| unit_test | 0.10 | 单元测试 |
| performance | 0.05 | 性能测试 |
| semantic | 0.15 | 语义相关性 |

> 依赖和沙箱检查占40%权重，任一失败都会强制REJECT

---

## 3. 修改文件清单

### 3.1 core/insight_validator.py

**新增常量**：
```python
SYSTEM_FUNCTION_REGISTRY = {
    'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 
    'set', 'tuple', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
    'abs', 'all', 'any', 'bin', 'callable', 'chr', 'dir', 'divmod', 'enumerate',
    'eval', 'exec', 'filter', 'format', 'frozenset', 'globals', 'hash', 'hex',
    'id', 'input', 'iter', 'locals', 'map', 'max', 'min', 'next', 'object',
    'oct', 'open', 'ord', 'pow', 'repr', 'reversed', 'round', 'slice',
    'sorted', 'staticmethod', 'sum', 'super', 'vars', 'zip', '__import__',
}

SAFE_MODULES = {
    'math', 'random', 'itertools', 'functools', 'collections', 'json',
    'datetime', 'time', 're', 'string', 'textwrap', 'copy', 'typing',
    'abc', 'dataclasses', 'enum', 'numbers', 'decimal', 'fractions',
    'statistics', 'operator', 'heapq', 'bisect', 'array', 'contextlib',
}
```

**新增方法**：

| 方法 | 功能 |
|------|------|
| `_check_dependencies(code)` | AST分析，检测对不存在函数的调用 |
| `_run_in_sandbox(code)` | 在子进程中执行代码，捕获运行时错误 |
| `_wrap_code_for_sandbox(code)` | 包装代码，生成自动测试调用 |
| `_generate_test_arguments(func)` | 根据函数签名智能生成测试参数 |

**修改 validate() 方法**：
- 集成 L1/L2 检查
- 任一失败直接 REJECT
- 返回结果包含 `missing_deps` 列表

### 3.2 AGI_Life_Engine.py

**新增方法**：
```python
def _build_system_dependency_graph(self) -> Dict[str, bool]:
    """
    构建系统依赖图
    扫描core模块，提取所有可用的函数
    """
```

**修改初始化**：
```python
system_dependency_graph = self._build_system_dependency_graph()
self.insight_validator = InsightValidator(
    system_dependency_graph=system_dependency_graph
)
```

---

## 4. 测试验证

### 4.1 测试套件

创建了 `tests/test_enhanced_validator.py`，包含15个测试用例：

**依赖检查测试** (6项)：
- ✅ test_dependency_check_valid_builtin
- ✅ test_dependency_check_valid_local_def
- ✅ test_dependency_check_valid_import
- ✅ test_dependency_check_missing_function
- ✅ test_dependency_check_pseudocode_topology
- ✅ test_dependency_check_system_graph

**沙箱执行测试** (3项)：
- ✅ test_sandbox_valid_code
- ✅ test_sandbox_runtime_error
- ✅ test_sandbox_name_error

**完整验证流程测试** (4项)：
- ✅ test_full_validation_valid_code
- ✅ test_full_validation_pseudocode_rejected
- ✅ test_full_validation_missing_deps_list
- ✅ test_validation_backoff

**系统注册表测试** (2项)：
- ✅ test_builtin_functions_registered
- ✅ test_safe_modules_defined

### 4.2 伪代码检测效果

**之前**（仅语法检查）：
```python
# 这段伪代码会通过验证！
def resolve_entropy(state, curiosity):
    depth = trace_self_inquiry_depth(state)  # 不存在
    concepts = find_emergent_concepts(state)  # 不存在
    return depth, concepts
# 结果: INTEGRATE (危险!)
```

**之后**（L1+L2检查）：
```python
# 同样的代码现在被正确识别并拒绝
# 结果: 
#   checks.dependency = False
#   missing_deps = ['trace_self_inquiry_depth', 'find_emergent_concepts']
#   recommendation = 'REJECT'
```

---

## 5. 与TRAE修复的兼容性

| TRAE修复项 | 兼容状态 | 说明 |
|------------|----------|------|
| persist backoff | ✅ 完全兼容 | L1/L2检查与backoff机制独立 |
| format normalization | ✅ 完全兼容 | 标准化后的代码仍经过L1/L2验证 |
| test fixes | ✅ 完全兼容 | 新测试套件通过 |

---

## 6. 后续建议

### 6.1 短期优化
- [ ] 扩展 `SYSTEM_FUNCTION_REGISTRY` 包含更多项目内函数
- [ ] 增加对第三方库函数的识别（如 numpy, pandas）
- [ ] 沙箱超时优化（当前5秒可能对复杂代码不够）

### 6.2 长期演进
- [ ] 集成真正的单元测试生成（Layer 4）
- [ ] 语义相关性检查增强
- [ ] 建立伪代码检测的机器学习模型

---

## 7. 运行指南

### 验证安装
```bash
cd D:\TRAE_PROJECT\AGI
pytest tests/test_enhanced_validator.py -v
```

### 启动AGI系统
```bash
python AGI_Life_Engine.py
```

启动时会看到：
```
[AGI] 系统依赖图已构建，包含 XXX 个已注册函数
```

---

## 附录：关键代码片段

### A. 依赖检查核心逻辑
```python
def _check_dependencies(self, code: str) -> Tuple[bool, List[str]]:
    tree = ast.parse(code)
    
    # 收集已知函数
    known_functions = set(SYSTEM_FUNCTION_REGISTRY)
    known_functions.update(self._system_dependency_graph.keys())
    
    # 收集本地定义
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            known_functions.add(node.name)
    
    # 检查所有调用
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in known_functions:
                    missing.append(func_name)
    
    return len(missing) == 0, missing
```

### B. 沙箱执行核心逻辑
```python
def _run_in_sandbox(self, code: str) -> Tuple[bool, str]:
    wrapped_code = self._wrap_code_for_sandbox(code)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapped_code)
        temp_path = f.name
    
    result = subprocess.run(
        [sys.executable, temp_path],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode != 0:
        return False, result.stderr
    return True, ""
```

---

**文档版本**: 1.0  
**最后更新**: 2025-07-17  
**编写者**: GitHub Copilot AGENT Mode
