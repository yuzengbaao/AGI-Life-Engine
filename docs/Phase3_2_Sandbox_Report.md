# Phase 3.2 虚拟沙箱 (The Crib) 实施报告

**日期**: 2025-12-24
**状态**: ✅ 已验证 / Ready for Deployment

## 1. 概述
本报告记录了 AGI 系统从 Phase 2.3 向 Phase 3.2 过渡的关键基础设施——虚拟沙箱 (The Crib) 的设计、实施与验证过程。
我们成功将单一的实验环境重构为分层架构，引入了支持自我进化的 "ShadowRunner"。

## 2. 架构设计：分层沙箱

为了兼顾外部代码的安全执行与内部代码的自我迭代，我们采用了以下分层策略：

### 2.1 基类: IsolatedExecutor
- **位置**: `core/research/lab.py`
- **职责**: 封装底层的 `subprocess` 调用、超时控制 (默认为 5s/10s) 和 I/O 捕获。
- **改进**: 
    - 强制设置 `PYTHONIOENCODING=utf-8`，解决了 Windows 环境下子进程输出中文乱码导致测试失败的问题。
    - 统一了日志格式。

### 2.2 子类 A: ResearchLab (游乐场)
- **目标**: 执行不可信的、生成的代码片段（如数学验证、逻辑测试）。
- **安全策略**:
    - **严格隔离**: `PYTHONPATH` 被清空，禁止访问 `core`。
    - **白名单**: 仅允许 `math`, `numpy`, `torch` 等科学计算库。
    - **AST 扫描**: 静态分析代码，拦截 `os`, `sys` 等危险导入。

### 2.3 子类 B: ShadowRunner (手术室)
- **目标**: 允许 AGI 修改自身代码并在安全环境中测试，支持 Phase 3.2 的自我进化循环。
- **核心机制**: **智能影子 (Smart Shadowing)**
    - **Copy-on-Write**: 仅复制被修改的文件到临时目录，而非全量复制项目。
    - **动态链接**: 通过 `PYTHONPATH` 优先加载影子目录，实现“覆盖”效果。
    - **Dry Run (空跑)**: 在运行测试前，尝试 `import` 修改后的模块。如果导入失败（语法错误、依赖循环），直接拦截。

## 3. 实施细节

### 3.1 代码重构
- 文件 `core/research/lab.py` 已完全重写。
- 引入了 `uuid` 用于生成唯一的 Session ID。
- 增加了 `cleanup` 方法以自动清理临时影子目录。

### 3.2 测试验证
- 创建了全新的测试套件: `tests/test_sandbox_v2.py`
- 包含三个关键测试用例:
    1. `test_research_lab_isolation`: 验证是否成功拦截 `import os`。
    2. `test_research_lab_allowed`: 验证是否允许 `import math`。
    3. `test_shadow_runner_dry_run`: 验证影子环境创建、模块 Mock 注入、以及 `dry_run` 导入检查功能。

## 4. 验证结果

运行命令: `python tests/test_sandbox_v2.py`

```text
=== 测试 ResearchLab 允许的模块 ===
结果: ... 3.141592653589793
...
=== 测试 ResearchLab 隔离性 ===
🚫 实验被拒绝: 导入 'os' 不在白名单中。
...
=== 测试 ShadowRunner 空跑机制 ===
影子环境创建于: ...\session_ae89fc48
[SUCCESS] 导入成功
...
Ran 3 tests in 0.906s

OK
```

**结论**: 所有测试通过。系统已具备 Phase 3.2 所需的安全沙箱基础。

## 5. 下一步计划

1. **集成 Planner**: 将 `ShadowRunner` 集成到 AGI 的 `SelfEvolutionPlanner` 中。
2. **错误诊断**: 连接 `analyze_traceback` 工具，让 AGI 能理解 Dry Run 失败的原因。
3. **热回滚**: 基于 ShadowRunner 的机制，设计系统级的自动回滚策略。
