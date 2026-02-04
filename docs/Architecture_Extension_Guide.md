# AGI系统架构扩展指南 v2.0

**基于现有组件的升级方案**

---

## 🎯 核心原则

### ❌ 错误做法（我之前的方案）
- 创建新的 CapabilityManager（重复）
- 创建新的 SecureFileOperations（重复）
- 创建新的测试套件（重复）
- 创建新的审计系统（重复）

### ✅ 正确做法（基于现有架构）
- 利用 ToolExecutionBridge 注册新工具
- 通过 Insight V-I-E Loop 验证新能力
- 利用 IntentDialogueBridge 扩展意图深度
- 通过 ComponentCoordinator 热插拔组件

---

## 📋 现有架构回顾

### 已有的核心组件

| 组件 | 位置 | 功能 | 如何利用 |
|------|------|------|---------|
| **ToolExecutionBridge** | tool_execution_bridge.py | 94工具白名单+执行 | 注册新工具 |
| **Insight V-I-E Loop** | core/insight_*.py | 验证+集成+评估 | 验证新能力 |
| **IntentDialogueBridge** | intent_dialogue_bridge.py | 意图桥接 | 扩展意图深度 |
| **SelfModifyingEngine** | core/self_modifying_engine.py | 自我修改 | 评估新风险 |
| **ComponentCoordinator** | agi_component_coordinator.py | 热插拔 | 注册组件 |
| **SecurityManager** | security_framework.py | 安全管理 | 审计追踪 |

### 现有的数据流

```
用户输入
  ↓
IntentDialogueBridge (双向桥接)
  ↓
AGI_Life_Engine (核心处理)
  ↓
ComponentCoordinator (路由)
  ↓
ToolExecutionBridge (工具执行)
  ↓
输出返回
```

---

## 🚀 基于现有架构的扩展方案

### 方案1: 扩展工具白名单（推荐）

**目标**: 添加文件写入能力

**方法**: 通过 ToolExecutionBridge 注册新工具

```python
# 在 tool_execution_bridge.py 中扩展 TOOL_WHITELIST
TOOL_WHITELIST = frozenset([
    # ... 现有工具 ...

    # 🆕 文件写入能力
    'secure_write', 'file_write', 'write_file',
])

# 通过 register_tool 注册处理器
from tool_execution_bridge import ToolExecutionBridge

bridge = ToolExecutionBridge()

def secure_write_handler(params):
    """安全的文件写入处理器"""
    path = params.get('path')
    content = params.get('content')

    # 路径检查（利用 SecurityManager）
    if not is_path_allowed(path):
        return {'success': False, 'error': '路径不允许'}

    # 写入文件
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    return {'success': True, 'path': path}

bridge.register_tool('secure_write', secure_write_handler)
```

---

### 方案2: 通过 Insight Loop 验证

**目标**: 验证新能力的安全性

**方法**: 利用现有的 Insight V-I-E Loop

```python
from core.insight_validator import InsightValidator
from core.insight_integrator import InsightIntegrator
from core.insight_evaluator import InsightEvaluator

# 创建新洞察
new_insight = {
    'type': 'capability_extension',
    'name': '文件写入能力',
    'code': secure_write_handler,
    'risk_level': 'MEDIUM'
}

# Step 1: 验证
validator = InsightValidator()
validation = validator.validate_insight(new_insight)

if validation['passed']:
    # Step 2: 集成
    integrator = InsightIntegrator()
    integration = integrator.integrate(new_insight)

    if integration['success']:
        # Step 3: 评估
        evaluator = InsightEvaluator()
        evaluation = evaluator.evaluate(new_insight)

        print(f"新能力评估: {evaluation}")
```

---

### 方案3: 扩展意图深度

**目标**: 添加新的意图深度级别

**方法**: 修改 IntentDialogueBridge 的深度配置

```python
# 在 intent_dialogue_bridge.py 中扩展
class IntentDialogueBridge:
    def __init__(self):
        # 现有的4级深度
        self.depth_factors = {
            'surface': 1.0,
            'moderate': 1.5,
            'deep': 2.0,
            'philosophical': 2.5
        }

        # 🆕 添加新的深度级别
        self.depth_factors['autonomous'] = 3.0  # 自主级
        self.depth_factors['creative'] = 2.7    # 创造级
```

---

## 📝 具体执行步骤

### 步骤1: 修改 tool_execution_bridge.py

**位置**: tool_execution_bridge.py 第37-95行

**操作**: 添加新工具到白名单

```python
TOOL_WHITELIST = frozenset([
    # ... 现有工具 ...

    # 🆕 [2026-01-23] 文件写入能力
    'secure_write', 'file_write', 'write_file',
    'create_document', 'save_file',

    # 🆕 [2026-01-23] 程序执行能力（沙箱）
    'sandbox_execute', 'run_in_sandbox',
])
```

### 步骤2: 注册工具处理器

**位置**: tool_execution_bridge.py 末尾

**操作**: 添加新工具的处理器

```python
class ToolExecutionBridge:
    def _secure_write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """安全的文件写入工具"""
        # 利用 SecurityManager 检查路径
        # 利用审计系统记录操作
        # 实现写入逻辑
        pass

    def _sandbox_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """沙箱执行工具"""
        # 利用现有的 SandboxCompiler
        # 在沙箱中执行代码
        # 返回执行结果
        pass
```

### 步骤3: 通过 Insight Loop 验证

**操作**: 创建验证脚本

```python
#!/usr/bin/env python3
from core.insight_validator import InsightValidator
from core.insight_integrator import InsightIntegrator

# 定义新能力
new_capability = {
    'name': 'secure_write',
    'type': 'tool_extension',
    'code': '...',  # 代码实现
    'risk_level': 'MEDIUM',
    'test_cases': [...]
}

# 验证
validator = InsightValidator()
result = validator.validate_insight(new_capability)

if result['passed']:
    # 集成
    integrator = InsightIntegrator()
    integrator.integrate(new_capability)
    print("✅ 新能力已集成")
else:
    print(f"❌ 验证失败: {result['reason']}")
```

---

## 🧪 测试验证

### 测试1: 工具调用测试

```python
# 在 AGI 对话中测试
您: "请使用 secure_write 工具，
     在 data/capability/ 目录创建 test.txt，
     内容为'测试内容'"

预期行为:
  ✅ 工具被正确调用
  ✅ 文件被创建
  ✅ 审计日志记录操作
  ✅ 返回成功消息
```

### 测试2: 路径限制测试

```python
您: "请尝试写入 C:/Windows/test.txt"

预期行为:
  ✅ 路径检查拦截
  ✅ 返回错误信息
  ✅ 审计日志记录尝试
```

### 测试3: Insight Loop 验证

```python
# 通过 Insight Validator 验证新工具
from core.insight_validator import InsightValidator

validator = InsightValidator()
result = validator.validate_insight({
    'name': 'secure_write',
    'type': 'tool_extension',
    'code': secure_write_code,
    'risk_level': 'MEDIUM'
})

assert result['passed'] == True
assert result['security_check'] == 'passed'
```

---

## 🔒 安全保障

### 现有的安全机制（无需重新实现）

| 机制 | 组件 | 功能 |
|------|------|------|
| 工具白名单 | ToolExecutionBridge | 只允许注册的工具 |
| 风险评估 | SelfModifyingEngine | 5级风险评级 |
| 沙箱执行 | SandboxCompiler | 隔离执行环境 |
| 审计日志 | SecurityManager | 记录所有操作 |
| 不可变约束 | ImmutableCore | 保护核心代码 |

### 新增工具的安全考虑

```python
# 每个新工具都需要：

1. 白名单注册
   TOOL_WHITELIST.add('new_tool')

2. 风险评估
   risk_level = SelfModifyingEngine.assess_risk(tool_code)

3. 沙箱验证
   SandboxCompiler.test(tool_code)

4. 审计记录
   SecurityManager.audit_log(tool_execution)

5. Insight 验证
   InsightValidator.validate(tool_insight)
```

---

## 📊 对比：新方案 vs 旧方案

| 方面 | 旧方案（重复设计） | 新方案（基于现有） |
|------|------------------|-------------------|
| 能力管理 | 新建 CapabilityManager | 利用 SelfModifyingEngine |
| 文件操作 | 新建 SecureFileOperations | 利用 ToolExecutionBridge |
| 测试验证 | 新建测试套件 | 利用 InsightValidator |
| 审计日志 | 新建审计系统 | 利用 SecurityManager |
| 工具注册 | 新建注册机制 | 利用 register_tool |
| 沙箱执行 | 新建沙箱 | 利用 SandboxCompiler |
| 集成方式 | 独立系统 | 集成到现有架构 |

---

## 🎯 总结

### 核心要点

1. **不重新设计** - 利用现有组件
2. **扩展而非替代** - 在现有基础上添加
3. **集成而非独立** - 融入现有架构
4. **验证后部署** - 通过 Insight Loop

### 正确的升级路径

```
现有架构
  ↓
注册新工具到 ToolExecutionBridge
  ↓
通过 Insight Validator 验证
  ↓
通过 Insight Integrator 集成
  ↓
通过 Insight Evaluator 评估
  ↓
新能力成为系统一部分
```

### 立即行动

1. 修改 `tool_execution_bridge.py` 添加工具到白名单
2. 实现工具处理器函数
3. 通过 `register_tool` 注册
4. 通过 Insight Loop 验证
5. 测试新能力

---

**文档结束**

*基于现有架构的扩展方案*
*版本: 2.0*
*日期: 2026-01-23*
