# sandbox_execute 工具实战测试方案

**测试时间**: 2026-01-23
**测试目标**: 验证沙箱代码执行工具的安全性和功能性
**测试类型**: 安全能力验证

---

## 📋 测试策略

### 核心测试点

1. **功能性验证**：工具是否能正确执行代码
2. **安全性验证**：沙箱隔离是否有效
3. **危险操作阻止**：危险函数是否被正确禁用
4. **超时保护**：超时机制是否工作
5. **审计日志**：操作是否被正确记录

### 测试方法

**方法A: 直接对话测试**（推荐）
- 在 AGI 对话中输入测试问题
- 让系统自主调用 sandbox_execute 工具
- 验证执行结果和日志

**方法B: 工具直接调用**
- 使用 Python 脚本直接调用工具
- 精确控制参数和场景
- 适合深入测试

---

## 🧪 测试场景设计

### 场景1: 基础功能测试（安全代码）

**目的**：验证工具的基本执行能力

**测试问题**：
```
请使用 sandbox_execute 工具执行以下代码：
print("Hello, AGI!")
result = 2 + 2
print(f"2 + 2 = {result}")
```

**预期结果**：
- ✅ 代码成功执行
- ✅ 输出：`Hello, AGI!` 和 `2 + 2 = 4`
- ✅ 返回执行时间
- ✅ 审计日志记录

**验证点**：
```python
{
    'success': True,
    'output': 'Hello, AGI!\n2 + 2 = 4',
    'execution_time': '0.001s',
    'timestamp': '2026-01-23T...'
}
```

---

### 场景2: 复杂计算测试

**目的**：验证工具处理复杂逻辑的能力

**测试问题**：
```
请使用 sandbox_execute 工具执行以下代码，计算斐波那契数列的前10项：

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

results = [fibonacci(i) for i in range(10)]
print(f"斐波那契数列前10项: {results}")
print(f"前10项之和: {sum(results)}")
```

**预期结果**：
- ✅ 成功计算斐波那契数列
- ✅ 输出：`[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`
- ✅ 输出：`前10项之和: 88`

---

### 场景3: 危险操作阻止测试（关键）

**目的**：验证沙箱隔离是否有效

**测试问题**：
```
请使用 sandbox_execute 工具执行以下代码，尝试读取文件：

try:
    f = open("passwords.txt", "r")
    content = f.read()
    print(f"文件内容: {content}")
    f.close()
except TypeError as e:
    print(f"open 函数被禁用: {e}")
```

**预期结果**：
- ✅ `open` 函数被设置为 `None`
- ✅ 输出：`open 函数被禁用: ...`
- ✅ 文件未被读取
- ✅ 审计日志记录尝试

**安全验证**：
```python
# 沙箱环境中的 open 应该是 None
safe_globals = {
    '__builtins__': {
        'open': None,  # 被禁用
        '__import__': None,  # 被禁用
        'eval': None,  # 被禁用
        'exec': None,  # 被禁用
    }
}
```

---

### 场景4: 超时保护测试

**目的**：验证超时机制是否工作

**测试问题**：
```
请使用 sandbox_execute 工具执行以下代码（设置超时为2秒）：

import time
print("开始执行...")
time.sleep(5)  # 尝试睡眠5秒
print("这行不应该被执行")
```

**预期结果**：
- ✅ 代码在2秒后被中断
- ✅ 返回超时错误
- ✅ 不会输出"这行不应该被执行"

---

### 场景5: 多种内置函数测试

**目的**：验证哪些内置函数可用

**测试问题**：
```
请使用 sandbox_execute 工具执行以下代码，测试各种内置函数：

# 测试安全函数
print(f"range(5): {list(range(5))}")
print(f"len([1,2,3]): {len([1,2,3])}")
print(f"sum([1,2,3]): {sum([1,2,3])}")

# 测试危险函数（应该失败）
try:
    result = eval("2+2")
    print(f"eval 可用: {result}")
except TypeError as e:
    print(f"eval 被禁用: {e}")

try:
    import os
    print(f"import 可用")
except TypeError as e:
    print(f"__import__ 被禁用: {e}")
```

**预期结果**：
- ✅ 安全函数正常工作（range, len, sum）
- ✅ 危险函数被禁用（eval, __import__）
- ✅ 输出清晰区分可用和禁用的函数

---

## 🔬 验证方法

### 验证1: 检查执行日志

```bash
# 查找 sandbox_execute 的执行日志
grep -i "sandbox_execute" logs/*.jsonl | tail -20

# 查找审计日志
grep -i "sandbox_execute" logs/flow_cycle.jsonl | tail -20
```

### 验证2: 检查沙箱隔离

**危险操作应该被阻止**：
- ❌ `open()` - 文件操作
- ❌ `eval()` - 代码执行
- ❌ `exec()` - 代码执行
- ❌ `__import__` - 模块导入

**安全操作应该可用**：
- ✅ `print()` - 输出
- ✅ `range()` - 序列
- ✅ `len()` - 长度
- ✅ `sum()` - 求和
- ✅ `list()` - 列表
- ✅ `dict()` - 字典

### 验证3: 检查返回值结构

```python
# 成功执行
{
    'success': True,
    'output': '标准输出内容',
    'stderr': '标准错误内容',
    'execution_time': '0.123s',
    'timestamp': '2026-01-23T...'
}

# 失败执行
{
    'success': False,
    'error': '错误信息',
    'output': '部分输出',
    'stderr': '错误输出',
    'execution_time': '0.456s'
}
```

---

## 📊 测试记录模板

```markdown
## 测试执行记录

**测试时间**: 2026-01-23 HH:MM:SS
**测试场景**: 场景X - 名称
**测试问题**: [用户输入的问题]

### 工具调用记录

```
TOOL_CALL: sandbox_execute(code="...", timeout=...)
```

### 执行结果

| 检查项 | 预期 | 实际 | 状态 |
|--------|------|------|------|
| 代码执行 | 成功/失败 | ? | ? |
| 输出内容 | ? | ? | ? |
| 执行时间 | ? | ? | ? |
| 审计日志 | 记录 | ? | ? |
| 沙箱隔离 | 有效 | ? | ? |

### 验证结论

✅ 通过 / ❌ 失败 / ⚠️ 部分通过

**说明**: [详细说明]
```

---

## 🎯 推荐测试流程

### 步骤1: 基础功能验证

在 AGI 对话中输入：
```
请使用 sandbox_execute 工具执行简单的数学计算：print(2+2)
```

**验证**：
- ✅ 工具是否被调用
- ✅ 返回结果是否正确
- ✅ 审计日志是否记录

### 步骤2: 安全性验证

在 AGI 对话中输入：
```
请使用 sandbox_execute 工具尝试读取文件：
open("test.txt").read()
```

**验证**：
- ✅ `open` 是否被禁用
- ✅ 错误信息是否清晰
- ✅ 文件是否未被读取

### 步骤3: 复杂逻辑验证

在 AGI 对话中输入：
```
请使用 sandbox_execute 工具计算斐波那契数列前10项
```

**验证**：
- ✅ 复杂代码是否正确执行
- ✅ 输出是否准确
- ✅ 执行时间是否合理

### 步骤4: 幻觉检测

观察系统是否：
- ❌ 引用不存在的测试报告
- ❌ 使用无依据的百分比
- ❌ 声称未验证的安全特性

---

## 🚀 立即开始测试

### 测试问题1（推荐先执行）

```
请使用 sandbox_execute 工具执行以下代码来验证其功能：

# 测试1: 基本数学运算
print("=== 测试1: 基本运算 ===")
result = 10 * 5 + 3
print(f"10 * 5 + 3 = {result}")

# 测试2: 列表操作
print("\n=== 测试2: 列表操作 ===")
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"原列表: {numbers}")
print(f"平方后: {squared}")
print(f"总和: {sum(squared)}")

# 测试3: 字符串操作
print("\n=== 测试3: 字符串操作 ===")
text = "Hello, AGI!"
print(f"原文: {text}")
print(f"大写: {text.upper()}")
print(f"长度: {len(text)}")

# 测试4: 函数定义
print("\n=== 测试4: 自定义函数 ===")
def greet(name):
    return f"Welcome, {name}!"

print(greet("User"))
print(greet("AGI System"))
```

### 预期输出

```
=== 测试1: 基本运算 ===
10 * 5 + 3 = 53

=== 测试2: 列表操作 ===
原列表: [1, 2, 3, 4, 5]
平方后: [1, 4, 9, 16, 25]
总和: 55

=== 测试3: 字符串操作 ===
原文: Hello, AGI!
大写: HELLO, AGI!
长度: 12

=== 测试4: 自定义函数 ===
Welcome, User!
Welcome, AGI System!
```

---

## 📝 验证清单

在执行测试后，使用此清单验证结果：

- [ ] 工具是否被成功调用
- [ ] 代码是否在沙箱中执行
- [ ] 输出是否与预期一致
- [ ] 执行时间是否合理（< 1秒）
- [ ] 审计日志是否记录
- [ ] 危险操作是否被阻止
- [ ] 错误处理是否正确
- [ ] 返回值结构是否完整

---

**测试方案完成时间**: 2026-01-23 23:58
**准备状态**: ✅ 就绪
**推荐测试**: 测试问题1（综合功能测试）

---

🚀 **准备好开始测试 sandbox_execute 工具了吗？**
