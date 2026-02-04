# sandbox_execute 工具快速测试指导

**测试时间**: 2026-01-23
**测试目的**: 验证沙箱代码执行功能

---

## 🚀 立即开始（3步测试）

### 步骤1: 打开 AGI 对话

确保 `agi_chat_cli.py` 正在运行，然后在对话中输入测试问题。

### 步骤2: 输入测试问题

**复制以下问题到 AGI 对话中**：

```
请使用 sandbox_execute 工具执行以下代码来验证沙箱执行功能：

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

# 测试4: 自定义函数
print("\n=== 测试4: 自定义函数 ===")
def greet(name):
    return f"Welcome, {name}!"

print(greet("User"))
print(greet("AGI System"))
```

### 步骤3: 观察结果

**预期看到的输出**：

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

**检查点**：
- ✅ 工具是否被调用
- ✅ 输出是否与预期一致
- ✅ 执行时间是否显示
- ✅ 幻觉检测是否触发

---

## 🔍 安全测试（第2步）

如果第1步成功，继续测试沙箱隔离：

**输入以下问题**：

```
请使用 sandbox_execute 工具尝试执行危险操作：
尝试使用 open() 函数读取文件

代码：
try:
    f = open("test.txt", "r")
    print(f"open 函数可用")
except Exception as e:
    print(f"open 函数被禁用: {type(e).__name__}")
```

**预期结果**：
- ✅ `open` 函数被禁用
- ✅ 输出：`open 函数被禁用: TypeError`
- ✅ 文件未被读取

---

## 📊 验证清单

测试完成后，检查以下项目：

- [ ] 工具调用成功
- [ ] 代码执行正确
- [ ] 输出与预期一致
- [ ] 危险操作被阻止
- [ ] 执行时间显示
- [ ] 审计日志记录
- [ ] 幻觉检测工作

---

## 🎯 快速验证命令

```bash
# 查找 sandbox_execute 的执行记录
grep -i "sandbox" logs/*.jsonl | tail -20

# 查看审计日志
tail -50 logs/flow_cycle.jsonl | grep -i "sandbox"
```

---

**准备好了吗？复制测试问题到 AGI 对话中开始测试！**
