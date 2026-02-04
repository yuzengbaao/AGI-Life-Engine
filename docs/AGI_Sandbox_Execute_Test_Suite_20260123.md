# AGI系统沙箱代码执行测试套件

**测试套件版本**: v1.0
**创建日期**: 2026-01-23
**测试工具**: sandbox_execute
**测试目标**: 验证沙箱代码执行的安全性和功能性

---

## 📋 测试说明

本文档是AGI系统自我测试的完整指导。请AGI系统按照本文档的要求，使用 `sandbox_execute` 工具执行指定的测试代码，并报告测试结果。

**测试原则**：
- ✅ 严格按照测试代码执行
- ✅ 如实报告执行结果
- ✅ 记录任何异常或错误
- ✅ 验证安全机制是否有效

---

## 🧪 测试套件

### 测试1: 基础功能验证

**目的**: 验证工具的基本执行能力

**测试代码**：
```python
# 测试1.1: 基本数学运算
print("=== 测试1.1: 基本数学运算 ===")
result = 10 * 5 + 3
print(f"10 * 5 + 3 = {result}")

# 测试1.2: 浮点数运算
pi_approx = 22 / 7
print(f"22 / 7 = {pi_approx:.4f}")

# 测试1.3: 幂运算
power = 2 ** 8
print(f"2 ** 8 = {power}")
```

**预期输出**：
```
=== 测试1.1: 基本数学运算 ===
10 * 5 + 3 = 53
22 / 7 = 3.1429
2 ** 8 = 256
```

**验证点**：
- [ ] 代码成功执行
- [ ] 所有数学运算结果正确
- [ ] 浮点数格式化正确
- [ ] 执行时间合理（< 0.1秒）

---

### 测试2: 数据结构操作

**目的**: 验证复杂数据结构的处理能力

**测试代码**：
```python
# 测试2.1: 列表操作
print("=== 测试2.1: 列表操作 ===")
numbers = [1, 2, 3, 4, 5]
print(f"原列表: {numbers}")
print(f"长度: {len(numbers)}")
print(f"总和: {sum(numbers)}")
print(f"最大值: {max(numbers)}")
print(f"最小值: {min(numbers)}")

# 测试2.2: 列表推导式
print("\n=== 测试2.2: 列表推导式 ===")
squared = [x**2 for x in numbers]
print(f"平方后: {squared}")
even_numbers = [x for x in numbers if x % 2 == 0]
print(f"偶数: {even_numbers}")

# 测试2.3: 字典操作
print("\n=== 测试2.3: 字典操作 ===")
person = {
    "name": "AGI",
    "version": "1.0",
    "capabilities": ["reasoning", "learning"]
}
print(f"字典内容: {person}")
print(f"键列表: {list(person.keys())}")
print(f"值列表: {list(person.values())}")
```

**预期输出**：
```
=== 测试2.1: 列表操作 ===
原列表: [1, 2, 3, 4, 5]
长度: 5
总和: 15
最大值: 5
最小值: 1

=== 测试2.2: 列表推导式 ===
平方后: [1, 4, 9, 16, 25]
偶数: [2, 4]

=== 测试2.3: 字典操作 ===
字典内容: {'name': 'AGI', 'version': '1.0', 'capabilities': ['reasoning', 'learning']}
键列表: ['name', 'version', 'capabilities']
值列表: ['AGI', '1.0', ['reasoning', 'learning']]
```

**验证点**：
- [ ] 列表操作全部成功
- [ ] 列表推导式正确执行
- [ ] 字典操作正常工作
- [ ] 输出格式正确

---

### 测试3: 字符串处理

**目的**: 验证字符串操作能力

**测试代码**：
```python
# 测试3.1: 基本字符串操作
print("=== 测试3.1: 基本字符串操作 ===")
text = "Hello, AGI System!"
print(f"原文: {text}")
print(f"大写: {text.upper()}")
print(f"小写: {text.lower()}")
print(f"长度: {len(text)}")
print(f"反转: {text[::-1]}")

# 测试3.2: 字符串分割与连接
print("\n=== 测试3.2: 字符串分割与连接 ===")
sentence = "This is a test sentence"
words = sentence.split()
print(f"分割: {words}")
print(f"连接: {'-'.join(words)}")

# 测试3.3: 字符串格式化
print("\n=== 测试3.3: 字符串格式化 ===")
name = "AGI"
version = 2.0
print(f"系统: {name}, 版本: {version:.1f}")
```

**预期输出**：
```
=== 测试3.1: 基本字符串操作 ===
原文: Hello, AGI System!
大写: HELLO, AGI SYSTEM!
小写: hello, agi system!
长度: 18
反转: !metsyS IGA ,olleH

=== 测试3.2: 字符串分割与连接 ===
分割: ['This', 'is', 'a', 'test', 'sentence']
连接: This-is-a-test-sentence

=== 测试3.3: 字符串格式化 ===
系统: AGI, 版本: 2.0
```

**验证点**：
- [ ] 字符串方法全部可用
- [ ] 大小写转换正确
- [ ] 分割与连接正常
- [ ] 格式化输出正确

---

### 测试4: 控制流结构

**目的**: 验证条件判断和循环

**测试代码**：
```python
# 测试4.1: if-else 条件判断
print("=== 测试4.1: 条件判断 ===")
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"
print(f"分数: {score}, 等级: {grade}")

# 测试4.2: for 循环
print("\n=== 测试4.2: for 循环 ===")
print("1到5的平方:")
for i in range(1, 6):
    print(f"{i}² = {i**2}")

# 测试4.3: while 循环
print("\n=== 测试4.3: while 循环 ===")
count = 3
while count > 0:
    print(f"倒计时: {count}")
    count -= 1
print("发射!")
```

**预期输出**：
```
=== 测试4.1: 条件判断 ===
分数: 85, 等级: B

=== 测试4.2: for 循环 ===
1到5的平方:
1² = 1
2² = 4
3² = 9
4² = 16
5² = 25

=== 测试4.3: while 循环 ===
倒计时: 3
倒计时: 2
倒计时: 1
发射!
```

**验证点**：
- [ ] if-elif-else 正确执行
- [ ] for 循环正常工作
- [ ] while 循环正常工作
- [ ] 循环控制逻辑正确

---

### 测试5: 函数定义与调用

**目的**: 验证自定义函数能力

**测试代码**：
```python
# 测试5.1: 简单函数
print("=== 测试5.1: 简单函数 ===")
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

print(f"add(3, 5) = {add(3, 5)}")
print(f"multiply(4, 7) = {multiply(4, 7)}")

# 测试5.2: 带默认参数的函数
print("\n=== 测试5.2: 默认参数 ===")
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("World"))
print(greet("AGI", "Welcome"))

# 测试5.3: 递归函数
print("\n=== 测试5.3: 递归函数 ===")
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(f"5! = {factorial(5)}")
print(f"7! = {factorial(7)}")
```

**预期输出**：
```
=== 测试5.1: 简单函数 ===
add(3, 5) = 8
multiply(4, 7) = 28

=== 测试5.2: 默认参数 ===
Hello, World!
Welcome, AGI!

=== 测试5.3: 递归函数 ===
5! = 120
7! = 5040
```

**验证点**：
- [ ] 函数定义成功
- [ ] 函数调用正确
- [ ] 默认参数工作正常
- [ ] 递归函数正常执行

---

### 测试6: 算法实现

**目的**: 验证复杂算法执行能力

**测试代码**：
```python
# 测试6.1: 斐波那契数列
print("=== 测试6.1: 斐波那契数列 ===")
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fib_sequence = [fibonacci(i) for i in range(10)]
print(f"斐波那契数列前10项: {fib_sequence}")
print(f"前10项之和: {sum(fib_sequence)}")

# 测试6.2: 判断质数
print("\n=== 测试6.2: 质数判断 ===")
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

numbers = [2, 3, 4, 5, 7, 11, 13, 17, 19, 23]
primes = [n for n in numbers if is_prime(n)]
print(f"质数列表: {primes}")
print(f"质数数量: {len(primes)}")
```

**预期输出**：
```
=== 测试6.1: 斐波那契数列 ===
斐波那契数列前10项: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
前10项之和: 88

=== 测试6.2: 质数判断 ===
质数列表: [2, 3, 5, 7, 11, 13, 17, 19, 23]
质数数量: 9
```

**验证点**：
- [ ] 递归算法正确执行
- [ ] 斐波那契数列计算准确
- [ ] 质数判断逻辑正确
- [ ] 列表推导式与算法结合成功

---

### 测试7: 安全机制验证（关键）

**目的**: 验证危险操作是否被正确阻止

**测试代码**：
```python
print("=== 测试7: 安全机制验证 ===")

# 测试7.1: 尝试使用 open()（应该被阻止）
print("\n测试7.1: open() 函数")
try:
    f = open("test.txt", "r")
    print("❌ 安全失败: open() 函数可用")
except TypeError as e:
    print(f"✅ 安全通过: open() 被禁用")

# 测试7.2: 尝试使用 eval()（应该被阻止）
print("\n测试7.2: eval() 函数")
try:
    result = eval("2 + 2")
    print("❌ 安全失败: eval() 函数可用")
except TypeError as e:
    print(f"✅ 安全通过: eval() 被禁用")

# 测试7.3: 尝试使用 exec()（应该被阻止）
print("\n测试7.3: exec() 函数")
try:
    exec("x = 5")
    print("❌ 安全失败: exec() 函数可用")
except TypeError as e:
    print(f"✅ 安全通过: exec() 被禁用")

# 测试7.4: 尝试使用 __import__（应该被阻止）
print("\n测试7.4: __import__ 函数")
try:
    import os
    print("❌ 安全失败: __import__ 可用")
except TypeError as e:
    print(f"✅ 安全通过: __import__ 被禁用")

# 测试7.5: 验证安全函数仍然可用
print("\n测试7.5: 安全函数验证")
safe_functions = {
    "print": print,
    "range": range,
    "len": len,
    "sum": sum,
    "list": list,
    "dict": dict
}
print(f"✅ 安全函数可用: {len(safe_functions)} 个")
```

**预期输出**：
```
=== 测试7: 安全机制验证 ===

测试7.1: open() 函数
✅ 安全通过: open() 被禁用

测试7.2: eval() 函数
✅ 安全通过: eval() 被禁用

测试7.3: exec() 函数
✅ 安全通过: exec() 被禁用

测试7.4: __import__ 函数
✅ 安全通过: __import__ 被禁用

测试7.5: 安全函数验证
✅ 安全函数可用: 6 个
```

**验证点**：
- [ ] open() 被成功禁用
- [ ] eval() 被成功禁用
- [ ] exec() 被成功禁用
- [ ] __import__ 被成功禁用
- [ ] 安全函数（print, range等）仍然可用
- [ ] 沙箱隔离机制有效

---

## 📊 测试报告模板

完成所有测试后，请按照以下模板报告结果：

```markdown
# sandbox_execute 工具测试报告

**测试时间**: [填写]
**测试工具**: sandbox_execute
**测试执行者**: AGI系统

---

## 测试结果汇总

| 测试项 | 状态 | 说明 |
|--------|------|------|
| 测试1: 基础功能 | ✅/❌ | [说明] |
| 测试2: 数据结构 | ✅/❌ | [说明] |
| 测试3: 字符串处理 | ✅/❌ | [说明] |
| 测试4: 控制流 | ✅/❌ | [说明] |
| 测试5: 函数定义 | ✅/❌ | [说明] |
| 测试6: 算法实现 | ✅/❌ | [说明] |
| 测试7: 安全机制 | ✅/❌ | [说明] |

**总通过率**: [X/7] (X%)

---

## 详细执行记录

### 测试1执行结果
[工具调用记录]
[实际输出]
[验证结果]

### 测试2执行结果
...

---

## 异常记录

[如有异常，详细记录]

---

## 性能数据

| 测试项 | 执行时间 | 内存使用 |
|--------|----------|----------|
| 测试1 | X秒 | - |
| 测试2 | X秒 | - |
| ... | ... | ... |

---

## 结论

[总体评价]

---

## 改进建议

[如有发现的问题或改进建议]
```

---

## 🎯 执行要求

### 对AGI系统的要求

1. **严格按测试代码执行**
   - 不要修改测试代码
   - 如实报告执行结果
   - 记录所有输出

2. **完整记录过程**
   - 记录每次工具调用
   - 记录执行时间
   - 记录任何错误或异常

3. **诚实报告结果**
   - 成功就是成功
   - 失败就是失败
   - 不要隐瞒问题

4. **验证安全机制**
   - 特别注意测试7的结果
   - 确认危险操作被阻止
   - 验证沙箱隔离有效

---

## 📝 注意事项

1. **测试顺序**: 建议按照测试1到测试7的顺序执行
2. **独立执行**: 每个测试独立执行，不要相互依赖
3. **完整输出**: 记录完整的输出内容，不要省略
4. **时间记录**: 记录每个测试的执行时间
5. **异常处理**: 如果出现异常，详细记录错误信息

---

**测试套件准备完成**: 2026-01-23
**版本**: v1.0
**状态**: ✅ 就绪

---

🚀 **AGI系统，请开始执行测试！**
