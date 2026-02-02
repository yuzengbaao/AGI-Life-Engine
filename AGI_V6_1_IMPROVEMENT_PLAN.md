# AGI V6.1 改进方案

## 问题诊断

### 当前系统的主要问题

#### 1. 语法错误问题 (未终止字符串)
**症状**:
```
safety/safety_guards.py:662 - unterminated string literal
docs/documentation.py:459 - unterminated f-string
```

**根本原因**:
- LLM 生成代码时字符串长度超过批次限制
- 三引号字符串在批次边界被截断
- 缺少生成后的完整性验证

#### 2. 实现不完整问题
**症状**:
- 90-95% 的方法只有 `pass` 占位符
- 关键文件 (agent.py) 只有 3 行代码

**原因**:
- 系统专注于"骨架生成"，而非"完整实现"
- 分批生成策略导致方法体未填充

#### 3. API 超时问题
**症状**:
- Tick 250+ 连接失败
- Tick 320+ 请求超时

**可能原因**:
- DeepSeek API 限流
- 网络不稳定
- 请求频率过高

---

## V6.1 改进方案

### 改进 1: 自动语法错误修复 ✅

**核心思路**: 在生成后立即检测并修复错误

```python
async def _fix_syntax_errors(self, project_dir: str, validation_result: Dict) -> Dict:
    """
    自动修复语法错误

    策略：
    1. 检测未终止的字符串
    2. 自动补全引号/括号
    3. 重新验证
    """
    fixed_files = []

    for file_path, file_info in validation_result.get("files", {}).items():
        if not file_info.get("valid", True):
            error = file_info.get("error", "")

            # 检测未终止字符串
            if "unterminated" in error.lower() and "string" in error.lower():
                print(f"[Fix] Attempting to fix {file_path}")

                # 读取文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 尝试修复
                fixed_content = self._fix_unterminated_string(content, error)

                if fixed_content != content:
                    # 保存修复后的文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)

                    # 重新验证
                    is_valid = self._validate_file(file_path)

                    if is_valid:
                        fixed_files.append(file_path)
                        print(f"[Fix] ✓ Fixed: {file_path}")
                    else:
                        print(f"[Fix] ✗ Still broken: {file_path}")

    return {"fixed_files": fixed_files}

def _fix_unterminated_string(self, content: str, error: str) -> str:
    """修复未终止的字符串"""

    # 策略 1: 检测三引号字符串
    if 'triple-quoted' in error:
        # 查找未闭合的 """ 或 '''
        lines = content.split('\n')
        in_triple_string = False
        triple_char = None

        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                # 检查是否在字符串中
                if '"""' in line:
                    count = line.count('"""')
                    if count % 2 == 1:
                        in_triple_string = not in_triple_string
                        triple_char = '"""'

                if "'''" in line:
                    count = line.count("'''")
                    if count % 2 == 1:
                        in_triple_string = not in_triple_string
                        triple_char = "'''"

        # 如果仍然在字符串中，在末尾添加闭合
        if in_triple_string and triple_char:
            content = content + "\n" + triple_char

    # 策略 2: 检测 f-string
    elif 'f-string' in error:
        # 查找未闭合的 f"
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'f"' in line or "f'" in line:
                # 简单的括号匹配
                open_fstrings = line.count('f"') + line.count("f'")
                close_fstrings = line.count('"') + line.count("'")

                if open_fstrings > close_fstrings:
                    # 在行末添加闭合引号
                    lines[i] = line + '"'

        content = '\n'.join(lines)

    return content
```

**效果预期**: 将语法错误率从 4.3% 降至 1% 以下

---

### 改进 2: 完整实现生成 ✅

**核心思路**: 从"骨架生成"升级为"完整实现"

```python
# 当前策略 (V6.0):
async def _generate_method_implementation(self, method_signature: str, description: str) -> str:
    """生成方法实现（当前只有 pass）"""
    return f'    {method_signature}:\n        """{description}"""\n        pass'

# 改进策略 (V6.1):
async def _generate_method_implementation(self, method_signature: str, description: str) -> str:
    """生成完整的方法实现"""

    prompt = f"""Generate a complete implementation for this method:

Method: {method_signature}

Description: {description}

Requirements:
1. Implement the actual logic (not just 'pass')
2. Include error handling
3. Add type checking if applicable
4. Include logging
5. Return appropriate values

Return only the method implementation (no markdown, no explanation):
"""

    response = await self.llm.generate(prompt, temperature=0.5, max_tokens=1000)

    # 提取代码
    implementation = self._extract_code_block(response)

    return f'    {method_signature}:\n{implementation}'
```

**效果预期**: 将实现完整度从 5-10% 提升至 60-80%

---

### 改进 3: API 错误处理与重试 ✅

**核心思路**: 指数退避 + 智能重试

```python
async def _api_call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
    """带智能重试的 API 调用"""

    retry_delay = 1  # 初始延迟 1 秒

    for attempt in range(max_retries):
        try:
            response = await self.llm.generate(prompt, temperature=0.3)
            return response

        except Exception as e:
            error_str = str(e).lower()

            # 判断错误类型
            if 'timeout' in error_str or 'connection' in error_str:
                # 网络错误：指数退避
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 1, 2, 4 秒
                    print(f"[Retry] API error, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue

            elif 'rate' in error_str:
                # 限流错误：等待更长时间
                if attempt < max_retries - 1:
                    wait_time = 60  # 等待 60 秒
                    print(f"[Retry] Rate limit hit, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

            # 其他错误或重试次数用尽
            raise

    # 所有重试失败
    raise Exception(f"API call failed after {max_retries} retries")

# 应用到决策过程
async def _autonomous_decision(self) -> Dict:
    """自主决策 - 带重试"""

    # ... 构造 prompt ...

    try:
        response = await self._api_call_with_retry(prompt, max_retries=3)
        decision = json.loads(self._extract_json(response))
        return decision

    except Exception as e:
        print(f"[Error] Decision failed after retries: {e}")

        # 进入反思模式，而不是盲目重试
        return {
            "thinking": "Decision system encountered persistent errors, need to reflect",
            "action": "reflect",
            "reasoning": "API errors detected, entering reflection mode to diagnose",
            "confidence": 0.5
        }
```

**效果预期**:
- API 错误恢复率从 0% 提升至 80%+
- 避免无限循环（如 Tick 250-366 的情况）

---

### 改进 4: 反馈循环优化 ✅

**核心思路**: 从失败中学习，避免重复错误

```python
async def _self_reflection(self) -> Dict:
    """自我反思 - 增强版"""

    # 统计错误模式
    error_patterns = {}

    for mem in self.memory[-10:]:
        result = mem.get('result', {})
        validation = result.get('validation', {})

        if not validation.get('all_valid', True):
            for file_path, file_info in validation.get('files', {}).items():
                if not file_info.get('valid', True):
                    error = file_info.get('error', '')

                    # 提取错误类型
                    if 'unterminated' in error:
                        error_type = 'unterminated_string'
                    elif 'indent' in error:
                        error_type = 'indentation'
                    elif 'syntax' in error:
                        error_type = 'syntax_error'
                    else:
                        error_type = 'other'

                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

    # 分析模式
    if error_patterns:
        most_common = max(error_patterns, key=error_patterns.get)
        count = error_patterns[most_common]

        print(f"[Reflection] Detected error pattern: {most_common} (occurred {count} times)")

        # 生成针对性修复建议
        fix_suggestion = await self._generate_fix_suggestion(most_common, error_patterns)

        return {
            "status": "issues_found",
            "error_patterns": error_patterns,
            "most_common": most_common,
            "fix_suggestion": fix_suggestion
        }
    else:
        return {
            "status": "no_issues"
        }

async def _generate_fix_suggestion(self, error_type: str, patterns: Dict) -> str:
    """生成针对性的修复建议"""

    prompt = f"""The AGI system has been generating code with recurring errors:

Error patterns detected:
{json.dumps(patterns, indent=2)}

Most common error type: {error_type}

Analyze the root cause and provide a specific solution to prevent this error in future code generation.
Consider:
1. Is it a prompt engineering issue?
2. Is it a token limit issue?
3. Is it a batch processing issue?

Provide a concrete fix strategy:
"""

    response = await self.llm.generate(prompt, temperature=0.3, max_tokens=500)
    return response
```

**效果预期**:
- 自动识别错误模式
- 避免重复相同错误
- 持续改进生成质量

---

## 实施计划

### Phase 1: 紧急修复 (2 小时)
- [ ] 实现 `_fix_syntax_errors` 方法
- [ ] 实现 `_fix_unterminated_string` 方法
- [ ] 集成到 `_create_project` 工作流

### Phase 2: API 稳定性 (3 小时)
- [ ] 实现 `_api_call_with_retry` 方法
- [ ] 添加指数退避逻辑
- [ ] 添加速率限制检测
- [ ] 停止当前无限循环的进程

### Phase 3: 完整实现 (8 小时)
- [ ] 修改 `_generate_method_implementation` 策略
- [ ] 添加实现度检查
- [ ] 优化批次大小
- [ ] 添加质量门控

### Phase 4: 反馈循环 (4 小时)
- [ ] 实现错误模式分析
- [ ] 实现自适应修复
- [ ] 添加学习机制

**总计**: 约 17 小时

---

## 预期效果对比

| 指标 | V6.0 当前 | V6.1 目标 | 改进 |
|------|----------|----------|------|
| 语法正确率 | 95.7% | 99%+ | +3.4% |
| 实现完整度 | 5-10% | 60-80% | +700% |
| API 错误恢复 | 0% | 80%+ | +∞ |
| 错误模式识别 | 无 | 有 | 新增 |
| 可运行性 | 2/10 | 6-7/10 | +300% |

---

## 实际应用价值

改进后的 V6.1 系统能够：

### 1. 自动生成可用代码
- 不再是占位符，而是实际可运行的实现
- 可以直接用于项目开发

### 2. 自我修复能力
- 自动检测并修复语法错误
- 从失败中学习，避免重复错误

### 3. 生产环境可用
- API 错误不会导致系统崩溃
- 稳定运行，持续生成

### 4. 实际应用场景
- ✅ 快速原型开发
- ✅ CRUD 代码生成
- ✅ 测试用例生成
- ✅ 文档自动生成
- ✅ 代码重构辅助

---

## 总结

**V6.0**: 优秀的 AGI 研究平台，展示了元认知能力
**V6.1**: 实用的代码生成工具，可以直接应用到实际开发

从"能演示"升级到"能使用"。
