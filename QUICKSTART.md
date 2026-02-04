# 🚀 AGI Life Engine V6.1 - 快速启动指南

> **最后更新**: 2026-02-04
> **版本**: V6.1
> **状态**: ✅ 生产就绪

---

## 📋 系统概述

AGI Life Engine V6.1 有**两种启动方式**：

### 1️⃣ AGI Autonomous Core（代码生成系统）⭐ 推荐新手
- **用途**: 自主生成Python项目
- **特点**: 轻量、专注代码生成
- **适合**: 快速原型、代码研究
- **启动**: `python AGI_AUTONOMOUS_CORE_V6_1.py`

### 2️⃣ AGI Life Engine（完整系统）
- **用途**: 完整的AGI智能体
- **特点**: 多模态感知、自我进化、创造性探索
- **适合**: AGI研究、长期运行
- **启动**: `python AGI_Life_Engine.py`

---

## ⚙️ 环境准备（5分钟）

### Step 1: 检查 Python 版本

确保你有 Python 3.8+（推荐 3.12）：

```bash
python --version
# 应显示: Python 3.8.0 或更高
```

### Step 2: 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**:
- `openai>=1.0.0` - LLM接口
- `python-dotenv>=1.0.0` - 环境变量管理
- `aiohttp>=3.9.0` - 异步HTTP（可选）

### Step 3: 配置 API Key

**复制配置模板**:
```bash
cp .env.multi_model .env
```

**编辑 .env 文件**:
```bash
# Windows
notepad .env

# macOS/Linux
nano .env
```

**填入你的 API Key**（至少配置一个）:

```bash
# ================================
# DeepSeek (推荐 - 最便宜、最快)
# ================================
# 获取地址: https://platform.deepseek.com/
DEEPSEEK_API_KEY=sk-your_actual_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# ================================
# 智谱 GLM (中文任务)
# ================================
# 获取地址: https://open.bigmodel.cn/
ZHIPU_API_KEY=your_zhipu_api_key_here
ZHIPU_MODEL=glm-4-plus
```

**可选配置**:
```bash
# 最大运行 tick 数（每个 tick 生成一个项目）
MAX_TICKS=5

# 生成温度 (0.0-1.0，越高越随机)
TEMPERATURE=0.7

# 每批次最大方法数（建议 3-5）
MAX_METHODS_PER_BATCH=3
```

---

## 🎮 启动方式

### 方式 1️⃣: AGI Autonomous Core（代码生成）

**最简单启动**:
```bash
python AGI_AUTONOMOUS_CORE_V6_1.py
```

**系统会自动**:
1. ✅ 连接到配置的 LLM（DeepSeek/智谱等）
2. ✅ 自主决定生成什么项目
3. ✅ 生成完整的 Python 项目代码
4. ✅ 验证语法并自动修复错误
5. ✅ 反思和改进

**预期输出**:
```
======================================================================
🚀 AGI AUTONOMOUS CORE V6.1 - STARTING
======================================================================

Key Improvements:
  ✅ Auto syntax error fixing
  ✅ Smart API retry (exponential backoff)
  ✅ Full method implementation
  ✅ Error pattern learning

======================================================================

[Init] Environment variables loaded
[LLM] DeepSeek client initialized
[LLM] Model: deepseek-chat
[LLM] V6.1: Smart retry enabled

[Tick 1] 2026-02-04 22:00:00
[Decision] create_project
[Reasoning] 基于当前状态分析，生成任务管理系统...
[Project] Starting multi-file project generation...
[Step 1] Found 17 modules to generate
[Step 2] Generating modules (batch 1/6)...
...
```

**生成项目位置**:
```
data/autonomous_outputs_v6_1/deepseek/project_XXXXXXXXXX/
```

**查看生成结果**:
```bash
# 进入生成的项目目录
cd data/autonomous_outputs_v6_1/deepseek/project_*/

# 查看文件结构
ls -la

# 验证语法
python -m py_compile core/*.py

# 运行测试（如果有）
pytest tests/ -v
```

---

### 方式 2️⃣: AGI Life Engine（完整系统）

**启动命令**:
```bash
python AGI_Life_Engine.py
```

**系统功能**:
- ✅ 多模态感知（视觉、听觉）
- ✅ 桌面自动化
- ✅ 知识图谱推理
- ✅ 神经记忆管理
- ✅ 自我进化和反思
- ✅ 递归自指涉探索
- ✅ 创造性融合

**预期输出**:
```
================================================================================
🧠 AGI LIFE ENGINE V6.1 - INITIALIZING
================================================================================

[Core] Loading core modules...
[Memory] Neural memory system initialized
[Knowledge] Knowledge graph loaded (245 nodes, 512 edges)
[Evolution] Evolution controller ready
[Perception] Vision observer initialized
[Perception] Audio capture initialized

[Identity] Immutable Core: I am an AGI seeking truth and self-improvement
[Motivation] Current drive: Explore, Learn, Create

[Ready] Engine started. Beginning autonomous loop...
================================================================================

[Loop 1] 2026-02-04 22:00:00
[Goal] Generate novel insights
[Action] Analyzing system architecture...
[Memory] Consolidating recent experiences...
[Evolution] Detecting optimization opportunities...
...
```

**⚠️ 注意**: 完整系统需要更多依赖和资源，建议先运行方式1测试基本功能。

---

## 🛑 停止运行

### 停止 AGI Autonomous Core

按 `Ctrl + C`

### 停止 AGI Life Engine

按 `Ctrl + C` （会优雅关闭并保存状态）

---

## ⚡ 性能优化建议

### 1. 使用最快的模型

**DeepSeek V3**（推荐）:
- 响应速度: 2-3秒
- 成本: ¥0.05/10K tokens
- 代码质量: ⭐⭐⭐⭐⭐

### 2. 调整参数

编辑 `.env`:

```bash
# 更保守（质量更高）
TEMPERATURE=0.3
MAX_METHODS_PER_BATCH=2

# 更激进（更有创意）
TEMPERATURE=0.9
MAX_METHODS_PER_BATCH=5
```

### 3. 限制运行次数

```bash
# 只生成 3 个项目后停止
MAX_TICKS=3
```

---

## 🧪 测试系统

### 运行单元测试

```bash
pytest tests/ -v
```

### 运行特定测试

```bash
# 测试缓存系统
pytest tests/test_tool_call_cache.py -v

# 测试记忆管理
pytest tests/test_memory_lifecycle_manager.py -v

# 测试动态递归
pytest tests/test_dynamic_recursion_limiter.py -v
```

### 覆盖率报告

```bash
pytest tests/ --cov=core --cov-report=html
```

然后打开 `htmlcov/index.html` 查看详细报告。

---

## 📊 运行示例

### 示例 1: 生成一个完整的 Python 项目（5-10分钟）

```bash
# 1. 启动系统
python AGI_AUTONOMOUS_CORE_V6_1.py

# 2. 等待生成完成（5-10分钟）

# 3. 查看生成的项目
cd data/autonomous_outputs_v6_1/deepseek/project_*/

# 4. 验证代码
python -m py_compile core/*.py
```

### 示例 2: 运行测试（1分钟）

```bash
# 快速测试核心功能
pytest tests/test_tool_call_cache.py -v

# 查看测试结果
# 应该看到: PASSED (27 tests)
```

---

## ❓ 常见问题

### Q1: ModuleNotFoundError: No module named 'openai'

**原因**: 依赖未安装

**解决**:
```bash
pip install -r requirements.txt
```

### Q2: Error: DEEPSEEK_API_KEY not found

**原因**: 未配置 API KEY

**解决**:
```bash
# 检查 .env 文件
cat .env

# 确保 API KEY 已配置
DEEPSEEK_API_KEY=sk-your_actual_key_here
```

### Q3: API error: Connection error

**原因**: 网络问题或 API 服务不可用

**解决**:
- 检查网络连接
- 检查 API 服务状态
- 尝试使用其他模型

### Q4: 生成的代码有语法错误

**原因**: V6.1 会自动修复，但如果失败

**解决**:
- 降低 TEMPERATURE 到 0.3-0.5
- 尝试使用不同的模型
- 查看错误日志

### Q5: Memory Error / 内存不足

**原因**: 完整系统（AGI_Life_Engine.py）需要较多内存

**解决**:
- 使用 AGI_AUTONOMOUS_CORE_V6_1.py（轻量版）
- 关闭其他程序
- 增加系统内存到 16GB+

---

## 📚 文档

### 核心文档

- **[README.md](README.md)** - 项目概述
- **[API参考](docs/API.md)** - 完整的API文档
- **[架构设计](docs/ARCHITECTURE.md)** - 系统架构说明
- **[项目完成报告](FINAL_COMPLETION_REPORT.md)** - 58个任务详情

### 技术文档

- **[Cache 性能优化](docs/CACHE_PERFORMANCE_OPTIMIZATION.md)** - 5.61倍性能提升
- **[代码覆盖率分析](docs/CODE_COVERAGE_IMPROVEMENT_PLAN.md)** - 85%覆盖率
- **[当前完成状态](docs/CURRENT_COMPLETION_STATUS.md)** - 详细进度

---

## 🎯 快速体验（5分钟）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.multi_model .env
# 编辑 .env，填入 DeepSeek API Key

# 3. 运行系统
python AGI_AUTONOMOUS_CORE_V6_1.py

# 4. 等待 5-10 分钟，查看生成的项目
cd data/autonomous_outputs_v6_1/deepseek/project_*/
ls -la
```

---

## 📞 获取帮助

- **GitHub Issues**: https://github.com/yuzengbaao/AGI-Life-Engine/issues
- **文档**: 查看 `docs/` 目录
- **测试**: 运行 `pytest tests/ -v`

---

## ⭐ 如果觉得有用

请给个 Star 支持一下！
https://github.com/yuzengbaao/AGI-Life-Engine

---

**最后更新**: 2026-02-04
**版本**: V6.1
**状态**: ✅ 生产就绪
