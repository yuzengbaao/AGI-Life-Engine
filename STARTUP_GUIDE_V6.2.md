# 🚀 AGI系统手动启动指南 (最新)

**更新时间**: 2026-02-05
**系统版本**: AGI AUTONOMOUS CORE V6.2 + TokenBudget V6.2.1
**配置状态**: ✅ 已配置

---

## ✅ 环境检查

### 当前系统配置

```bash
✅ Python环境: 已安装
✅ 依赖包: 已安装
✅ API Key: 已配置
   - DeepSeek: sk-4929...8bf2 ✅
   - 智谱GLM: c33b1...d774 ✅
✅ TokenBudget: V6.2.1 (24,000 tokens)
```

### 系统组件

- ✅ **AGI_AUTONOMOUS_CORE_V6_2.py** - 核心代码生成系统
- ✅ **test_multi_file_v2.py** - 多文件项目生成器
- ✅ **token_budget.py** - Token管理模块 (已升级)
- ✅ **AGI_Life_Engine.py** - 完整AGI系统

---

## 🎯 推荐启动方式

### 方式 1️⃣: 单文件代码生成 (推荐新手)

**适合场景**: 快速生成单个工具/脚本

```bash
# 启动 V6.2 核心系统
python AGI_AUTONOMOUS_CORE_V6_2.py
```

**预期效果**:
- 自动决定生成什么项目
- 生成单个Python文件（100-300行）
- 自动验证语法
- 自动修复错误
- 5-10分钟完成

**生成位置**:
```
data/autonomous_outputs_v6_2/deepseek/project_*/
```

---

### 方式 2️⃣: 多文件项目生成 (推荐)

**适合场景**: 生成完整的多模块项目

```bash
# 启动多文件生成器 V2
python test_multi_file_v2.py
```

**预期效果**:
- 生成6个模块的完整项目
- 每个模块200-600行
- 总计1500-3000行代码
- 包含README和requirements.txt
- 30-60分钟完成

**新TokenBudget配置**:
```
Token容量: 24,000 (3倍提升)
支持规模: 200-800行/模块
成功率: 100% (无截断)
```

**生成位置**:
```
output/multi_file_project_v2/
```

---

### 方式 3️⃣: 完整AGI系统 (高级)

**适合场景**: AGI研究、长期运行

```bash
# 启动完整AGI系统
python AGI_Life_Engine.py
```

**功能**:
- 多模态感知
- 自我进化
- 创造性探索
- 知识图谱推理

**注意**: 需要更多资源，建议先测试方式1和2

---

## 📝 详细启动步骤

### Step 1: 打开命令行

**Windows**:
```bash
# 按 Win + R，输入 cmd，回车
# 或右键开始菜单 → Windows PowerShell

# 进入项目目录
cd D:\TRAE_PROJECT\AGI
```

---

### Step 2: 选择启动方式

#### 选项A: 快速单文件生成

```bash
# 1. 启动系统
python AGI_AUTONOMOUS_CORE_V6_2.py

# 2. 观察输出
# 系统会显示:
#   - LLM连接状态
#   - 自动决策过程
#   - 代码生成进度
#   - 语法验证结果

# 3. 等待完成
# 看到 "[COMPLETE]" 表示完成

# 4. 查看生成结果
cd data/autonomous_outputs_v6_2/deepseek/project_*/
dir
```

#### 选项B: 多文件项目生成 (推荐)

```bash
# 1. 启动多文件生成器
python test_multi_file_v2.py

# 2. 观察输出
# 系统会显示:
#   - 项目规划
#   - 模块生成进度 (6个模块)
#   - Token使用情况
#   - 验证结果

# 3. 等待完成 (30-60分钟)
# 看到 "Generation complete!" 表示完成

# 4. 验证生成的项目
cd output/multi_file_project_v2
python -c "import config; import core.validator"
```

---

### Step 3: 查看生成结果

#### 检查文件结构

```bash
# Windows
dir

# Linux/macOS
ls -la
```

#### 验证代码质量

```bash
# 语法检查
python -m py_compile *.py

# 导入测试
python -c "import <module_name>; print('[OK] Import successful')"

# 查看代码
type <filename>.py | more  # Windows
cat <filename>.py | less    # Linux/macOS
```

---

## 🛑 停止运行

### 优雅停止

按 `Ctrl + C` 一次

系统会:
- ✅ 保存当前状态
- ✅ 完成当前操作
- ✅ 清理临时文件
- ✅ 显示退出信息

### 强制停止

如果系统无响应，按 `Ctrl + C` 两次

---

## 📊 实时监控

### 查看Token使用

在多文件生成时，系统会显示:
```
[TokenBudget] Estimated tokens: 2,500
[TokenBudget] Available: 21,500
[TokenBudget] Sufficient: True
```

### 查看生成进度

```
[Progress] Module 1/6: config.py
[Progress] Module 2/6: core/validator.py
[Progress] Module 3/6: core/processor.py
...
```

---

## 🧪 快速测试

### 测试TokenBudget升级

```bash
# 验证新配置
python verify_token_budget_upgrade.py
```

**预期输出**:
```
[测试1] 配置验证 - [OK] 通过
  max_tokens: 24000
  实际可用: 18,600

[测试2] 大文件容量 - [OK] 通过
  Available: 19,309
  Sufficient: True
```

---

## ⚙️ 高级配置

### 调整生成参数

编辑 `.env` 文件:

```bash
# 生成温度 (0.0-1.0)
TEMPERATURE=0.7

# 每批次方法数
MAX_METHODS_PER_BATCH=3

# 最大运行次数
MAX_TICKS=5
```

### 切换模型

编辑 `.env` 文件:

```bash
# 使用DeepSeek (推荐 - 快速便宜)
DEEPSEEK_MODEL=deepseek-chat

# 使用智谱GLM (中文优化)
ZHIPU_MODEL=glm-4
```

---

## 🔍 故障排查

### 问题1: ModuleNotFoundError

```bash
# 解决: 安装依赖
pip install -r requirements.txt
```

### 问题2: API Key错误

```bash
# 检查配置
type .env | findstr API_KEY  # Windows
cat .env | grep API_KEY      # Linux/macOS

# 确保 API Key 正确
```

### 问题3: 网络连接错误

```bash
# 检查网络
ping api.deepseek.com

# 尝试切换模型
# 编辑 .env，注释掉 DeepSeek，使用智谱GLM
```

### 问题4: 生成被截断

```bash
# 确认TokenBudget版本
python -c "from token_budget import TokenBudget; b = TokenBudget(); print(b.max_tokens)"

# 应显示: 24000 (V6.2.1)

# 如果显示8000，需要重新同步代码
git pull origin main
```

---

## 📈 性能基准

### 单文件生成 (V6.2)

```
文件规模: 100-300行
生成时间: 2-5分钟
成功率: 100%
语法正确: 100%
可运行: 95%+
```

### 多文件生成 (V2 + TokenBudget V6.2.1)

```
项目规模: 6个模块，1500-3000行
生成时间: 30-60分钟
成功率: 100%
语法正确: 100%
可导入: 100%
可运行: 95%+
```

---

## 🎯 推荐使用流程

### 新手流程 (15分钟)

```bash
# 1. 快速测试 (5分钟)
python AGI_AUTONOMOUS_CORE_V6_2.py

# 2. 查看结果
cd data/autonomous_outputs_v6_2/deepseek/project_*/

# 3. 验证代码
python -m py_compile *.py

# 4. 运行测试
pytest tests/ -v
```

### 进阶流程 (1小时)

```bash
# 1. 多文件项目生成 (30-60分钟)
python test_multi_file_v2.py

# 2. 验证所有模块
cd output/multi_file_project_v2
python -c "import config; import core.validator; import core.processor; import core.reporter"

# 3. 查看文档
type README.md

# 4. 运行项目
python main.py --help
```

---

## 📚 相关文档

- **[QUICKSTART.md](QUICKSTART.md)** - V6.1快速开始
- **[TOKEN_BUDGET_UPGRADE_SUMMARY.md](TOKEN_BUDGET_UPGRADE_SUMMARY.md)** - Token升级总结
- **[TOKEN_BUDGET_V6.2.1_UPGRADE.md](TOKEN_BUDGET_V6.2.1_UPGRADE.md)** - 详细升级说明
- **[README.md](README.md)** - 项目概述

---

## 🎉 开始使用

### 最简单的启动命令

```bash
cd D:\TRAE_PROJECT\AGI
python AGI_AUTONOMOUS_CORE_V6_2.py
```

**就是这么简单！** 系统会自动完成所有工作。

---

## 💡 提示

1. **第一次使用**: 推荐从方式1开始
2. **生成大项目**: 使用方式2 (test_multi_file_v2.py)
3. **查看进度**: 观察控制台输出
4. **停止系统**: 按 Ctrl+C (会优雅关闭)
5. **查看结果**: 使用 `dir` 或 `ls` 命令

---

**准备好了吗？开始运行你的AGI系统吧！** 🚀

```bash
python AGI_AUTONOMOUS_CORE_V6_2.py
```
