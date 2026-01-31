# AGI 自主系统 V6.1 - 多基座模型使用指南

## 🎯 快速开始

### 1. 配置 API KEY

```bash
# 复制配置文件模板
cp .env.multi_model .env

# 编辑 .env，填入你的 API KEY
# 例如：
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxxxxxx
KIMI_API_KEY=sk-xxxxxxxxxxxx
QWEN_API_KEY=sk-xxxxxxxxxxxx
GEMINI_API_KEY=AIzaxxxxxxxxxxxx
```

### 2. 运行单个实例

```bash
# 使用 DeepSeek（推荐，代码能力强）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek

# 使用智谱 GLM（稳健型）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model zhipu

# 使用 Kimi（超长上下文）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model kimi

# 使用千问 Qwen（平衡性能）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model qwen

# 使用 Gemini（多模态）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model gemini
```

### 3. 运行多实例对比

```bash
# 同时运行所有已配置的模型实例
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
```

---

## 📊 基座模型对比

| 特性 | DeepSeek | 智谱 GLM | Kimi | 千问 Qwen | Gemini |
|------|----------|----------|------|-----------|--------|
| **代码能力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **中文能力** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **上下文长度** | 128K | 200K | 256K | 128K | 1M |
| **输出限制** | 8K/64K | 128K | 262K | 8K | 8K |
| **响应速度** | 快 | 中 | 慢 | 快 | 中 |
| **成本/10K tokens** | ¥0.05 | ¥0.20 | ¥0.22 | ¥0.10 | ¥0.55 |
| **适用场景** | 代码生成 | 中文任务 | 长文档 | 通用 | 多模态 |

---

## 🎯 使用建议

### 场景 1: 生成代码项目
```
推荐: DeepSeek V3
原因:
- 代码生成能力最强
- 理解代码逻辑好
- 成本最低

命令:
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek
```

### 场景 2: 生成中文文档
```
推荐: 智谱 GLM-4.7 或 Kimi
原因:
- 中文理解能力强
- 文档写作流畅
- 适合说明性文本

命令:
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model zhipu
```

### 场景 3: 探索性研究
```
推荐: Kimi 或 Gemini
原因:
- 超长上下文，可以记住更多历史
- 创造性更强
- 可能产生意想不到的方案

命令:
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model kimi
```

### 场景 4: 性能对比实验
```
推荐: 运行多实例对比
原因:
- 同时运行多个基座模型
- 对比生成质量和决策风格
- 找出最适合你的模型

命令:
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
```

---

## 📁 输出目录结构

```
data/autonomous_outputs_v6_1/
├── deepseek/
│   ├── project_1769862020/
│   │   ├── core/
│   │   ├── ai_engine/
│   │   └── project_metadata.json
│   └── project_1769862500/
│       └── ...
├── zhipu/
│   └── project_1769862021/
│       └── ...
├── kimi/
│   └── ...
├── qwen/
│   └── ...
└── gemini/
    └── ...
```

---

## 🔍 观察基座模型差异

### 1. 决策风格差异

```bash
# 运行多实例后，查看每个实例的第一个决策
grep -r "Decision:" data/autonomous_outputs_v6_1/*/

# DeepSeek 可能更倾向于：
# "生成代码框架项目"

# Kimi 可能更倾向于：
# "生成游戏或创意项目"

# GLM 可能更倾向于：
# "先生成再反思"
```

### 2. 代码质量差异

```bash
# 对比生成的代码质量
python -m py_compile data/autonomous_outputs_v6_1/deepseek/project_*/core/*.py
python -m py_compile data/autonomous_outputs_v6_1/zhipu/project_*/core/*.py

# 统计代码行数
find data/autonomous_outputs_v6_1/deepseek -name "*.py" | xargs wc -l
find data/autonomous_outputs_v6_1/zhipu -name "*.py" | xargs wc -l
```

### 3. 成本对比

```bash
# 查看每个实例的 token 使用情况
grep -r "total_tokens" data/autonomous_outputs_v6_1/*/project_*/project_metadata.json
```

---

## ⚙️ 高级配置

### 调整生成参数

编辑 `.env` 文件：

```bash
# 更随机/更有创意的生成
TEMPERATURE=0.9

# 更确定/更一致的生成
TEMPERATURE=0.3

# 每批处理更多方法（更快但可能质量下降）
MAX_METHODS_PER_BATCH=5

# 运行更多 tick（生成更多项目）
MAX_TICKS=10
```

### 自定义项目描述

如果你想测试特定的生成任务，可以修改代码中的 `project_description`。

---

## 🐛 常见问题

### Q: 为什么某个模型没有运行？
A: 检查 `.env` 文件中是否配置了对应的 `API_KEY`。

### Q: 如何停止运行？
A: `Ctrl+C` 或关闭终端窗口。

### Q: 生成的代码有语法错误怎么办？
A: 这是正常的，系统会不断改进。可以等待下一次 tick 的改进。

### Q: 如何对比不同模型的效果？
A: 运行 `--model all`，然后查看每个实例的 `project_metadata.json`。

---

## 📈 性能指标说明

### 查看生成统计

```bash
# DeepSeek 统计
cat data/autonomous_outputs_v6_1/deepseek/project_*/project_metadata.json | jq '.stats'

# 对比所有模型
for model in deepseek zhipu kimi qwen gemini; do
    echo "=== $model ==="
    cat data/autonomous_outputs_v6_1/$model/project_*/project_metadata.json | jq '.stats'
done
```

### 关键指标

- `files_generated`: 生成的文件数
- `total_methods`: 总方法数
- `total_batches`: 总批次数
- `total_tokens`: 估算的 token 使用量
- `duration`: 生成耗时（秒）

---

## 🎯 实验建议

### 实验 1: 同一任务，不同模型

```bash
# 所有模型生成相同类型的项目
# 观察：哪个模型生成的代码质量最高
```

### 实验 2: 长期运行对比

```bash
# 每个模型运行 50+ ticks
# 观察：哪个模型的改进能力最强
# 方法：修改代码中的 max_ticks=50
```

### 实验 3: 决策模式分析

```bash
# 记录每个模型的所有决策
# 分析：是否存在明显的决策风格差异
# 方法：grep "Decision:" 输出文件
```

---

## 💡 下一步

1. **配置 API KEY**: 编辑 `.env` 文件
2. **运行单模型测试**: `python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek`
3. **观察生成结果**: 查看 `data/autonomous_outputs_v6_1/deepseek/`
4. **运行多实例对比**: `python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all`
5. **分析差异**: 对比不同模型的生成质量和决策风格

祝你实验愉快！🚀
