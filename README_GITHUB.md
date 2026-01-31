# 🤖 AGI Autonomous Core - Multi-Base Model Edition

> **自主 AGI 系统 - 多基座模型支持版本**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![AGI](https://img.shields.io/badge/AGI-Autonomous-orange.svg)]()

---

## 📋 项目简介

**AGI Autonomous Core** 是一个实验性的自主智能体系统，能够自主决策、生成项目、自我反思和改进。

### 🎯 核心特性

- ✅ **完全自主运行**：无需人工干预，系统自主决策和行动
- ✅ **多基座模型支持**：支持 DeepSeek、智谱 GLM、Kimi、千问 Qwen、Gemini
- ✅ **多文件项目生成**：自动生成完整的 Python 项目，包含多个模块
- ✅ **批量代码生成**：突破 API token 限制，支持大型项目
- ✅ **自我反思机制**：分析自己的输出并持续改进
- ✅ **多实例对比**：同时运行不同基座模型，对比决策风格和生成质量

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install openai python-dotenv asyncio
```

### 2. 配置 API KEY

```bash
# 复制配置模板
cp .env.multi_model .env

# 编辑 .env，添加你的 API KEY
# 至少配置一个：
DEEPSEEK_API_KEY=your_deepseek_api_key
ZHIPU_API_KEY=your_zhipu_api_key
KIMI_API_KEY=your_kimi_api_key
QWEN_API_KEY=your_qwen_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 3. 运行系统

#### 方式 1: 运行单个基座模型

```bash
# 使用 DeepSeek（推荐，代码生成能力强）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek

# 使用智谱 GLM
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model zhipu

# 使用 Kimi
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model kimi

# 使用千问 Qwen
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model qwen

# 使用 Gemini
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model gemini
```

#### 方式 2: 运行多实例对比

```bash
# 同时运行所有已配置的模型
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
```

#### 方式 3: Windows 快速启动

```bash
START_MULTI_MODEL.bat
```

---

## 📊 基座模型对比

| 模型 | 代码能力 | 上下文 | 成本 | 适用场景 |
|------|----------|--------|------|----------|
| **DeepSeek V3** | ⭐⭐⭐⭐⭐ | 128K | ¥0.05/10K | 代码生成、逻辑推理 |
| **智谱 GLM-4.7** | ⭐⭐⭐⭐ | 200K | ¥0.20/10K | 中文任务、稳健型 |
| **Kimi 2.5** | ⭐⭐⭐ | 256K | ¥0.22/10K | 超长文档、创意探索 |
| **千问 Qwen** | ⭐⭐⭐⭐ | 128K | ¥0.10/10K | 通用任务、平衡性能 |
| **Gemini 2.5** | ⭐⭐⭐⭐⭐ | 1M | $0.08/10K | 多模态、复杂推理 |

### 决策风格差异

- **DeepSeek**: 逻辑推理型，倾向于系统性技术项目
- **Kimi**: 创造探索型，倾向于实验性、游戏类项目
- **智谱 GLM**: 稳健保守型，倾向于反思和改进
- **千问 Qwen**: 平衡实用型，倾向于实用工具
- **Gemini**: 多模态创新型，倾向于可视化、复杂系统

---

## 📁 项目结构

```
AGI_AUTONOMOUS_CORE_V6_1/
├── AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py  # 主系统（多基座支持）
├── compare_models.py                        # 快速对比测试
├── START_MULTI_MODEL.bat                    # Windows 快速启动
├── .env.multi_model                         # 配置模板
├── MULTI_MODEL_GUIDE.md                     # 详细使用指南
├── MULTI_MODEL_SUMMARY.md                   # 完整总结
├── README_GITHUB.md                         # 本文件
├── requirements.txt                         # 依赖列表
├── LICENSE                                  # MIT 许可证
└── data/
    └── autonomous_outputs_v6_1/             # 生成输出
        ├── deepseek/
        ├── zhipu/
        ├── kimi/
        ├── qwen/
        └── gemini/
```

---

## 🎯 使用场景

### 场景 1: 代码生成

```bash
# 使用 DeepSeek 生成代码项目
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek
```

### 场景 2: 基座模型对比研究

```bash
# 运行所有模型，对比决策风格和生成质量
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all

# 查看决策差异
grep -h "Decision:" data/autonomous_outputs_v6_1/*/project_*/generation_result.json
```

### 场景 3: 快速测试

```bash
# 测试所有模型的基本功能和性能
python compare_models.py
```

---

## 🔧 系统架构

### 核心组件

```
┌─────────────────────────────────────────────────────────┐
│              AutonomousAGI_V6_1                        │
│                  (主控系统)                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ BaseLLM     │  │ MultiModel   │  │ Memory       │  │
│  │ (多基座      │  │ BatchGen     │  │ (经验记忆)    │  │
│  │  抽象层)     │  │ (批量生成)    │  │              │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  自主循环：                                              │
│  决策 → 行动 → 观察 → 反思 → 改进 → 决策...            │
└─────────────────────────────────────────────────────────┘
```

### 工作流程

1. **初始化**: 加载基座模型配置
2. **自主决策**: 根据当前状态决定下一步行动
3. **项目生成**: 解析项目结构，批量生成代码
4. **质量验证**: 语法检查和结构验证
5. **结果保存**: 生成元数据和统计信息
6. **反思改进**: 分析历史，优化策略

---

## 📈 版本历史

### V6.1 (当前版本)
- ✨ 新增多基座模型支持
- ✨ 多实例并行运行
- ✨ 基座模型对比功能
- 🐛 修复批量生成截断问题
- 📝 完整文档和使用指南

### V6.0
- ✨ 多文件项目生成
- ✨ 独立模块生成
- ✨ 自动目录结构创建

### V5.0
- ✨ 批量代码生成
- ⚠️ 仅支持单文件

### V3.5 - V4.0
- 🎯 早期版本，代码截断问题

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

---

## ⚠️ 免责声明

本项目仅用于教育和研究目的。生成的代码质量取决于所使用的基座模型。请自行评估生成代码的适用性。

---

## 📞 联系方式

- 项目主页: [GitHub Repository](#)
- 问题反馈: [GitHub Issues](#)

---

## 🙏 致谢

感谢以下开源项目和基座模型提供商：

- [DeepSeek](https://www.deepseek.com/)
- [智谱 AI](https://www.zhipuai.cn/)
- [Moonshot AI](https://www.moonshot.cn/)
- [阿里云千问](https://tongyi.aliyun.com/)
- [Google Gemini](https://ai.google.dev/)

---

## 📚 相关资源

- [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) - 详细使用指南
- [MULTI_MODEL_SUMMARY.md](MULTI_MODEL_SUMMARY.md) - 完整总结
- [.env.multi_model](.env.multi_model) - 配置模板

---

**Made with ❤️ for AGI Research**
