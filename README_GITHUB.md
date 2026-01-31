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

### 📖 详细文档

| 文档 | 说明 |
|------|------|
| **[安装指南](INSTALL.md)** | 📦 详细的安装步骤，支持 Windows/macOS/Linux |
| **[使用说明](USER_GUIDE.md)** | 📖 完整的使用教程，包含快速开始和进阶使用 |
| **[技术指南](MULTI_MODEL_GUIDE.md)** | 🔧 多基座模型技术对比和配置 |
| **[贡献指南](CONTRIBUTING.md)** | 🤝 如何贡献代码 |

### ⚡ 5 分钟快速安装

#### 1. 克隆项目

```bash
git clone https://github.com/yuzengbaao/-AGI-Autonomous-Core.git
cd -AGI-Autonomous-Core
```

#### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 3. 配置 API KEY

```bash
# 复制配置模板
cp .env.multi_model .env

# 编辑 .env，添加你的 API KEY
# Windows: notepad .env
# macOS/Linux: nano .env

# 至少配置一个：
DEEPSEEK_API_KEY=sk-your_actual_api_key_here
```

**📍 获取 API KEY**：
- [DeepSeek](https://platform.deepseek.com/) - 推荐，代码生成强
- [智谱 GLM](https://open.bigmodel.cn/) - 中文任务好
- [Moonshot Kimi](https://platform.moonshot.cn/) - 超长上下文
- [阿里千问](https://dashscope.aliyuncs.com/) - 平衡性能
- [Google Gemini](https://ai.google.dev/) - 多模态

#### 4. 运行系统

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

## 📚 文档中心

### 📘 用户文档

| 文档 | 内容 | 适用对象 |
|------|------|----------|
| **[安装指南](INSTALL.md)** | Windows/macOS/Linux 详细安装步骤 | 所有用户 |
| **[使用说明](USER_GUIDE.md)** | 完整使用教程，包含快速开始、详细教程、常见问题 | 所有用户 |
| **[快速开始](#-5-分钟快速安装)** | 5 分钟快速安装 | 有经验用户 |
| **[FAQ](USER_GUIDE.md#常见问题)** | 常见问题解答 | 遇到问题的用户 |

### 🔧 技术文档

| 文档 | 内容 | 适用对象 |
|------|------|----------|
| **[多模型指南](MULTI_MODEL_GUIDE.md)** | 多基座模型对比和配置 | 开发者/研究者 |
| **[技术总结](MULTI_MODEL_SUMMARY.md)** | 完整技术总结和架构说明 | 开发者/研究者 |
| **[贡献指南](CONTRIBUTING.md)** | 如何贡献代码 | 贡献者 |
| **[更新日志](CHANGELOG.md)** | 版本历史和更新内容 | 所有用户 |

### 📞 获取帮助

- **安装问题** → 查看 [安装指南 - 常见安装问题](INSTALL.md#常见安装问题)
- **使用问题** → 查看 [使用说明 - 常见问题](USER_GUIDE.md#常见问题)
- **Bug 反馈** → [提交 Issue](https://github.com/yuzengbaao/-AGI-Autonomous-Core/issues)

---

## 🔗 相关链接

- [GitHub 仓库](https://github.com/yuzengbaao/-AGI-Autonomous-Core)
- [DeepSeek 官网](https://www.deepseek.com/)
- [智谱 AI 官网](https://www.zhipuai.cn/)
- [Moonshot AI 官网](https://www.moonshot.cn/)
- [阿里云千问](https://tongyi.aliyun.com/)
- [Google Gemini](https://ai.google.dev/)

---

**Made with ❤️ for AGI Research**
