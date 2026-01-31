# Release Notes - AGI Autonomous Core V6.1

## 🎉 重大更新

### 多基座模型支持

这是 AGI Autonomous Core 的一次重大更新，现在支持多个基座大模型：

- ✅ **DeepSeek V3** - 代码生成专家
- ✅ **智谱 GLM-4.7** - 中文任务专家
- ✅ **Moonshot Kimi 2.5** - 超长上下文
- ✅ **阿里千问 Qwen** - 平衡性能
- ✅ **Google Gemini 2.5** - 多模态能力

### 核心特性

- 🔀 **多实例并行** - 同时运行多个基座模型进行对比
- 📊 **性能对比工具** - 快速测试和对比不同模型
- 🚀 **批量生成策略** - 突破 API token 限制
- 📁 **多文件项目** - 自动生成完整的模块化项目
- 🤖 **自主决策** - 系统自主决定下一步行动
- 🔄 **自我反思** - 分析输出并持续改进

---

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/AGI_Autonomous_Core.git
cd AGI_Autonomous_Core

# 安装依赖
pip install -r requirements.txt

# 配置 API KEY
cp .env.multi_model .env
# 编辑 .env 添加你的 API KEY
```

---

## 🚀 快速开始

### 单模型运行

```bash
# 使用 DeepSeek
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek

# 使用智谱 GLM
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model zhipu
```

### 多实例对比

```bash
# 同时运行所有模型
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
```

---

## 📊 基座模型对比

| 模型 | 代码能力 | 上下文 | 成本 | 适用场景 |
|------|----------|--------|------|----------|
| DeepSeek V3 | ⭐⭐⭐⭐⭐ | 128K | ¥0.05/10K | 代码生成 |
| 智谱 GLM-4.7 | ⭐⭐⭐⭐ | 200K | ¥0.20/10K | 中文任务 |
| Kimi 2.5 | ⭐⭐⭐ | 256K | ¥0.22/10K | 长文档 |
| 千问 Qwen | ⭐⭐⭐⭐ | 128K | ¥0.10/10K | 通用 |
| Gemini 2.5 | ⭐⭐⭐⭐⭐ | 1M | $0.08/10K | 多模态 |

详细对比请查看 [MULTI_MODEL_SUMMARY.md](MULTI_MODEL_SUMMARY.md)

---

## 🎯 使用场景

1. **代码生成** - 使用 DeepSeek 生成高质量代码
2. **模型研究** - 对比不同基座模型的决策风格
3. **自主探索** - 让系统自主创建和改进项目
4. **AGI 研究** - 研究自主智能体的行为模式

---

## 📝 文档

- [README.md](README.md) - 项目主页
- [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) - 使用指南
- [MULTI_MODEL_SUMMARY.md](MULTI_MODEL_SUMMARY.md) - 完整总结
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [CHANGELOG.md](CHANGELOG.md) - 更新日志

---

## 🐛 已知问题

- 首次运行可能需要较长时间（生成 17 个模块约 90 分钟）
- 某些基座模型可能有 API 限流
- 生成的代码质量取决于所使用的基座模型

---

## 🔮 未来计划

### V6.2
- [ ] Web 界面
- [ ] 可视化生成过程
- [ ] 支持更多基座模型
- [ ] 性能优化

### V7.0
- [ ] 支持本地 LLM
- [ ] 分布式多实例协作
- [ ] 更智能的改进机制
- [ ] 项目模板系统

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📜 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

感谢以下开源项目和基座模型提供商：

- [DeepSeek](https://www.deepseek.com/)
- [智谱 AI](https://www.zhipuai.cn/)
- [Moonshot AI](https://www.moonshot.cn/)
- [阿里云千问](https://tongyi.aliyun.com/)
- [Google Gemini](https://ai.google.dev/)

---

**完整更新日志请查看 [CHANGELOG.md](CHANGELOG.md)**

---

**下载**: [AGI_Autonomous_Core_V6_1.zip](../../archive/refs/tags/v6.1.0/AGI_Autonomous_Core_V6.1.zip)

**Checksum**: SHA-256: [待计算]
