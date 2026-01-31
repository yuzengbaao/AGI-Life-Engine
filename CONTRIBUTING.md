# 贡献指南

感谢你对 AGI Autonomous Core 项目的关注！

## 🤝 如何贡献

### 报告问题

如果你发现了 bug 或者有功能建议：

1. 检查 [Issues](../../issues) 是否已有相关问题
2. 如果没有，创建新的 Issue，详细描述：
   - 问题和复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（Python 版本、操作系统等）

### 提交代码

1. **Fork 仓库**
   ```bash
   # 在 GitHub 上点击 Fork 按钮
   ```

2. **克隆到本地**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AGI_Autonomous_Core.git
   cd AGI_Autonomous_Core
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **进行更改**
   - 遵循现有代码风格
   - 添加必要的注释和文档
   - 确保代码通过测试

5. **提交更改**
   ```bash
   git add .
   git commit -m "Add some feature"
   ```

6. **推送到 GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**
   - 在 GitHub 上打开 Pull Request
   - 详细描述你的更改
   - 等待代码审查

## 📝 代码规范

### Python 代码风格

- 遵循 PEP 8 规范
- 使用类型注解
- 添加文档字符串
- 保持函数简洁

```python
async def generate_code(
    self,
    prompt: str,
    max_tokens: int = 8000
) -> str:
    """
    Generate code based on the given prompt.

    Args:
        prompt: The input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Generated code as string
    """
    # Implementation
    pass
```

### 测试

- 为新功能添加测试
- 确保所有测试通过
- 测试覆盖率 > 80%

```bash
# 运行测试
pytest tests/

# 检查覆盖率
pytest --cov=src tests/
```

## 🎯 贡献方向

我们欢迎以下方向的贡献：

### 核心功能
- [ ] 支持更多基座模型（Llama, Claude, 等）
- [ ] 优化批量生成策略
- [ ] 改进自我反思机制
- [ ] 增强错误处理和恢复

### 工具和文档
- [ ] 改进文档
- [ ] 添加更多使用示例
- [ ] 创建可视化界面
- [ ] 性能分析和优化

### 测试和质量
- [ ] 增加单元测试
- [ ] 集成测试
- [ ] 性能基准测试

## 📧 联系方式

如有任何问题，请通过以下方式联系：

- GitHub Issues: [创建 Issue](../../issues)
- Email: your-email@example.com

## 📜 许可证

贡献的代码将采用相同的 [MIT 许可证](LICENSE)。

---

再次感谢你的贡献！🎉
