# GitHub 上传检查清单

## ✅ 上传前检查

### 1. 敏感信息检查 ⚠️ **非常重要**

```bash
# 检查是否包含真实 API KEY
grep -r "sk-" --exclude-dir=data --exclude-dir=.git --exclude-dir=__pycache__
grep -r "API_KEY" --exclude-dir=data --exclude-dir=.git --exclude-dir=__pycache__

# 检查 .env 文件
ls -la | grep "\.env$"

# 确保 .env 在 .gitignore 中
grep "\.env" .gitignore
```

**必须确认：**
- ✅ `.env` 文件不在上传列表中
- ✅ `.gitignore` 已正确配置
- ✅ 没有真实 API KEY 在代码中
- ✅ 只上传 `.env.multi_model` 和 `.env.example` 作为模板

---

### 2. 文件清理

```bash
# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# 清理生成的数据（保留示例）
# data/autonomous_outputs_v6_1/ 会被 .gitignore 忽略
```

---

### 3. Git 初始化

```bash
# 初始化 Git 仓库
git init

# 添加所有文件
git add .

# 检查将要提交的文件
git status

# 确认没有敏感文件
git status | grep ".env"
# 如果看到 .env，执行：git reset .env
```

---

### 4. 首次提交

```bash
# 创建首次提交
git commit -m "Initial commit: AGI Autonomous Core V6.1

- Multi-base model support (DeepSeek, Zhipu, Kimi, Qwen, Gemini)
- Autonomous code generation system
- Multi-file project generation
- Batch generation strategy
- Comprehensive documentation"
```

---

### 5. GitHub 仓库创建

#### 在 GitHub 网站创建仓库：

1. 访问 https://github.com/new
2. 仓库名称：`AGI_Autonomous_Core`
3. 描述：`Autonomous AGI System with Multi-Base Model Support`
4. 可见性：**Public** 或 **Private**（根据你的需求）
5. **不要**初始化 README、.gitignore 或 LICENSE（我们已经有了）
6. 点击 "Create repository"

---

### 6. 连接本地仓库到 GitHub

```bash
# 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/AGI_Autonomous_Core.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

---

### 7. 验证上传

访问你的 GitHub 仓库，确认：

- ✅ README.md 显示正常
- ✅ LICENSE 文件存在
- ✅ 没有敏感文件（.env）
- ✅ 代码文件完整
- ✅ 文档文件齐全

---

## 📁 推荐上传的文件

### 核心系统文件
```
✅ AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py
✅ compare_models.py
✅ START_MULTI_MODEL.bat
```

### 配置文件
```
✅ .env.multi_model       （配置模板）
✅ .env.example           （示例配置）
✅ .gitignore             （Git 忽略规则）
✅ requirements.txt       （依赖列表）
```

### 文档文件
```
✅ README_GITHUB.md       （项目主页，重命名为 README.md）
✅ MULTI_MODEL_GUIDE.md
✅ MULTI_MODEL_SUMMARY.md
✅ CONTRIBUTING.md
✅ CHANGELOG.md
✅ LICENSE
```

### 可选：历史版本
```
✅ AGI_AUTONOMOUS_CORE_V6_0.py
✅ AGI_AUTONOMOUS_CORE_V5_0.py
...
```

---

## ❌ 不要上传的文件

### 敏感文件
```
❌ .env                    （包含真实 API KEY）
❌ .env.local
❌ .env.backup
```

### 生成的数据
```
❌ data/autonomous_outputs_v3_5/
❌ data/autonomous_outputs_v4_0/
❌ data/autonomous_outputs_v5_0/
❌ data/autonomous_outputs_v6_0/
❌ data/autonomous_outputs_v6_1/
```

### 缓存和临时文件
```
❌ __pycache__/
❌ *.pyc
❌ *.pyo
❌ .pytest_cache/
❌ .mypy_cache/
```

### IDE 配置
```
❌ .vscode/
❌ .idea/
```

### 虚拟环境
```
❌ venv/
❌ env/
❌ .venv/
```

---

## 🎯 快速上传脚本

创建文件 `upload_to_github.sh`：

```bash
#!/bin/bash

echo "=========================================="
echo "AGI Autonomous Core - GitHub Upload Script"
echo "=========================================="
echo ""

# 1. 检查敏感文件
echo "1. Checking for sensitive files..."
if git ls-files | grep -q "\.env$"; then
    echo "❌ ERROR: .env file is staged! Remove it first:"
    echo "   git reset .env"
    exit 1
fi
echo "✅ No sensitive files found"
echo ""

# 2. 清理缓存
echo "2. Cleaning cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Cache cleaned"
echo ""

# 3. 显示即将上传的文件
echo "3. Files to be uploaded:"
git status --short
echo ""

# 4. 确认
read -p "Continue with upload? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 1
fi

# 5. 提交和推送
echo "4. Committing changes..."
git add .
git commit -m "Update: AGI Autonomous Core V6.1

- Multi-base model support
- Comprehensive documentation
- Ready for GitHub release"

echo "5. Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Upload complete!"
echo "Visit: https://github.com/YOUR_USERNAME/AGI_Autonomous_Core"
```

---

## 🔄 上传后操作

### 1. 重命名 README（如果需要）

```bash
# 在 GitHub 仓库设置中，或者在本地：
git mv README_GITHUB.md README.md
git commit -m "Rename README_GITHUB.md to README.md"
git push
```

### 2. 添加 GitHub Topics

在仓库设置中添加标签：
- `agi`
- `autonomous-agent`
- `code-generation`
- `multi-model`
- `deepseek`
- `llm`
- `python`

### 3. 设置仓库描述

```
Autonomous AGI System with Multi-Base Model Support -
An experimental autonomous agent that can self-decide, generate projects, and improve iteratively.
```

### 4. 创建 Releases（可选）

- 在 GitHub 上创建 Release v6.1.0
- 添加发布说明
- 标记为 Pre-release 或 Latest

---

## 🎉 完成确认

上传完成后，你的仓库应该包含：

```
AGI_Autonomous_Core/
├── README.md              ✅ 项目主页
├── LICENSE                ✅ MIT 许可证
├── CHANGELOG.md           ✅ 更新日志
├── CONTRIBUTING.md        ✅ 贡献指南
├── .gitignore             ✅ Git 配置
├── requirements.txt       ✅ 依赖列表
├── AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py
├── compare_models.py
├── START_MULTI_MODEL.bat
├── .env.multi_model       ✅ 配置模板
├── MULTI_MODEL_GUIDE.md
└── MULTI_MODEL_SUMMARY.md
```

---

## 📞 需要帮助？

如果遇到问题，检查：

1. **Git 配置**：`git config --list`
2. **远程仓库**：`git remote -v`
3. **分支状态**：`git branch -a`
4. **日志**：`git log --oneline`

---

**祝你上传顺利！🚀**
