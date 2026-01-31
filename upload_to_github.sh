#!/bin/bash
# AGI Autonomous Core - GitHub Upload Script
# 快速上传脚本

echo "=========================================="
echo "AGI Autonomous Core - GitHub Upload"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py" ]; then
    echo "❌ ERROR: Not in AGI project directory"
    exit 1
fi

# 1. 检查敏感文件
echo "1. Checking for sensitive files..."
if git ls-files 2>/dev/null | grep -q "^\.env$"; then
    echo "❌ ERROR: .env file is staged!"
    echo "   Run: git reset .env"
    exit 1
fi

if [ -f ".env" ] && [ ! -f ".env.gitignore" ]; then
    echo "⚠️  WARNING: .env file exists"
    echo "   Make sure it's in .gitignore"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "✅ Sensitive files check passed"
echo ""

# 2. 清理缓存
echo "2. Cleaning cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "✅ Cache cleaned"
echo ""

# 3. 显示状态
echo "3. Current git status:"
git status --short
echo ""

# 4. 确认
read -p "Continue with upload? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 1
fi

# 5. 初始化 git（如果需要）
if [ ! -d ".git" ]; then
    echo "4. Initializing git repository..."
    git init
    git branch -M main
fi

# 6. 检查远程仓库
if ! git remote get-url origin &>/dev/null; then
    echo "5. Setting up remote repository..."
    echo "Please enter your GitHub username:"
    read -p "Username: " username
    git remote add origin "https://github.com/${username}/AGI_Autonomous_Core.git"
    echo "✅ Remote added: origin"
fi

# 7. 添加文件
echo "6. Staging files..."
git add .

# 8. 提交
echo "7. Committing changes..."
git commit -m "Initial commit: AGI Autonomous Core V6.1

Features:
- Multi-base model support (DeepSeek, Zhipu, Kimi, Qwen, Gemini)
- Autonomous code generation system
- Multi-file project generation
- Batch generation strategy
- Comprehensive documentation

For more information, see README.md"

# 9. 推送
echo "8. Pushing to GitHub..."
echo "This may require authentication..."
git push -u origin main

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps:"
echo "1. Visit your GitHub repository"
echo "2. Verify files are uploaded correctly"
echo "3. Update repository description and topics"
echo "4. Create a Release (optional)"
