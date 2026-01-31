#!/bin/bash
# GitHub 上传前最终检查脚本

echo "=========================================="
echo "GitHub Upload - Pre-Upload Check"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# 1. 检查敏感文件
echo "1. Checking for sensitive files..."
SENSITIVE_FILES=$(git ls-files 2>/dev/null | grep -E "^\.(env|env\.local|env\.backup|env\.production)" || true)

if [ -n "$SENSITIVE_FILES" ]; then
    echo "❌ ERROR: Sensitive files found:"
    echo "$SENSITIVE_FILES"
    echo ""
    echo "Run: git rm --cached $SENSITIVE_FILES"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ No sensitive files tracked"
fi
echo ""

# 2. 检查 .gitignore
echo "2. Checking .gitignore..."
if [ -f ".gitignore" ]; then
    if grep -q "\.env$" .gitignore; then
        echo "✅ .env is in .gitignore"
    else
        echo "⚠️  WARNING: .env not in .gitignore"
        WARNINGS=$((WARNINGS + 1))
    fi
    
    if grep -q "data/" .gitignore; then
        echo "✅ data/ is in .gitignore"
    else
        echo "⚠️  WARNING: data/ not in .gitignore"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "❌ ERROR: .gitignore not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 3. 检查真实 API KEY
echo "3. Checking for real API keys in code..."
API_KEYS=$(grep -r "sk-" --exclude-dir=.git --exclude-dir=data --include="*.py" . 2>/dev/null | head -5 || true)

if [ -n "$API_KEYS" ]; then
    echo "❌ ERROR: Possible API keys found:"
    echo "$API_KEYS"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ No API keys found in code"
fi
echo ""

# 4. 检查必要文件
echo "4. Checking required files..."
REQUIRED_FILES=(
    "README_GITHUB.md"
    "LICENSE"
    ".gitignore"
    "requirements.txt"
    "AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file not found"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# 5. 检查文档
echo "5. Checking documentation..."
DOC_FILES=(
    "MULTI_MODEL_GUIDE.md"
    "MULTI_MODEL_SUMMARY.md"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
)

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "⚠️  WARNING: $file not found (optional)"
        WARNINGS=$((WARNINGS + 1))
    fi
done
echo ""

# 6. 清理检查
echo "6. Checking for cache files..."
PYCACHE=$(find . -type d -name __pycache__ 2>/dev/null | wc -l)
PYC=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)

if [ "$PYCACHE" -gt 0 ]; then
    echo "⚠️  WARNING: Found $PYCACHE __pycache__ directories"
    echo "   Run: find . -type d -name __pycache__ -exec rm -rf {} +"
    WARNINGS=$((WARNINGS + 1))
fi

if [ "$PYC" -gt 0 ]; then
    echo "⚠️  WARNING: Found $PYC .pyc files"
    echo "   Run: find . -type f -name '*.pyc' -delete"
    WARNINGS=$((WARNINGS + 1))
fi

if [ "$PYCACHE" -eq 0 ] && [ "$PYC" -eq 0 ]; then
    echo "✅ No cache files found"
fi
echo ""

# 7. 检查 Git 状态
echo "7. Git status..."
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "✅ Git repository initialized"
    
    # 检查是否有未提交的更改
    if git diff --quiet && git diff --cached --quiet; then
        echo "✅ No uncommitted changes"
    else
        echo "⚠️  WARNING: Uncommitted changes found"
        git status --short
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "⚠️  WARNING: Git not initialized"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# 总结
echo "=========================================="
echo "Check Summary"
echo "=========================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo "❌ CRITICAL: Found $ERRORS error(s) that must be fixed!"
    echo ""
    echo "Please fix the errors before uploading."
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo "⚠️  WARNING: Found $WARNINGS warning(s)"
    echo ""
    echo "You can proceed, but consider fixing warnings."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ All checks passed! Ready to upload."
    echo ""
    echo "Next steps:"
    echo "1. Run: bash upload_to_github.sh"
    echo "   or manually:"
    echo "2. git remote add origin https://github.com/YOUR_USERNAME/AGI_Autonomous_Core.git"
    echo "3. git push -u origin main"
fi

exit 0
