# 📦 AGI Autonomous Core V6.1 - 详细安装指南

> **一步步教你安装和配置 AGI 自主系统**

---

## 📑 目录

- [系统要求](#系统要求)
- [安装前准备](#安装前准备)
- [Windows 安装](#windows-安装)
- [macOS 安装](#macos-安装)
- [Linux 安装](#linux-安装)
- [配置 API KEY](#配置-api-key)
- [验证安装](#验证安装)
- [常见安装问题](#常见安装问题)
- [卸载说明](#卸载说明)

---

## 系统要求

### 最低配置

| 组件 | 要求 |
|------|------|
| **操作系统** | Windows 10+, macOS 10.14+, Ubuntu 18.04+ |
| **Python** | 3.8 或更高版本 |
| **内存** | 4GB RAM |
| **磁盘** | 500MB 可用空间 |
| **网络** | 宽带互联网连接 |

### 推荐配置

| 组件 | 要求 |
|------|------|
| **操作系统** | Windows 11, macOS 12+, Ubuntu 20.04+ |
| **Python** | 3.10 或更高版本 |
| **内存** | 8GB+ RAM |
| **磁盘** | 2GB+ 可用空间 |
| **网络** | 稳定的宽带连接 |

### Python 版本检查

打开命令行/终端，输入：

```bash
# Windows
python --version

# macOS/Linux
python3 --version
```

**期望输出**：
```
Python 3.8.0 或更高版本
```

**如果未安装 Python**：

#### Windows
1. 访问 https://www.python.org/downloads/
2. 下载最新的 Python 3.x 安装包
3. 运行安装程序
4. **重要**：勾选 "Add Python to PATH"
5. 点击 "Install Now"

#### macOS
```bash
# 使用 Homebrew（推荐）
brew install python@3.10

# 或从官网下载
# https://www.python.org/downloads/macos/
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

---

## 安装前准备

### 1. 检查网络连接

确保可以访问外网（需要访问 LLM API）：

```bash
# 测试连接
ping api.deepseek.com
ping open.bigmodel.cn
```

### 2. 检查磁盘空间

```bash
# Windows
dir

# macOS/Linux
df -h
```

确保至少有 500MB 可用空间。

### 3. 检查权限

确保有写入权限（需要创建虚拟环境和生成文件）。

---

## Windows 安装

### 步骤 1: 下载项目

#### 方法 A: 使用 Git（推荐）

1. 安装 Git：
   - 下载：https://git-scm.com/download/win
   - 运行安装程序，使用默认设置

2. 克隆项目：
   ```bash
   # 打开命令提示符或 PowerShell
   cd C:\Users\YourUsername\Desktop

   # 克隆仓库
   git clone https://github.com/yuzengbaao/-AGI-Autonomous-Core.git
   cd -AGI-Autonomous-Core
   ```

#### 方法 B: 下载 ZIP

1. 访问 https://github.com/yuzengbaao/-AGI-Autonomous-Core
2. 点击绿色的 "Code" 按钮
3. 选择 "Download ZIP"
4. 解压到 desired location
5. 打开解压目录

### 步骤 2: 创建虚拟环境

```bash
# 在项目目录中
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 验证激活（命令行前会显示 (venv)）
```

**如果激活失败**：

```bash
# PowerShell 可能需要更改执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后再试
venv\Scripts\activate
```

### 步骤 3: 升级 pip

```bash
# 确保已激活虚拟环境
python -m pip install --upgrade pip
```

### 步骤 4: 安装依赖

```bash
# 安装所需包
pip install -r requirements.txt
```

**如果安装失败**：

```bash
# 尝试使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 5: 验证安装

```bash
# 检查已安装的包
pip list

# 应该看到：
# openai         x.x.x
# python-dotenv  x.x.x
# aiohttp        x.x.x
```

---

## macOS 安装

### 步骤 1: 安装 Homebrew（推荐）

```bash
# 打开终端，运行：
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 步骤 2: 安装 Python

```bash
# 使用 Homebrew 安装
brew install python@3.10

# 验证安装
python3 --version
```

### 步骤 3: 下载项目

```bash
# 克隆项目
cd ~
git clone https://github.com/yuzengbaao/-AGI-Autonomous-Core.git
cd -AGI-Autonomous-Core
```

### 步骤 4: 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 验证激活（命令行前会显示 (venv)）
```

### 步骤 5: 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 步骤 6: 验证安装

```bash
# 检查已安装的包
pip list
```

---

## Linux 安装

### 步骤 1: 安装系统依赖

#### Ubuntu/Debian

```bash
# 更新包列表
sudo apt update

# 安装 Python 和相关工具
sudo apt install -y python3 python3-pip python3-venv git

# 验证安装
python3 --version
pip3 --version
```

#### CentOS/RHEL/Fedora

```bash
# 安装 Python 和相关工具
sudo dnf install -y python3 python3-pip python3-venv git

# 或使用 yum
sudo yum install -y python3 python3-pip python3-venv git
```

### 步骤 2: 下载项目

```bash
# 克隆项目
cd ~
git clone https://github.com/yuzengbaao/-AGI-Autonomous-Core.git
cd -AGI-Autonomous-Core
```

### 步骤 3: 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 验证激活
```

### 步骤 4: 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 步骤 5: 验证安装

```bash
# 检查已安装的包
pip list
```

---

## 配置 API KEY

### 步骤 1: 获取 API KEY

你需要至少一个 LLM 服务的 API KEY。

#### DeepSeek（推荐）

1. 访问 https://platform.deepseek.com/
2. 注册账号（使用手机号或邮箱）
3. 登录后，点击左侧菜单 "API Keys"
4. 点击 "Create new key"
5. 复制 API KEY（格式：sk-xxxxxxxxxxxx）
6. **重要**：妥善保管，不要泄露

#### 智谱 GLM

1. 访问 https://open.bigmodel.cn/
2. 注册/登录
3. 进入 "API 密钥" 页面
4. 创建新密钥

#### Moonshot Kimi

1. 访问 https://platform.moonshot.cn/
2. 注册/登录
3. 获取 API KEY

#### 阿里千问

1. 访问 https://dashscope.aliyuncs.com/
2. 注册/登录（需要阿里云账号）
3. 创建 API KEY

#### Google Gemini

1. 访问 https://ai.google.dev/
2. 注册/登录（需要 Google 账号）
3. 创建 API KEY

### 步骤 2: 配置环境变量

```bash
# 在项目根目录
# 复制配置模板
cp .env.multi_model .env
```

### 步骤 3: 编辑 .env 文件

#### Windows

```bash
# 使用记事本编辑
notepad .env

# 或使用 VS Code
code .env
```

#### macOS/Linux

```bash
# 使用 nano
nano .env

# 或使用 vim
vim .env

# 或使用 VS Code
code .env
```

### 步骤 4: 填写 API KEY

编辑 `.env` 文件，将 `your_xxx_api_key_here` 替换为真实的 API KEY：

```bash
# ================================
# DeepSeek (推荐用于代码生成)
# ================================
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat

# ================================
# 智谱 GLM (如果你有)
# ================================
# ZHIPU_API_KEY=your_zhipu_api_key_here
# ZHIPU_MODEL=glm-4-plus

# 其他模型类似...
```

**重要提示**：
- ✅ 至少配置一个 API KEY
- ✅ 不要添加空格或引号
- ✅ 保存文件后确保修改生效
- ❌ 不要将 `.env` 文件分享给他人
- ❌ 不要将 `.env` 上传到 GitHub

### 步骤 5: 验证配置

```bash
# 检查 .env 文件是否存在
ls -la .env

# 查看 .env 文件内容（确认 API KEY 已配置）
cat .env

# 确保没有多余的空格或引号
```

---

## 验证安装

### 测试 1: Python 环境

```bash
# 激活虚拟环境
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 测试 Python
python -c "print('Python 工作正常！')"

# 测试导入 openai
python -c "import openai; print('OpenAI 库已安装')"
```

**期望输出**：
```
Python 工作正常！
OpenAI 库已安装
```

### 测试 2: API 连接

```bash
# 运行快速测试
python compare_models.py
```

选择测试 `1`（代码生成测试）。

**如果看到以下内容，说明配置正确**：
```
==========================================
BASE MODEL COMPARISON TEST
==========================================

✅ Found 1 model(s) found:
   - deepseek

Select test to run:
...
```

**如果看到错误**：
- `DEEPSEEK_API_KEY not found` → 检查 `.env` 文件配置
- `Connection error` → 检查网络连接和 API KEY 是否正确

### 测试 3: 运行主系统

```bash
# 运行系统（使用 DeepSeek）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek
```

**期望输出**：
```
======================================================================
AGI AUTONOMOUS CORE V6.1 - DEEPSEEK
======================================================================
[Instance] ID: inst_deepseek_xxxxx
[Model] deepseek
[Init] Workspace: data/autonomous_outputs_v6_1/deepseek
[Init] Ready. Base model: deepseek
======================================================================

[Tick 1] HH:MM:SS
----------------------------------------------------------------------
[Decision] create_project: ...
```

按 `Ctrl+C` 可以停止运行。

---

## 常见安装问题

### 问题 1: Python 不是内部或外部命令

**原因**：Python 未安装或未添加到 PATH

**解决方案**：

**Windows**：
1. 重新安装 Python
2. **重要**：勾选 "Add Python to PATH"
3. 或手动添加到 PATH：
   - 打开 "系统属性" → "高级" → "环境变量"
   - 在 "Path" 中添加 Python 安装路径
   - 例如：`C:\Users\YourName\AppData\Local\Programs\Python\Python310`

### 问题 2: pip 不是最新版本

**解决方案**：
```bash
python -m pip install --upgrade pip
```

### 问题 3: 虚拟环境激活失败

**Windows PowerShell**：
```bash
# 更改执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 再试
venv\Scripts\activate
```

**Linux/macOS**：
```bash
# 确保有执行权限
chmod +x venv/bin/activate

# 激活
source venv/bin/activate
```

### 问题 4: 依赖安装失败

**解决方案**：

```bash
# 清除缓存
pip cache purge

# 使用镜像源（中国用户）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题 5: API KEY 无效

**可能原因**：
- API KEY 格式错误
- API KEY 已过期
- API KEY 输入错误（多余空格或引号）

**解决方案**：
1. 重新复制 API KEY
2. 确保 `.env` 文件中：
   - 没有引号：`DEEPSEEK_API_KEY=sk-xxx` ✓
   - 不是：`DEEPSEEK_API_KEY="sk-xxx"` ✗
3. 检查是否有多余空格

### 问题 6: 网络连接错误

**解决方案**：

```bash
# 测试 API 连接
curl https://api.deepseek.com/v1

# 如果失败，检查：
# 1. 网络连接
# 2. 防火墙设置
# 3. 代理配置（如果使用代理）
```

### 问题 7: 权限错误（Linux/macOS）

**解决方案**：

```bash
# 确保有写权限
chmod +w .
chmod +w data/

# 或使用 sudo（不推荐）
sudo pip install -r requirements.txt
```

---

## 卸载说明

### Windows

```bash
# 1. 停止所有运行的实例
# 按 Ctrl+C

# 2. 退出虚拟环境
deactivate

# 3. 删除项目目录
rmdir /s -q -AGI-Autonomous-Core

# 4. 删除虚拟环境（如果在其他位置）
rmdir /s -q venv
```

### macOS/Linux

```bash
# 1. 停止所有运行的实例
# 按 Ctrl+C

# 2. 退出虚拟环境
deactivate

# 3. 删除项目目录
rm -rf -AGI-Autonomous-Core

# 4. 删除虚拟环境（如果在其他位置）
rm -rf venv
```

### 完全清理（包括生成的数据）

```bash
# Windows
rmdir /s -q data

# macOS/Linux
rm -rf data
```

---

## 📞 需要帮助？

如果安装过程中遇到问题：

1. **查看日志**：检查错误信息
2. **查看文档**：阅读 [USER_GUIDE.md](USER_GUIDE.md)
3. **提交 Issue**：https://github.com/yuzengbaao/-AGI-Autonomous-Core/issues
4. **查看常见问题**：[FAQ](USER_GUIDE.md#常见问题)

---

## 🎉 安装完成

安装完成后，你可以：

1. **阅读使用指南**：[USER_GUIDE.md](USER_GUIDE.md)
2. **运行第一个项目**：`python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek`
3. **对比不同模型**：`python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all`

**祝你使用愉快！🚀**
