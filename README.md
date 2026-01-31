🤖 AGI Autonomous Core V6.1 - 完整使用指南
自主 AGI 系统 - 多基座模型支持版本

一个实验性的自主智能体系统，能够自主决策、生成代码、自我反思和持续改进

📑 目录
项目简介
功能特性
系统要求
详细安装步骤
配置说明
快速开始
详细使用教程
基座模型对比
常见问题
故障排除
进阶使用
开发指南
项目简介
AGI Autonomous Core 是一个基于 Python 的自主智能体系统，具有以下核心能力：

🤖 自主决策：系统根据当前状态自主决定下一步行动
💻 代码生成：自动生成完整的多模块 Python 项目
🔄 自我反思：分析自己的输出并持续改进
🌐 多模型支持：支持 DeepSeek、智谱 GLM、Kimi、千问、Gemini
应用场景
代码生成研究：研究 LLM 的代码生成能力
AGI 行为观察：观察自主智能体的决策模式
模型对比：对比不同基座模型的性能和风格
自动化开发：自主生成项目代码
功能特性
核心功能
✅ 完全自主运行

无需人工干预，系统自主决策和行动
24/7 持续运行，无需用户在场
自主生成项目、反思、改进
✅ 多基座模型支持

DeepSeek V3（代码生成专家）
智谱 GLM-4.7（中文任务专家）
Moonshot Kimi 2.5（超长上下文）
阿里千问 Qwen（平衡性能）
Google Gemini 2.5（多模态能力）
✅ 多文件项目生成

自动生成完整的 Python 项目
支持多模块、多文件
自动创建目录结构
完整的依赖管理
✅ 批量代码生成

突破 API token 限制
分批实现方法（3个/批）
支持任意大小的项目
✅ 自我反思机制

分析生成代码质量
识别问题并改进
持续优化生成策略
系统要求
必需环境
项目	要求
操作系统	Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
Python 版本	Python 3.8 或更高（推荐 3.10+）
内存	至少 4GB RAM（推荐 8GB+）
磁盘空间	至少 500MB 可用空间
网络	稳定的互联网连接（访问 LLM API）
Python 版本检查
# 检查 Python 版本
python --version
# 或
python3 --version

# 应该显示：Python 3.8.0 或更高
详细安装步骤
步骤 1: 克隆或下载项目
方法 A: 使用 Git 克隆（推荐）
# 克隆仓库
git clone https://github.com/yuzengbaao/-AGI-Autonomous-Core.git
cd -AGI-Autonomous-Core
方法 B: 下载 ZIP 文件
访问 https://github.com/yuzengbaao/-AGI-Autonomous-Core
点击绿色的 "Code" 按钮
选择 "Download ZIP"
解压到本地目录
打开命令行/终端，进入解压目录
步骤 2: 创建虚拟环境（强烈推荐）
Windows
# 使用 venv 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 验证激活成功（命令行前会显示 (venv)）
macOS/Linux
# 使用 venv 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 验证激活成功（命令行前会显示 (venv)）
为什么使用虚拟环境？

✅ 隔离项目依赖
✅ 避免污染系统 Python
✅ 不同项目可以使用不同版本的包
✅ 易于管理和卸载
步骤 3: 安装依赖
# 确保已激活虚拟环境
# 安装所需包
pip install -r requirements.txt
requirements.txt 内容：

openai>=1.0.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
如果安装失败：

# 升级 pip
python -m pip install --upgrade pip

# 然后重新安装依赖
pip install -r requirements.txt
步骤 4: 配置 API KEY
4.1 获取 API KEY
你需要至少一个 LLM 服务的 API KEY：

DeepSeek（推荐）

访问 https://platform.deepseek.com/
注册账号
进入 "API Keys" 页面
创建新的 API KEY
复制保存（格式：sk-xxxxxxxxxxxx）
智谱 GLM

访问 https://open.bigmodel.cn/
注册账号
获取 API KEY
Moonshot Kimi

访问 https://platform.moonshot.cn/
注册账号
获取 API KEY
阿里千问

访问 https://dashscope.aliyuncs.com/
注册账号
获取 API KEY
Google Gemini

访问 https://ai.google.dev/
注册账号
获取 API KEY
4.2 配置环境变量
# 复制配置模板
cp .env.multi_model .env

# 编辑 .env 文件
# Windows 记事本：
notepad .env

# macOS/Linux 使用 nano：
nano .env
.env 文件示例：

# ================================
# DeepSeek (推荐用于代码生成)
# ================================
DEEPSEEK_API_KEY=sk-your_actual_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# ================================
# 智谱 GLM (稳健型，适合中文)
# ================================
ZHIPU_API_KEY=your_zhipu_api_key_here
ZHIPU_MODEL=glm-4-plus

# ================================
# Moonshot Kimi (超长上下文)
# ================================
KIMI_API_KEY=sk-your_kimi_api_key_here
KIMI_MODEL=moonshot-v1-128k

# ================================
# 阿里千问 Qwen (平衡性能)
# ================================
QWEN_API_KEY=sk-your_qwen_api_key_here
QWEN_MODEL=qwen-plus

# ================================
# Google Gemini (多模态能力)
# ================================
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
重要提示：

⚠️ 将 your_xxx_api_key_here 替换为真实的 API KEY
⚠️ 不要将 .env 文件分享给他人或上传到 GitHub
⚠️ 至少配置一个 API KEY 才能运行
✅ 可以同时配置多个，系统会自动选择
4.3 验证配置
# 检查 .env 文件是否存在
ls -la .env

# 查看 .env 文件内容（确保 API KEY 已配置）
cat .env
步骤 5: 验证安装
# 运行快速测试
python compare_models.py
如果看到以下输出，说明安装成功：

==========================================
BASE MODEL COMPARISON TEST
==========================================

[Available] 1 model(s) found:
   - deepseek

Select test to run:
1. Code Generation (Python merge function)
2. Architecture Design (URL shortener)
3. Creative Task (AI todo app ideas)
4. All tests

Enter choice (1-4):
按 Ctrl+C 退出测试。

配置说明
高级配置选项
编辑 .env 文件，可以添加以下配置：

# 生成温度 (0.0-1.0)
TEMPERATURE=0.7
# - 0.0: 更确定，输出一致
# - 0.7: 平衡（推荐）
# - 1.0: 更随机，更有创意

# 每批最大方法数（建议 3-5）
MAX_METHODS_PER_BATCH=3
# - 3: 稳定，质量高（推荐）
# - 5: 更快，但可能质量下降

# 最大运行 tick 数
MAX_TICKS=5
# - 系统会生成 N 个项目后自动停止
# - 不设置则无限运行
基座模型选择策略
根据你的需求选择模型：

需求	推荐模型	原因
代码生成	DeepSeek V3	代码能力最强，成本最低
中文文档	智谱 GLM 或 Kimi	中文理解好，表达流畅
长文本处理	Kimi 2.5	256K 上下文，最长
快速原型	千问 Qwen	响应快，平衡性能
多模态	Gemini 2.5	支持图像、视频
成本敏感	DeepSeek V3	最便宜（¥0.05/10K）
创意探索	Kimi 或 Gemini	创造性强
快速开始
最简单的方式（Windows）
# 双击运行批处理文件
START_MULTI_MODEL.bat
按照屏幕提示选择模型和运行模式。

命令行方式
1. 运行单个模型
# 使用 DeepSeek（推荐）
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek

# 使用智谱 GLM
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model zhipu

# 使用 Kimi
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model kimi

# 使用千问 Qwen
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model qwen

# 使用 Gemini
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model gemini
2. 运行多实例对比
# 同时运行所有已配置的模型
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
预期输出
======================================================================
AGI AUTONOMOUS CORE V6.1 - DEEPSEEK
======================================================================
[Instance] ID: inst_deepseek_1769862020
[Model] deepseek
[Init] Workspace: data/autonomous_outputs_v6_1/deepseek
[Init] Ready. Base model: deepseek
======================================================================

[Tick 1] 20:20:06
----------------------------------------------------------------------
[Decision] create_project: As a new AGI system...
[Project] Starting multi-file project generation...
[Step 1] Found 17 modules to generate
[Step 2] Generating modules...
...
详细使用教程
教程 1: 生成你的第一个项目
目标：使用 DeepSeek 生成一个完整的 Python 项目

步骤：

准备环境

# 激活虚拟环境
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
启动系统

python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek
观察运行

系统会自动决定生成什么项目
显示每个模块的生成进度
显示批次和方法数
查看结果

# 生成的项目在：
data/autonomous_outputs_v6_1/deepseek/project_XXXXXXXXXX/

# 查看文件结构
ls -la data/autonomous_outputs_v6_1/deepseek/project_*/
测试生成的代码

# 验证语法
python -m py_compile data/autonomous_outputs_v6_1/deepseek/project_*/core/*.py
教程 2: 对比不同基座模型
目标：对比 DeepSeek 和智谱 GLM 的生成差异

步骤：

运行对比

python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
查看结果

# DeepSeek 生成目录
data/autonomous_outputs_v6_1/deepseek/

# 智谱 GLM 生成目录
data/autonomous_outputs_v6_1/zhipu/
对比差异

# 查看决策差异
grep -h "Decision:" data/autonomous_outputs_v6_1/*/project_*/generation_result.json

# 对比代码行数
find data/autonomous_outputs_v6_1/deepseek -name "*.py" | xargs wc -l
find data/autonomous_outputs_v6_1/zhipu -name "*.py" | xargs wc -l
教程 3: 快速性能测试
目标：测试所有模型的基本功能

步骤：

运行测试脚本

python compare_models.py
选择测试类型

1. Code Generation (Python merge function)
2. Architecture Design (URL shortener)
3. Creative Task (AI todo app ideas)
4. All tests
查看结果

测试会显示每个模型的响应时间
显示响应长度
保存对比结果到 data/model_comparison_*.json
基座模型对比
详细对比表
特性	DeepSeek V3	智谱 GLM-4.7	Kimi 2.5	千问 Qwen	Gemini 2.5
代码能力	⭐⭐⭐⭐⭐	⭐⭐⭐⭐	⭐⭐⭐	⭐⭐⭐⭐	⭐⭐⭐⭐⭐
中文能力	⭐⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	⭐⭐⭐⭐
上下文长度	128K	200K	256K	128K	1M
输出限制	8K/64K	128K	262K	8K	8K
响应速度	快（2-3秒）	中（3-5秒）	慢（4-6秒）	快（2-4秒）	中（3-5秒）
成本/10K tokens	¥0.05	¥0.20	¥0.22	¥0.10	¥0.55
决策风格	逻辑推理型	稳健保守型	创造探索型	平衡实用型	多模态创新
典型项目	系统性技术项目	稳健实用项目	实验性创意项目	实用工具	可视化系统
决策风格示例
DeepSeek V3（逻辑推理型）

Decision: create_project
Reasoning: "基于当前状态分析，生成任务管理系统可以验证
          系统的代码生成能力，并为后续反思提供素材。
          这个项目包含完整的 CRUD 操作、AI 分析和 CLI 接口..."
Kimi 2.5（创造探索型）

Decision: create_project
Reasoning: "我想探索一些有趣的实验！生成一个游戏引擎怎么样？
          或者一个自动化交易系统？或者用 AI 写诗的工具？
          让我们尝试一些创新的东西..."
智谱 GLM-4.7（稳健保守型）

Decision: reflect
Reasoning: "已经生成了 3 个项目，应该先分析质量，总结经验，
          然后再继续生成。这样才能确保持续改进..."
常见问题
Q1: 如何获取 API KEY？
DeepSeek:

访问 https://platform.deepseek.com/
注册/登录账号
进入 "API Keys" 页面
点击 "Create new key"
复制 KEY（格式：sk-xxxxx）
智谱 GLM:

访问 https://open.bigmodel.cn/
注册/登录
进入 "API 密钥" 页面
创建新密钥
其他模型类似，访问各自的开放平台即可获取。

Q2: 为什么生成速度这么慢？
正常情况下：

单个模块生成时间：5-10 分钟
17 个模块总时间：~90 分钟
受限于 API 响应速度和网络状况
加速方法：

使用 DeepSeek（最快）
减少项目复杂度（减少模块数）
检查网络连接
Q3: 生成的代码有语法错误怎么办？
这是正常现象，系统会不断改进。如果问题严重：

检查 API KEY 是否正确
尝试使用不同的基座模型
调整 TEMPERATURE 参数（降低到 0.3-0.5）
Q4: 如何停止运行？
# 按 Ctrl+C
# 或关闭终端窗口
Q5: 可以同时运行多个实例吗？
可以！每个实例使用独立的输出目录：

# 终端 1
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model deepseek &

# 终端 2
python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model zhipu &
或使用：

python AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py --model all
Q6: 生成的代码在哪里？
# DeepSeek
data/autonomous_outputs_v6_1/deepseek/project_XXXXXXXXXX/

# 智谱 GLM
data/autonomous_outputs_v6_1/zhipu/project_YYYYYYYYYY/

# 其他模型类似
Q7: 如何查看生成统计？
# 查看项目元数据
cat data/autonomous_outputs_v6_1/deepseek/project_*/project_metadata.json

# 查看生成结果
cat data/autonomous_outputs_v6_1/deepseek/project_*/generation_result.json
故障排除
问题 1: ModuleNotFoundError: No module named 'openai'
原因：依赖未安装

解决：

# 激活虚拟环境
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install -r requirements.txt
问题 2: Error: DEEPSEEK_API_KEY not found
原因：未配置 API KEY

解决：

# 检查 .env 文件是否存在
ls -la .env

# 如果不存在，复制模板
cp .env.multi_model .env

# 编辑 .env，添加 API KEY
notepad .env  # Windows
nano .env     # macOS/Linux
问题 3: API error: Connection error
原因：网络问题或 API 服务不可用

解决：

检查网络连接
检查 API 服务状态
尝试使用其他模型
检查防火墙设置
问题 4: Permission denied (script)
原因：脚本没有执行权限（Linux/macOS）

解决：

chmod +x upload_to_github.sh
chmod +x pre_upload_check.sh
问题 5: 生成的代码无法运行
原因：多种可能

解决：

检查语法错误
python -m py_compile path/to/file.py
查看错误日志
尝试使用不同的基座模型
调整 TEMPERATURE 参数
进阶使用
自定义项目描述
编辑代码中的 project_description：

# 在 AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py 中
# 找到 _autonomous_decision 方法
# 修改默认的 project_description

return {
    "action": "create_project",
    "reasoning": "Starting with a new project",
    "project_description": """
    Generate a complete Python package:
    1) core/data_processor.py – data processing utilities
    2) core/analyzer.py – data analysis
    3) api/server.py – REST API server
    4) web/dashboard.py – web dashboard
    Include tests and documentation.
    """
}
调整生成参数
编辑 .env：

# 更保守的生成（质量更高）
TEMPERATURE=0.3
MAX_METHODS_PER_BATCH=2

# 更激进的生成（更有创意）
TEMPERATURE=0.9
MAX_METHODS_PER_BATCH=5
持续运行模式
# 修改代码中的 max_ticks 参数
# 或在 .env 中设置
MAX_TICKS=100  # 生成 100 个项目后停止
监控运行状态
# 实时查看日志
tail -f data/autonomous_outputs_v6_1/*/project_*/generation_result.json

# 统计生成的文件数
find data/autonomous_outputs_v6_1 -name "*.py" | wc -l

# 查看磁盘使用
du -sh data/autonomous_outputs_v6_1/
开发指南
项目结构
AGI-Autonomous-Core/
├── AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py  # 主系统
├── compare_models.py                        # 对比工具
├── START_MULTI_MODEL.bat                    # 快速启动
├── .env.multi_model                         # 配置模板
├── requirements.txt                         # 依赖列表
├── README_GITHUB.md                         # 项目说明
├── USER_GUIDE.md                            # 本文件
├── MULTI_MODEL_GUIDE.md                     # 技术指南
├── CONTRIBUTING.md                          # 贡献指南
└── data/                                    # 生成输出
    └── autonomous_outputs_v6_1/
        ├── deepseek/
        ├── zhipu/
        ├── kimi/
        ├── qwen/
        └── gemini/
核心类说明
BaseLLM

基座模型抽象类
支持多种 LLM provider
MultiModelBatchGenerator

多文件批量生成器
分批实现方法
自动目录结构创建
AutonomousAGI_V6_1

主控系统
自主决策循环
记忆和反思
扩展新的基座模型
# 1. 在 BaseModel 枚举中添加
class BaseModel(Enum):
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    YOUR_MODEL = "your_model"  # 添加这里

# 2. 在 BaseLLM 中添加初始化方法
def _init_your_model(self):
    """Initialize your model"""
    try:
        import openai
        api_key = os.getenv("YOUR_MODEL_API_KEY")
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://your-api-url.com/v1"
        )
        self.model = os.getenv("YOUR_MODEL_MODEL", "model-name")
    except Exception as e:
        print(f"[LLM] Error: {e}")

# 3. 在 _init_provider 中添加
providers = {
    ...
    BaseModel.YOUR_MODEL: self._init_your_model,
}
📞 支持与反馈
GitHub Issues: 提交问题
文档: 查看更多文档
