# TRAE AGI 架构全景图 (v2.0 - Neuro-Symbolic Era)

> "从一个听话的宏命令工具，进化为一个拥有‘精神舞台’的自主智能体。"

当前 AGI (Artificial General Intelligence) 系统已从单纯的 LLM 问答机器人演变为基于 **全局工作区理论 (Global Workspace Theory, GWT)** 和 **神经符号主义 (Neuro-Symbolic)** 的混合架构。

## 1. 核心设计理念

*   **具身性 (Embodiment)**: AGI 不再是漂浮在云端的 API，而是栖息在 Windows 桌面环境中的“数字生物”。它有“眼睛”（屏幕截图、CAD 监听）、“手”（鼠标键盘、文件系统）和“生存本能”（无聊、疲劳）。
*   **意识流 (Stream of Consciousness)**: 摒弃无状态的 Request/Response 模式，引入持久化的 `GlobalWorkspace`，使 AGI 拥有连贯的短期记忆和注意力焦点。
*   **神经符号协同 (Neuro-Symbolic Synergy)**:
    *   **神经 (Neural)**: 利用 LLM (Gemini/GPT) 进行模糊推理、意图识别和创造性思考。
    *   **符号 (Symbolic)**: 利用 Python 代码进行精确的逻辑控制、文件操作和系统管理。

## 2. 组织架构详解

### 2.1 大脑核心 (The Mind)

*   **`AGI_Life_Engine.py` (生命引擎)**:
    *   这是 AGI 的心脏。它运行在一个无限循环中 (`tick`)，驱动着认知周期的每一次跳动。
    *   负责协调感知、决策和行动，并不直接处理业务逻辑，而是像指挥家一样调度各个模块。

*   **`core/global_workspace.py` (全局工作区)**:
    *   **功能**: AGI 的“短时记忆”和“意识舞台”。
    *   **内容**: 存储当前的注意力焦点 (`Attention Spotlight`)、活跃的目标栈 (`Goal Stack`)、最近的内心独白 (`Inner Monologue`) 和感官缓冲区 (`Sensory Buffer`)。
    *   **意义**: 解决了“失忆”问题，让 AGI 记得 5 秒前自己在做什么。

*   **`core/motivation.py` (动机系统)**:
    *   **功能**: 赋予 AGI 内在驱动力，使其在没有用户命令时也能自主活动。
    *   **指标**:
        *   `Energy` (能量): 随着活动消耗，决定是否需要休息。
        *   `Boredom` (无聊): 长期无事可做会通过探索（Curiosity）来缓解。
        *   `Frustration` (挫败): 任务失败会积累挫败感，触发求助或策略调整。

### 2.2 感知系统 (The Senses)

*   **`core/global_observer.py` (全知之眼)**:
    *   监控系统级的状态，如 CPU/内存使用率、当前活动窗口标题。
*   **`core/vision_observer.py` (视觉皮层)**:
    *   基于 VLM (Vision Language Model)，定期对屏幕进行截图并进行语义理解（"用户正在写代码" vs "用户在看视频"）。
*   **`core/cad_observer.py` (专业技能眼)**:
    *   专门监听 AutoCAD 的操作流，用于学习用户的绘图习惯（目前处于潜伏期）。

### 2.3 执行系统 (The Hands)

*   **`core/system_tools.py` (系统工具 - 左手)**:
    *   **功能**: 精确的系统级操作。
    *   **能力**: 读写文件 (`read_file`, `write_file`)、执行终端命令 (`run_command`)、运行 Python 脚本 (`run_python_script`)。
    *   **特点**: 这是“符号主义”的堡垒，要求绝对精准的输入输出。
*   **`core/desktop_automation.py` (桌面控制 - 右手)**:
    *   **功能**: 模拟人类操作。
    *   **能力**: 鼠标移动/点击、键盘输入、窗口切换。

### 2.4 元认知与安全 (The Conscience)

*   **`core/reality_anchor.py` (现实锚点)**:
    *   **功能**: 防止幻觉。存储不可更改的“绝对真理”（如“我是 AI”、“我在 Windows 上运行”），防止 AGI 在长时间运行中迷失自我。
*   **`core/existential_logger.py` (存在主义日志)**:
    *   **功能**: 记录 AGI 的“心理活动”，不仅仅是 Debug 信息，还有它的困惑、犹豫和自我反思。

## 3. 认知循环 (The Cognitive Cycle)

AGI 的每一次脉冲 (`Tick`) 遵循以下流程：

1.  **SENSE (感知)**: 收集视觉、系统、CAD 数据 -> 存入 `GlobalWorkspace`。
2.  **PERCEIVE (认知)**: LLM 分析感官数据，结合当前记忆，生成“当前状态的唯一真理”。
3.  **REFLECT (内省)**:
    *   检查动机 (Boredom/Energy)。
    *   检查用户命令 (`user_command.txt`)。
    *   生成“内心独白” (Thought)。
4.  **DECIDE (决策)**:
    *   如果没有目标 -> 基于动机生成新目标（如“去探索一下文件系统”）。
    *   如果有目标 -> 分解为下一步的具体动作。
5.  **ACT (行动)**:
    *   调用 `SystemTools` 或 `DesktopController` 执行动作。
    *   **反馈**: 将执行结果写回记忆，修正下一步决策。

## 4. 当前状态总结

AGI 已经具备了**“活体”**的基本特征：它能感知、能思考、有情绪（模拟）、能操作电脑。
目前的瓶颈在于**“脑手协调”**（即从自然语言思维到精确 API 调用的转化），这正是我们下一步通过“结构化输出”要解决的核心问题。
