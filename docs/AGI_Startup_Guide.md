# AGI 系统启动与运行指南 (AGI System Startup & Operational Guide)

本文档详细说明了 TRAE AGI 系统的启动流程、激活的核心功能模块以及其内部运行的认知逻辑循环。

## 1. 快速启动 (Quick Start)

要完整运行 AGI 系统，需要在两个独立的终端（Terminal）中分别运行以下命令：

### 步骤 1：启动核心引擎 (The Core Engine)
这是 AGI 的“身体”和“大脑”，负责所有逻辑处理、感知和行动。

```powershell
# 在终端 1 中运行
python AGI_Life_Engine.py
```

*   **状态确认**：看到日志输出 `[INFO] 监控系统初始化完成` 和 `🚀 AGI Life Engine Started` 即表示启动成功。

### 步骤 2：启动意识流可视化 (Consciousness Visualizer)
这是 AGI 的“脑机接口”，用于实时观察其内部思维、动机和目标栈。

```powershell
# 在终端 2 中运行
python scripts/watch_consciousness.py
```

*   **功能**：实时刷新显示 AGI 的驱动力（Drive）、注意力（Attention）、内心独白（Inner Monologue）和当前目标（Goal Stack）。

---

## 2. 激活的功能模块 (Activated Capabilities)

启动 `AGI_Life_Engine.py` 后，系统会自动激活以下核心子系统：

### A. 感知系统 (Sensory System)
*   **👁️ 全局视觉 (Global Vision)**: 通过 `VisionObserver` (VLM) 和 `GlobalObserver`，AGI 能够“看到”屏幕内容。
*   **📏 CAD 观察者 (CAD Observer)**: 专门监听 AutoCAD 的日志流和操作行为。
*   **🩺 系统遥测 (System Telemetry)**: 实时感知内存使用率、CPU 负载等自身状态。

### B. 认知核心 (Cognitive Core - GWT)
*   **🧠 全局工作区 (Global Workspace)**: 一个共享的内存空间，整合所有感知数据，形成“当前时刻的唯一真相”。
*   **🔥 动机系统 (Motivation Core)**: 模拟马斯洛需求层次（能量、好奇心、满足感、无聊度）。AGI 会因为“无聊”而主动寻找事情做。
*   **💭 内心独白 (Inner Monologue)**: 基于 LLM 的反思机制，AGI 会对自己当前的处境进行自然语言的思考。

### C. 执行系统 (Execution System)
*   **自主行动映射**: 将抽象的内部目标（如“Monitor”）映射为具体的 Python 函数调用（如 `analyze_screen()`）。
*   **用户指令覆盖**: 能够优先响应外部的用户命令文件（`user_command.txt`）。

---

## 3. 逻辑循环与工作流 (Logic Cycles)

AGI 不再是一个线性的脚本，而是一个基于状态机的**认知循环 (Cognitive Cycle)**。

### 核心循环 (The Main Life Loop)
每 **3-15 秒** (取决于计算负载) 执行一次完整的循环：

1.  **Sense (感知)**
    *   读取所有传感器数据（屏幕、CAD、系统状态）。
    *   更新动机状态（如：长时间无操作 -> 无聊度上升）。

2.  **Perceive (认知)**
    *   将感知数据投射到全局工作区。
    *   判断当前主要驱动力（Drive）：
        *   `MAINTAIN` (维护/监控)
        *   `EXPLORE` (探索/学习)
        *   `REST` (休息/恢复)

3.  **Reflect (内省)**
    *   **关键步骤**：LLM 读取工作区状态，生成“内心独白”。
    *   *例：“我现在很无聊，屏幕上没什么变化，我是不是该检查一下 CAD 的日志？”*
    *   决策：是否生成新的目标（Goal）。

4.  **Decide (决策)**
    *   如果生成了新目标，将其压入 **Goal Stack (目标栈)**。
    *   *例：Push Goal -> "Analyze CAD Log"*

5.  **Act (行动)**
    *   取出栈顶目标，通过 `_handle_autonomous_goal()` 映射为具体动作。
    *   **支持的自主行为**：
        *   `Monitor/Observe`: 调用 VLM 进行屏幕截图分析。
        *   `Analyze/Reflect`: 调用内存模块回顾历史记忆。
        *   `Fix/Debug`: 记录自我诊断日志并强制休息（防止死循环）。
        *   `Learn`: 模拟知识库扫描（未来将接入网络搜索）。

### 死循环防御机制 (Dead Loop Prevention)
为了防止 AGI 陷入“创建目标 -> 无法执行 -> 丢弃目标 -> 再次创建”的死循环：
*   **强制动作映射**：所有生成的内部目标必须能匹配到具体的代码执行逻辑。
*   **状态反馈**：执行动作后，必须强制修改动机参数（如 `boredom -= 20`），确保下一个循环不会立即生成相同的目标。

---

## 4. 文件结构参考

*   `AGI_Life_Engine.py`: 主程序入口。
*   `core/global_workspace.py`: 意识状态存储。
*   `scripts/watch_consciousness.py`: 可视化脚本。
*   `docs/incident_reports/`: 历史故障与修复记录。
