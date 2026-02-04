# AGI 系统长程推理能力增强预研报告

**日期**: 2025-12-30
**作者**: 智能辅助开发助手
**状态**: 待审查

## 1. 问题定义与背景

### 1.1 现状描述：被蒙住眼睛的专家
目前的 AGI 系统表现出一种“被蒙住眼睛的专家”的特征。虽然底层的 LLM（大语言模型）具备强大的逻辑推理能力，但在实际任务执行中，Agent 往往表现得笨拙、短视，只能对眼前的输入做出反应，缺乏对全局信息的感知和长程规划能力。

### 1.2 核心症结分析
经过对 `core/agents` 目录下 `executor.py`, `planner.py`, `base_agent.py` 等核心文件的分析，我们确定了以下根本原因：

1.  **缺乏“感官”工具 (Sensory Deprivation)**:
    *   `ExecutorAgent` 只能执行特定的写文件、读文件命令，缺乏主动检索项目结构、搜索代码库、读取全局配置的高级工具。
    *   它像是一个在黑暗中摸索的人，只能处理手边的东西，无法“看到”整个房间的布局。

2.  **思维链缺失 (Missing Chain of Thought)**:
    *   `BaseAgent` 类过于单薄，缺乏强制性的系统提示词（System Prompt）来规范 LLM 的思考过程。
    *   当前的执行逻辑往往是 `Input -> Tool Execution` 的直接映射，跳过了 `Observation -> Hypothesis -> Validation -> Plan` 的关键推理步骤。

3.  **上下文碎片化 (Fragmented Context)**:
    *   Agent 的记忆往往局限于当前的对话窗口，缺乏对项目整体架构、历史决策和跨文件依赖关系的持久化认知。

## 2. 解决方案建议：双管齐下

为了解决上述问题，建议采取“提示词工程”与“函数架构升级”相结合的策略。

### 2.1 方案 A：提示词层面的重构 (Prompt Engineering)
**目标**: 强制开启 LLM 的“慢思考”模式。

*   **引入结构化思维流**: 在 `system_prompt` 中强制要求 Agent 输出 JSON 或特定 XML 标签包裹的思维过程，例如：
    ```xml
    <thought_process>
    1. 观察: 我看到了什么？(当前环境、用户意图)
    2. 假设: 问题可能出在哪里？
    3. 验证计划: 我需要检查哪些文件来证实假设？
    4. 行动: 执行具体操作。
    </thought_process>
    ```
*   **角色定义**: 明确 Agent 的角色不仅仅是“执行者”，更是“架构师”和“侦探”。

### 2.2 方案 B：函数与工具层面的增强 (Function/Tool Enhancement)
**目标**: 赋予 Agent “视觉”和“全局感知”。

*   **新增全局感知工具**:
    *   `inspect_project_structure(depth=2)`: 允许 Agent 查看目录树，建立空间感。
    *   `search_knowledge_base(query)`: 允许 Agent 检索过往的经验、文档和最佳实践。
    *   `grep_codebase(pattern)`: 允许 Agent 在全局范围内搜索代码引用，而非仅限于单文件阅读。

*   **上下文管理器 (Context Manager)**:
    *   引入一个动态的 Context 窗口，自动根据任务相关性加载关键文件的摘要，而非依赖 Agent 手动读取。

### 2.3 架构调整 (Architecture Refactoring)
*   **从 `Linear Execution` 转向 `ReAct Loop`**:
    *   目前的 `ExecutorAgent` 偏向于线性执行。
    *   建议重构为 `Observation-Thought-Action` 循环，允许 Agent 在执行每一步后重新评估环境变化。

## 3. 风险评估与应对 (Risk Assessment)

### 3.1 系统震颤风险 (System Tremors)
**风险**: 修改底层的 `BaseAgent` 或 `ExecutorAgent` 可能会破坏现有的、依赖特定输出格式的遗留任务（Legacy Tasks）。
**应对**:
*   **继承策略**: 不要直接修改 `ExecutorAgent`，而是创建一个新的子类 `ReasoningExecutorAgent` 或 `ArchitectAgent`。
*   **灰度发布**: 仅在处理“复杂任务”（由 Planner 判定）时激活新的 Agent，简单任务仍由旧 Agent 处理。

### 3.2 性能与成本
**风险**: 增加思维链和全局搜索会显著增加 Token 消耗和响应延迟。
**应对**:
*   **动态开关**: 允许用户配置“深度思考模式”与“快速响应模式”。

## 4. 实施路线图 (Implementation Roadmap)

1.  **阶段一：基础设施准备 (Week 1)**
    *   在 `core/agents/base_agent.py` 中增加 `system_prompt` 模板支持。
    *   开发 `core/system_tools.py` 的增强版，增加 `ProjectStructure` 和 `GlobalSearch` 功能。

2.  **阶段二：原型验证 (Week 1-2)**
    *   创建 `ReasoningAgent` 原型，集成新的工具集。
    *   在一个隔离的沙箱环境中测试其对复杂重构任务的处理能力。

3.  **阶段三：集成与调优 (Week 2)**
    *   将新 Agent 集成到 `Planner` 的分发逻辑中。
    *   根据测试结果调整提示词的严格程度。

## 5. 结论

系统目前的“笨拙”并非能力不足，而是机制受限。通过赋予 Agent 全局感知的工具（眼睛）和强制的思维链条（大脑皮层），我们可以解锁 LLM 真正的长程推理潜力。建议按照“最小侵入性”原则，通过扩展而非修改现有类的方式开始实施。