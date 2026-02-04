# AGI 系统长程推理能力增强施工计划

**日期**: 2025-12-30
**状态**: 待执行
**目标**: 激活系统内休眠的高级感知与推理组件，赋予 Agent 全局视野和长程推理能力。

## 1. 核心策略：资产重组 (Asset Reactivation)
利用现有的 `ProjectIndexer` (AST分析) 和 `KnowledgeReasoner` (逻辑推理) 组件，通过封装将其暴露给 `ExecutorAgent`，而非从零开发。

## 2. 详细施工步骤

### 阶段一：工具层激活 (Tool Layer Activation)
**目标**: 让 Agent “看见”项目全貌和“回忆”过往经验。

1.  **增强 `SystemTools` (`core/system_tools.py`)**:
    *   **集成 `ProjectIndexer`**: 引入 `inspect_project_structure` 工具，允许 Agent 查看指定深度的目录树。
    *   **集成 `MemoryBridge`**: 引入 `search_knowledge` 工具，允许 Agent 语义检索 `ExperienceMemory`。
    *   **集成 `KnowledgeGraph`**: 暴露 `query_codebase` 接口，允许 Agent 查询符号定义和引用关系（如“谁调用了这个函数？”）。

2.  **验证**:
    *   创建一个测试脚本，模拟 Agent 调用 `inspect_project_structure` 和 `search_knowledge`，确保返回结果格式清晰且 Token 消耗可控。

### 阶段二：大脑皮层构建 (Cortex Construction)
**目标**: 创建具备“慢思考”能力的 Agent 原型。

1.  **新建 `ReasoningAgent` (`core/agents/reasoning_agent.py`)**:
    *   继承自 `BaseAgent`。
    *   **核心特性**: 内置 `ReAct` (Reasoning + Acting) 循环。
    *   **System Prompt**: 强制要求输出 `<thought>`, `<observation>`, `<plan>` 标签。
    *   **工具集**: 默认挂载阶段一开发的所有高级感知工具。

2.  **逻辑**:
    *   接收任务 -> 观察环境 (Tree/Graph) -> 检索记忆 (Memory) -> 制定计划 -> 执行 -> 验证结果 -> 循环。

### 阶段三：神经连接 (Neural Connection)
**目标**: 将新大脑接入现有系统，接管复杂任务。

1.  **升级 `PlannerAgent` (`core/agents/planner.py`)**:
    *   修改任务分发逻辑：当检测到任务属于“复杂重构”、“代码分析”或“系统诊断”类别时，将子任务标记为 `REASONING_TASK`。

2.  **升级 `ExecutorAgent` (`core/agents/executor.py`)**:
    *   增加对 `ReasoningAgent` 的委托机制。如果收到 `REASONING_TASK`，则转交 `ReasoningAgent` 处理，而不是自己硬执行。

## 3. 风险控制与回滚

*   **沙箱测试**: 所有新工具首先在 `tests/` 目录下进行单元测试，不直接修改生产环境代码。
*   **非侵入式扩展**: `ReasoningAgent` 作为独立文件存在，不修改 `ExecutorAgent` 的核心逻辑，确保旧有简单任务不受影响。

## 4. 执行清单 (Checklist)

- [ ] `core/system_tools.py`: 添加 `inspect_project_structure`
- [ ] `core/system_tools.py`: 集成 `MemoryBridge` 和 `search_knowledge`
- [ ] `tests/test_new_tools.py`: 验证新工具有效性
- [ ] `core/agents/reasoning_agent.py`: 实现 ReAct 循环 Agent
- [ ] `docs/user_guide_reasoning.md`: 更新文档，说明如何触发深度推理模式

---
**批准人**: 用户
**执行人**: 智能辅助开发助手