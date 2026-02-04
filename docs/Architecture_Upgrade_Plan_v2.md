# AGI 架构升级计划书：流体智能与分层自我 (v2.0)

**日期**: 2025-12-15
**状态**: 实施中
**目标**: 实现基于"流体智能"与"分层自我"的架构升级，解决自我纠正循环问题，并提升系统灵活性。

---

## 1. 核心理论回顾

基于对《通用性的本质》等文档的分析，确立以下工程原则：

1.  **分层自我 (Layered Self)**: 解决 "Who am I" 的稳定性问题。
    *   **不可变核心 (Immutable Core)**: 身份、底层价值观（硬编码/只读）。
    *   **慢变层 (Slow Evolving)**: 长期记忆、性格倾向、偏好（数据库/配置文件，修改需多次确认）。
    *   **快变层 (Fast Adapting)**: 当前策略、短期情绪、临时工具（运行时状态，随任务快速迭代）。

2.  **流体智能 (Fluid Intelligence)**: 解决 "How to do" 的通用性问题。
    *   **通用基底**: AGI 引擎 + LLM (大脑皮层)。
    *   **流体软件**: 动态生成的 Python 脚本/技能 (Skill)。
    *   **工作流**: 遇到新任务 -> 观察 -> 生成专用代码 -> 执行 -> 固化为技能/丢弃。

---

## 2. 实施步骤

### 步骤 1: 构建分层自我模型 (Core Implementation)

**目标**: 创建 `core/layered_identity.py`，作为 AGI 的身份锚点。

*   **类设计**:
    *   `ImmutableCore`: 包含 UUID, CreationTime, BaseInstructions (如 "Do not harm user", "Be curious")。
    *   `SlowEvolvingLayer`: 包含 Beliefs, LongTermGoals, PersonalityTraits。提供 `update_belief(key, value, confidence)` 方法，只有当 `confidence` 累积到阈值时才真正修改。
    *   `FastAdaptingLayer`: 包含 CurrentContext, ShortTermFocus, EmotionalState (Transient)。

### 步骤 2: 集成至生命引擎 (Integration)

**目标**: 修改 `AGI_Life_Engine.py`，使用 `LayeredIdentity` 替代原有的简单 Persona。

*   **初始化**: 在 `__init__` 中实例化 `LayeredIdentity`。
*   **决策循环**: 在 `_cognitive_reflection_v2` 中，将 `Identity` 的状态注入到 System Prompt 中。
    *   *System Prompt 结构*:
        *   [Immutable]: 你是 TRAE AGI... (来自 Core)
        *   [Slow]: 你目前相信... (来自 SlowLayer)
        *   [Fast]: 你当前关注... (来自 FastLayer)

### 3. 增强流体技能循环 (Fluid Loop)

**目标**: 强化 "生成-执行-固化" 流程。

*   **技能生成**: 确认 `SkillLibrary` 已支持 `add_executable_skill`。
*   **主动学习**: 在 `GoalType.EXPLORE` 或 `GoalType.LEARN` 中，明确加入 "尝试编写脚本解决问题" 的逻辑。

---

## 3. 预期效果

1.  **稳定性提升**: "不可变核心" 防止 AGI 在反思中迷失自我或修改底层约束。
2.  **适应性增强**: "快变层" 允许 AGI 针对当前任务极速调整策略，而不会产生长期的认知负担。
3.  **能力累积**: "流体软件" 将一次性的成功操作固化为永久能力。

---

## 4. 施工清单

- [ ] 创建 `core/layered_identity.py`
- [ ] 修改 `AGI_Life_Engine.py` 集成新身份系统
- [ ] (可选) 创建 `core/meta_cognition.py` 用于管理修改冷却期 (本次暂由 Identity 内部管理)
- [ ] 重启系统进行验证
