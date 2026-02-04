# AI 模型基础研究、流派之争与未来进化路书

## 1. 主流学派：自回归变换器 (Autoregressive Transformers)
**代表模型**: GPT-4, Grok-3, Gemini 1.5/2, DeepSeek-V3/R1
**核心信条**: "Next Token Prediction is All You Need" (预测下一个词即智能)

这一派通过暴力美学（Scaling Laws）统治了当下的 AI 届。

*   **基础研究 (The Holy Grail)**:
    *   **Attention Is All You Need (2017)**: Google 团队提出的 Transformer 架构，抛弃了循环神经网络 (RNN)，利用自注意力机制并行处理长文本。
    *   **Scaling Laws (2020/2022)**: OpenAI (Kaplan) 和 DeepMind (Chinchilla) 证明，只要算力、数据和参数量同步增加，模型能力就会呈现幂律提升（涌现能力）。
    *   **RLHF (2017/2022)**: Christiano (OpenAI) 等人引入的“人类反馈强化学习”，解决了模型“胡说八道”的问题，使其对齐人类价值观。
    *   **MoE (Mixture of Experts)**: 稀疏混合专家模型 (DeepSeek/Grok 采用)，用更低的计算成本训练更大的参数量（如 671B 参数，每次推理只激活 37B）。

*   **当前瓶颈**: 
    *   **幻觉 (Hallucination)**: 本质上是概率预测，不懂真理，只懂“听起来像真的”。
    *   **物理世界缺失**: 它们是通过阅读文本了解世界的，是“缸中之脑”，没有物理直觉。

## 2. "非主流"学派：世界模型与具身智能
这一派认为 LLM 只是通往 AGI 的一部分，甚至是一条死胡同。

### A. 杨立坤 (Yann LeCun) - Meta FAIR
*   **核心观点**: LLM 是 "Off-Ramp" (下高速匝道)，通往 AGI 的正道是 **世界模型 (World Models)**。
*   **基础研究**: **JEPA (Joint-Embedding Predictive Architecture)**。
    *   **批判**: 自回归生成（一个字一个字崩）太低效且容易累积误差。人类不是预测下一个“像素”，而是预测“状态”。
    *   **方案**: 在抽象的特征空间（Latent Space）进行预测。不预测“视频的下一帧是什么”，而是预测“如果我踢这个球，球的状态会变成什么”。
    *   **前景**: **极高**。这是实现 Robot/Agent 具备常识物理推理的关键。Meta 的 I-JEPA 和 V-JEPA 正在验证这一路线。

### B. 李飞飞 (Li Fei-Fei) - Stanford / World Labs
*   **核心观点**: **空间智能 (Spatial Intelligence)** 是进化的源头。
*   **基础研究**: 计算机视觉 (ImageNet) -> 具身智能 (VoxPoser / VIMA)。
    *   **理念**: "Seeing is for Doing"。智能不仅仅是语言，更是理解 3D 空间、物体属性（硬/软、重/轻）以及如何与它们交互（Affordance）。
    *   **方案**: 结合 VLM (视觉语言模型) 与 3D 几何表征，让 AI 能理解并在 3D 世界中操作。
    *   **前景**: **极高**。这是 TRAE AGI `DesktopController` 进化的终极方向——不仅看到屏幕截图（2D），更能理解 UI 的层级结构和交互逻辑（类 3D）。

## 3. 未来进化方向：三大融合 (The Great Convergence)

未来的 AGI 不会是单一架构，而是以下三个方向的融合：

### 方向一：System 2 Reasoning (慢思考)
*   **代表**: **OpenAI o1 / DeepSeek-R1**
*   **进化点**: 从“直觉反应” (System 1, 快速生成) 转向“深思熟虑” (System 2, 强化学习搜索)。
*   **机制**: 在输出结果前，先进行内部的思维链 (CoT) 推演、试错、回溯。
*   **对 TRAE AGI 的启示**: 我们的 `PlannerAgent` 必须具备自我纠错能力，而不是一次性生成计划。

### 方向二：Embodied World Models (具身世界模型)
*   **代表**: **Tesla FSD v12 / Google RT-2 / Meta JEPA**
*   **进化点**: 语言不再是核心，**行动 (Action)** 才是核心。
*   **机制**: 视频输入 -> 物理规律预测 -> 动作输出。
*   **对 TRAE AGI 的启示**: 我们的 `VisionObserver` 需要进化为能预测“鼠标点击后果”的模型，而不仅仅是识别按钮。

### 方向三：Neuro-Symbolic (神经符号主义)
*   **代表**: **AlphaGeometry / AlphaProof**
*   **进化点**: 解决 LLM 数学差、逻辑不严谨的问题。
*   **机制**: 神经网络 (直觉) + 形式化逻辑/数学证明器 (严谨)。
*   **对 TRAE AGI 的启示**: 在涉及文件操作、系统命令时，不能只靠 LLM 瞎猜，需要结合确定性的代码逻辑检查。

## 4. 总结：TRAE AGI 的站位
我们目前处于 **"基于主流 LLM (DeepSeek) 的 System 2 尝试"** 阶段。
未来需要向 **"杨立坤的世界模型"** 靠拢——即不仅仅是“想”，而是通过 `DesktopController` 去“做”，并在做的过程中建立对操作系统这个“物理世界”的预测模型。
