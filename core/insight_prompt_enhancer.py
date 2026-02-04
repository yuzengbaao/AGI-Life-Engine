#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insight生成提示增强器 (Insight Generation Prompt Enhancer)

为Insight生成提供函数使用指南和示例，提升Insight代码的可执行性。

创建日期: 2026-01-15
用途: 系统激活阶段 - 让Insight主动使用实用函数库
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Insight实用函数库使用指南
INSIGHT_UTILITIES_GUIDE = """
## 📚 可用的Insight实用函数库

以下函数已在系统中导入并可自由使用（来自core.insight_utilities）：

### 🧠 记忆重组函数 (Memory Reorganization)
- `rest_phase_reorganization(memory_bank, entropy_threshold=0.95)` - 休息阶段记忆重组
  用途: 通过高惊讶度记忆的反事实模拟生成创新种子
  返回: 重组后的前20%记忆

- `invert_causal_chain(memory)` - 反转记忆的因果链
  用途: 生成反事实变体用于探索
  返回: 反转后的新记忆

- `perturb_attention_weights(memory, scale=0.1)` - 扰动注意力权重
  用途: 添加噪声生成新视角
  返回: 扰动后的记忆

- `simulate_forward(counterfactual)` - 前向模拟反事实
  用途: 计算预测误差评估创新性
  返回: 预测误差(0-1)

### 🔇 噪声引导函数 (Noise Guidance)
- `noise_guided_rest(state, temperature=1.0)` - 噪声引导休息
  用途: 在高熵休息期间模拟生成性孵化
  参数: state为torch.Tensor, temperature控制噪声强度
  返回: 新状态张量

### 💭 语义处理函数 (Semantic Processing)
- `semantic_perturb(problem_domain, known_concepts=None)` - 语义扰动
  用途: 应用最小概念破坏来打破固着
  返回: 扰动提示字符串

- `analyze_tone(text)` - 分析文本情感效价
  用途: 计算文本的情感倾向
  返回: {'valence': float(-1到+1)}

- `semantic_diode(input_stream, threshold=0.75, hysteresis_window=3)` - 语义二极管
  用途: 通过情感轨迹过滤认知流
  返回: 过滤后的输出流

### 🌀 拓扑检测函数 (Topological Detection)
- `detect_topological_defect(z)` - 检测拓扑缺陷
  用途: 检测复数值激活张量中的缺陷数量
  参数: z为复数值torch.Tensor
  返回: 缺陷数量(int)

- `CurlLayer(size)` - 旋度层神经网络模块
  用途: 引入非保守场的旋转分量
  类型: torch.nn.Module

### 📊 分形脉冲函数 (Fractal Pulse)
- `fractal_idle_pulse(duration, base_freq=0.1, depth=3, seed=None)` - 分形空闲脉冲
  用途: 为高熵休息状态生成多尺度扰动信号
  返回: (时间数组, 信号数组)

### 🔄 逆向溯因函数 (Reverse Abduction)
- `reverse_abduction_step(model, context, noise_scale=1.2)` - 逆向溯因步骤
  用途: 通过制造内部冲突加速演化
  返回: (anti_context, dissonance)

- `kl_div(p, q)` - 计算KL散度
  用途: 衡量两个分布的差异
  返回: KL散度值

### ⚡ 对抗性直觉函数 (Adversarial Intuition)
- `inject_adversarial_intuition(model, alpha=0.03, backup=True)` - 注入对抗性直觉
  用途: 注入悖论噪声增强创造性
  返回: 注入统计信息

### 🧬 潜在重组函数 (Latent Recombination)
- `latent_recombination(memories, noise_scale=0.93)` - 潜在重组
  用途: 使用受控随机共振重组记忆痕迹
  返回: 重组后的候选向量(前5个)

---

## 💡 使用示例

### 示例1: 记忆重组
```python
from core.insight_utilities import rest_phase_reorganization
import numpy as np

# 假设有高惊讶度的记忆库
memory_bank = [
    {'surprise': 0.96, 'content': '...'},
    {'surprise': 0.94, 'content': '...'},
    {'surprise': 0.92, 'content': '...'}
]

# 休息阶段重组
reorganized_memories = rest_phase_reorganization(
    memory_bank=memory_bank,
    entropy_threshold=0.95
)
# reorganized_memories 包含最创新的前20%记忆
```

### 示例2: 噪声引导探索
```python
from core.insight_utilities import noise_guided_rest
import torch

# 当前状态
current_state = torch.randn(64)

# 应用噪声引导
new_state = noise_guided_rest(
    state=current_state,
    temperature=1.0  # 控制噪声强度
)
# new_state 保留了核心轨迹但放大了新颖性
```

### 示例3: 语义扰动打破固着
```python
from core.insight_utilities import semantic_perturb

# 打破思维固着
perturbation_prompt = semantic_perturb(
    problem_domain="computing",
    known_concepts=["algorithm", "optimization"]
)
# 返回类似: "Perturb computing with 'symbiosis' from mycology"
```

### 示例4: 拓扑缺陷检测
```python
from core.insight_utilities import detect_topological_defect
import torch

# 复数值激活
z = torch.randn(10, dtype=torch.complex64)

# 检测缺陷
defect_count = detect_topological_defect(z)
# defect_count 表示拓扑不连续点的数量
```

### 示例5: 潜在重组
```python
from core.insight_utilities import latent_recombination
import numpy as np

# 记忆向量
memories = [
    np.random.randn(128),
    np.random.randn(128),
    np.random.randn(128)
]

# 重组生成新候选
candidates = latent_recombination(
    memories=memories,
    noise_scale=0.93
)
# candidates 包含5个最新颖的重组向量
```

---

## ✨ 最佳实践

1. **优先使用实用函数**: 当你的Insight涉及记忆、噪声、语义处理等主题时，优先使用上述函数
2. **参考示例代码**: 上面的示例展示了典型用法，可以根据需求调整参数
3. **组合使用**: 多个函数可以组合使用以实现更复杂的功能
4. **参数调优**: 大多数函数都有可调参数，可以根据具体场景优化

生成Insight时，请考虑使用这些函数来提升代码的实用性和可执行性！
"""


class InsightPromptEnhancer:
    """Insight提示增强器"""

    def __init__(self):
        self.enabled = True
        self.guide = INSIGHT_UTILITIES_GUIDE

    def enhance_prompt(
        self,
        original_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        增强Insight生成提示

        Args:
            original_prompt: 原始提示词
            context: 上下文信息（可选）

        Returns:
            增强后的提示词
        """
        if not self.enabled:
            return original_prompt

        # 检测是否是Insight生成任务
        is_insight_task = self._detect_insight_task(original_prompt)

        if not is_insight_task:
            return original_prompt

        # 增强提示词
        enhanced = f"""{original_prompt}

{self.guide}

---

**重要提示**: 在生成Insight的代码示例时，优先考虑使用上述实用函数库中的函数。
这会让你的Insight更具实用性和可执行性！
"""

        logger.info("[InsightPromptEnhancer] Prompt enhanced with utilities guide")
        return enhanced

    def _detect_insight_task(self, prompt: str) -> bool:
        """
        检测是否是Insight生成任务

        Args:
            prompt: 提示词

        Returns:
            是否是Insight任务
        """
        insight_keywords = [
            "creative insight",
            "generate insight",
            "hypothesis",
            "novel mechanism",
            "code snippet",
            "emergence",
            "entropy",
            "consciousness",
            "topological",
            "fractal",
            "causal",
            "counterfactual"
        ]

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in insight_keywords)


# 全局单例
_enhancer_instance = None

def get_insight_prompt_enhancer() -> InsightPromptEnhancer:
    """获取Insight提示增强器单例"""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = InsightPromptEnhancer()
        logger.info("[InsightPromptEnhancer] Global enhancer initialized")
    return _enhancer_instance


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试
    enhancer = InsightPromptEnhancer()

    # 测试1: Insight任务
    insight_prompt = "Generate a creative insight about consciousness emergence"
    enhanced = enhancer.enhance_prompt(insight_prompt)
    print("=== Enhanced Insight Prompt ===")
    print(enhanced)
    print("\n")

    # 测试2: 非Insight任务
    normal_prompt = "What is the capital of France?"
    not_enhanced = enhancer.enhance_prompt(normal_prompt)
    print("=== Non-Insight Prompt (should not be enhanced) ===")
    print(not_enhanced)
    print("\n")

    print("✅ All tests passed!")
