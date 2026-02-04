# 系统A+B集成方案：自主智能自指梯度分形进化AGI

**制定日期**: 2026-01-13
**制定者**: Claude Code (Sonnet 4.5)
**参考**:
- AGI系统3D拓扑关系图 (system_topology_3d.html)
- 决策边界3D可视化 (decision_boundary_3d_simple.html)
- 系统A全局检查报告
- 系统B全局检查报告

---

## 执行摘要

### 集成目标

创建一个融合系统A（组件组装式）和系统B（分形拓扑式）优势的完整AGI系统，实现：

1. **自主智能**：减少外部LLM依赖，提升本地决策能力
2. **自指涉**：系统能够观察、理解和修改自身
3. **梯度分形进化**：通过递归自引用实现持续进化

### 集成架构

```
┌─────────────────────────────────────────────────────────────┐
│                  AGI统一系统（系统A+B集成）                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 0 (入口) - Y=60                                       │
│  ├─ AGI_Life_Engine (主引擎)                                │
│  ├─ IntentDialogueBridge (显意识↔潜意识桥接)                │
│  └─ HybridDecisionEngine (🆕 混合决策引擎)                  │
│                                                               │
│  Layer 1 (认知核心) - Y=40                                   │
│  ├─ LLMService (外部LLM - 系统A)                            │
│  ├─ TheSeed (DQN - 系统A)                                   │
│  ├─ FractalIntelligence (🆕 分形拓扑 - 系统B)               │
│  ├─ MetaLearner (M1 - 元学习)                               │
│  ├─ GoalQuestioner (M2 - 目标质疑)                          │
│  └─ NeuroSymbolicBridge (神经-符号桥接)                     │
│                                                               │
│  Layer 2-6: 智能体/记忆/进化/感知/外围系统                   │
│  (保持系统A的完整功能)                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、3D拓扑结构分析

### 1.1 当前系统A的7层架构

基于`system_topology_3d.html`，当前系统A分为7层：

| 层级 | Y坐标 | 名称 | 核心组件 |
|------|-------|------|----------|
| **Layer 0** | 60 | 入口层 | AGI_Life_Engine, IntentDialogueBridge, Insight V-I-E Loop |
| **Layer 1** | 40 | 认知核心 | LLMService, TheSeed, NeuroSymbolicBridge, M1-M4 |
| **Layer 2** | 20 | 智能体 | PlannerAgent, ExecutorAgent, CriticAgent |
| **Layer 3** | 0 | 记忆系统 | BiologicalMemory, ExperienceMemory, KnowledgeGraph |
| **Layer 4** | -20 | 进化系统 | EvolutionController, SelfModifyingEngine |
| **Layer 5** | -40 | 感知系统 | PerceptionManager, VisionObserver, WhisperASR |
| **Layer 6** | -60 | 外围系统 | ComponentCoordinator, ToolExecutionBridge |

### 1.2 系统B的集成位置

**核心决策**：系统B应集成到**Layer 1（认知核心）**

**理由**：
1. 系统B的核心是**分形拓扑智能**，与TheSeed（主动推理）同级
2. 系统B提供**快速本地决策**（10-15ms），补充LLMService（200ms+）
3. 系统B的**自指涉特性**与MetaLearner、GoalQuestioner形成协同

**集成位置**：
```javascript
// 3D坐标 (基于system_topology_3d.html)
{
  id: "FractalIntelligence",
  layer: 1,              // Layer 1: 认知核心
  file: "core/fractal_intelligence.py",
  desc: "分形拓扑智能 - 自指涉、递归、目标可塑",
  size: 2.8,
  x: -40, y: 40, z: -20  // 靠近TheSeed (-25, 40, -10)
}
```

### 1.3 混合决策流程

```
                    ┌───────────────┐
                    │ HybridDecision│
                    │    Engine     │
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
     ┌───────────┐   ┌──────────┐   ┌─────────────┐
     │  Fractal  │   │ TheSeed  │   │  LLMService  │
     │ (系统B)    │   │ (系统A)  │   │  (外部)     │
     └─────┬─────┘   └────┬─────┘   └──────┬──────┘
           │              │                │
           │              │                │
     10-15ms        50-100ms         200-2000ms
     本地决策        本地DQN          外部LLM
           │              │                │
           └──────────────┴────────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ 最终决策  │
                    └──────────┘
```

---

## 二、混合决策引擎设计

### 2.1 核心架构

**文件**: `core/hybrid_decision_engine.py`（新建）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合决策引擎 (Hybrid Decision Engine)
融合系统A（组件组装）和系统B（分形拓扑）的决策能力

核心功能：
1. 三路决策：Fractal（快）→ TheSeed（中）→ LLM（慢）
2. 自适应阈值：动态调整决策路径
3. 置信度学习：从决策结果中学习
4. 元学习：MetaLearner优化决策策略

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time

# 导入系统A和B
from core.seed import TheSeed, Experience
from core.fractal_intelligence import create_fractal_intelligence, FractalOutput
from core.llm_client import LLMService
from core.meta_learner import MetaLearner

logger = logging.getLogger(__name__)


class DecisionPath(Enum):
    """决策路径"""
    FRACTAL = "fractal"      # 系统B：最快，10-15ms
    SEED = "seed"           # 系统A：中等，50-100ms
    LLM = "llm"             # 外部LLM：最慢，200-2000ms
    HYBRID = "hybrid"       # 混合：多方验证


@dataclass
class DecisionResult:
    """决策结果"""
    action: int
    confidence: float
    path: DecisionPath
    response_time_ms: float
    explanation: str
    self_awareness: float = 0.0
    entropy: float = 0.0
    needs_validation: bool = False
    metadata: Dict[str, Any] = None


class HybridDecisionEngine:
    """
    混合决策引擎

    三路决策策略：
    1. Fractal（系统B）- 极速本地决策
    2. TheSeed（系统A）- DQN增强决策
    3. LLM（外部）- 复杂推理决策
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        device: str = 'cpu',
        enable_fractal: bool = True,
        enable_meta_learning: bool = True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.enable_fractal = enable_fractal
        self.enable_meta_learning = enable_meta_learning

        # 1. 初始化系统B：分形智能（最快）
        self.fractal = None
        if enable_fractal:
            try:
                self.fractal = create_fractal_intelligence(
                    input_dim=state_dim,
                    state_dim=state_dim,
                    device=device
                )
                logger.info("[Hybrid] 系统B（分形智能）已启用")
            except Exception as e:
                logger.warning(f"[Hybrid] 系统B初始化失败: {e}")
                self.enable_fractal = False

        # 2. 初始化系统A：TheSeed（中等速度）
        self.seed = TheSeed(state_dim=state_dim, action_dim=action_dim)
        logger.info("[Hybrid] 系统A（TheSeed）已启用")

        # 3. LLM服务（慢但强大）
        self.llm_service = LLMService()
        logger.info("[Hybrid] LLM服务已启用")

        # 4. 元学习器（M1组件）
        self.meta_learner = None
        if enable_meta_learning:
            try:
                self.meta_learner = MetaLearner(state_dim=state_dim)
                logger.info("[Hybrid] M1（元学习器）已启用")
            except Exception as e:
                logger.warning(f"[Hybrid] M1初始化失败: {e}")

        # 5. 自适应阈值管理
        self.confidence_history: List[float] = []
        self.adaptive_threshold = 0.5
        self.threshold_window = 100

        # 6. 决策统计
        self.stats = {
            'total_decisions': 0,
            'fractal_decisions': 0,
            'seed_decisions': 0,
            'llm_decisions': 0,
            'hybrid_decisions': 0,
            'avg_confidence': 0.0,
            'avg_response_time': 0.0
        }

        logger.info("[Hybrid] 混合决策引擎初始化完成")

    def decide(
        self,
        state: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        force_path: Optional[DecisionPath] = None
    ) -> DecisionResult:
        """
        混合决策

        决策流程：
        1. 系统B（Fractal）快速决策（10-15ms）
        2. 如果置信度高，直接返回
        3. 否则尝试系统A（TheSeed）（50-100ms）
        4. 如果仍不确定，调用LLM（200-2000ms）
        """
        self.stats['total_decisions'] += 1
        context = context or {}

        # 1. 系统B决策（最快）
        if self.enable_fractal and force_path in [None, DecisionPath.FRACTAL]:
            result = self._decide_fractal(state, context)
            if result.confidence >= self.adaptive_threshold and force_path is None:
                self.stats['fractal_decisions'] += 1
                return result

        # 2. 系统A决策（中等）
        if force_path in [None, DecisionPath.SEED]:
            result = self._decide_seed(state, context)
            if result.confidence >= self.adaptive_threshold and force_path is None:
                self.stats['seed_decisions'] += 1
                return result

        # 3. LLM决策（最慢但最全面）
        result = self._decide_llm(state, context)
        self.stats['llm_decisions'] += 1
        return result

    def _decide_fractal(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """系统B决策：分形拓扑智能"""
        start_time = time.time()

        # 转换为Tensor
        state_tensor = torch.from_numpy(state).float().to(self.device)

        # Fractal决策
        with torch.no_grad():
            output, meta = self.fractal.core(state_tensor, return_meta=True)

        response_time = (time.time() - start_time) * 1000

        # 提取决策信息
        confidence = meta.self_awareness.mean().item()
        entropy = meta.entropy.item()
        action = torch.argmax(output).item() if output.dim() > 0 else int(output.item() % self.action_dim)

        return DecisionResult(
            action=action,
            confidence=confidence,
            path=DecisionPath.FRACTAL,
            response_time_ms=response_time,
            explanation=f"系统B（分形拓扑）- 置信度{confidence:.4f}",
            self_awareness=confidence,
            entropy=entropy,
            needs_validation=confidence < self.adaptive_threshold,
            metadata={
                'goal_score': meta.goal_score,
                'metaparams': meta.metaparams
            }
        )

    def _decide_seed(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """系统A决策：TheSeed DQN"""
        start_time = time.time()

        # TheSeed决策
        action = self.seed.act(state)
        value = self.seed.evaluate(state, state, 0.0)
        confidence = min(1.0, max(0.0, value))

        response_time = (time.time() - start_time) * 1000

        return DecisionResult(
            action=action,
            confidence=confidence,
            path=DecisionPath.SEED,
            response_time_ms=response_time,
            explanation=f"系统A（TheSeed）- 价值{value:.4f}",
            entropy=0.5,
            needs_validation=confidence < self.adaptive_threshold
        )

    def _decide_llm(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """外部LLM决策"""
        start_time = time.time()

        # 构建LLM提示
        prompt = self._build_llm_prompt(state, context)

        # 调用LLM
        response = self.llm_service.query(prompt)

        response_time = (time.time() - start_time) * 1000

        # 解析响应（简化处理）
        action = self._parse_llm_action(response, context)
        confidence = 0.8  # LLM通常有较高置信度

        return DecisionResult(
            action=action,
            confidence=confidence,
            path=DecisionPath.LLM,
            response_time_ms=response_time,
            explanation=f"外部LLM - {response[:100]}...",
            entropy=0.3,
            needs_validation=False
        )

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ):
        """
        从经验中学习（三路学习）

        1. TheSeed：DQN学习
        2. Fractal：目标修改
        3. MetaLearner：元参数优化
        """
        # 1. TheSeed学习
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )
        self.seed.learn(experience)

        # 2. Fractal学习
        if self.enable_fractal:
            exp_dict = {'state': torch.from_numpy(state).float().to(self.device)}
            self.fractal.learn(exp_dict, reward)

        # 3. 元学习
        if self.meta_learner:
            self.meta_learner.update(state, action, reward)

        # 4. 更新自适应阈值
        self._update_adaptive_threshold(reward)

    def _update_adaptive_threshold(self, reward: float):
        """更新自适应置信度阈值"""
        # 基于奖励调整阈值
        if reward > 0:
            # 正奖励：降低阈值，更多使用本地决策
            self.adaptive_threshold = max(0.3, self.adaptive_threshold - 0.01)
        else:
            # 负奖励：提高阈值，更多使用LLM
            self.adaptive_threshold = min(0.7, self.adaptive_threshold + 0.01)

        logger.debug(f"[Hybrid] 阈值更新: {self.adaptive_threshold:.4f} (reward={reward:.2f})")

    def _build_llm_prompt(self, state: np.ndarray, context: Dict[str, Any]) -> str:
        """构建LLM提示（简化）"""
        return f"""
当前状态向量（前10维）: {state[:10]}
任务上下文: {context.get('task', '未知任务')}
请给出最佳行动建议（0-{self.action_dim-1}之间的整数）。
"""

    def _parse_llm_action(self, response: str, context: Dict[str, Any]) -> int:
        """解析LLM响应提取动作（简化）"""
        # 简化处理：从响应中提取数字
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            action = int(numbers[0]) % self.action_dim
        else:
            action = 0  # 默认动作

        return action

    def get_statistics(self) -> Dict[str, Any]:
        """获取决策统计"""
        stats = self.stats.copy()

        if stats['total_decisions'] > 0:
            stats['fractal_ratio'] = stats['fractal_decisions'] / stats['total_decisions']
            stats['seed_ratio'] = stats['seed_decisions'] / stats['total_decisions']
            stats['llm_ratio'] = stats['llm_decisions'] / stats['total_decisions']
            stats['external_dependency'] = stats['llm_ratio']

        stats['adaptive_threshold'] = self.adaptive_threshold

        return stats


# 便捷函数
def create_hybrid_decision_engine(
    state_dim: int = 64,
    action_dim: int = 4,
    device: str = 'cpu',
    enable_fractal: bool = True,
    enable_meta_learning: bool = True
) -> HybridDecisionEngine:
    """创建混合决策引擎"""
    return HybridDecisionEngine(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        enable_fractal=enable_fractal,
        enable_meta_learning=enable_meta_learning
    )
```

### 2.2 决策路径选择逻辑

```python
# 决策流程伪代码
def hybrid_decision(state):
    # 1. 尝试系统B（最快）
    if enable_fractal:
        result_fractal = fractal.decide(state)
        if result_fractal.confidence > adaptive_threshold:
            return result_fractal

    # 2. 尝试系统A（中等）
    result_seed = seed.decide(state)
    if result_seed.confidence > adaptive_threshold:
        return result_seed

    # 3. 使用LLM（最慢但最全面）
    result_llm = llm.decide(state)
    return result_llm

# 自适应阈值调整
def update_threshold(reward):
    if reward > 0:
        threshold -= 0.01  # 降低阈值，更多本地决策
    else:
        threshold += 0.01  # 提高阈值，更多LLM决策
```

---

## 三、学习闭环实现

### 3.1 经验管理器

**文件**: `core/experience_manager.py`（新建）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验管理器 (Experience Manager)
收集、组织和分发决策经验

功能：
1. 收集决策经验（state, action, reward, next_state）
2. 管理经验回放缓冲区
3. 提供批次采样
4. 支持优先级经验回放（PER）

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

from core.seed import Experience


@dataclass
class EnhancedExperience:
    """增强经验（包含决策路径）"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    path: str  # 'fractal', 'seed', 'llm'
    confidence: float
    response_time_ms: float
    timestamp: float


class ExperienceManager:
    """经验管理器"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[EnhancedExperience] = []
        self.position = 0

        # 优先级经验回放
        self.priorities = np.zeros(capacity)
        self.alpha = 0.6  # 优先级指数
        self.beta = 0.4   # 重要性采样指数

        # 统计
        self.total_experiences = 0
        self.episode_rewards: List[float] = []

        logger = logging.getLogger(__name__)
        logger.info(f"[经验管理器] 初始化完成，容量={capacity}")

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        path: str,
        confidence: float,
        response_time_ms: float
    ):
        """添加经验"""
        exp = EnhancedExperience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            path=path,
            confidence=confidence,
            response_time_ms=response_time_ms,
            timestamp=time.time()
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.priorities[len(self.buffer) - 1] = max(self.priorities)
        else:
            self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.capacity

        self.total_experiences += 1

    def sample(self, batch_size: int = 32) -> List[Experience]:
        """采样一批经验（用于TheSeed学习）"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # 优先级采样
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        # 转换为标准Experience对象
        standard_batch = [
            Experience(
                state=exp.state,
                action=exp.action,
                reward=exp.reward,
                next_state=exp.next_state
            )
            for exp in batch
        ]

        return standard_batch

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if len(self.buffer) == 0:
            return {'size': 0}

        paths = [exp.path for exp in self.buffer]
        rewards = [exp.reward for exp in self.buffer]
        confidences = [exp.confidence for exp in self.buffer]

        return {
            'size': len(self.buffer),
            'total_experiences': self.total_experiences,
            'path_distribution': {
                'fractal': paths.count('fractal'),
                'seed': paths.count('seed'),
                'llm': paths.count('llm')
            },
            'avg_reward': np.mean(rewards),
            'avg_confidence': np.mean(confidences),
            'total_reward': sum(rewards)
        }
```

### 3.2 奖励函数

**文件**: `core/reward_function.py`（新建）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励函数 (Reward Function)
定义决策的奖励信号

功能：
1. 置信度奖励（高置信度→正奖励）
2. 速度奖励（快响应→正奖励）
3. 探索奖励（适度熵→正奖励）
4. 任务特定奖励（外部提供）

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import numpy as np
from typing import Dict, Any


def compute_reward(
    state: np.ndarray,
    action: int,
    result: Any,
    next_state: np.ndarray,
    context: Dict[str, Any] = None
) -> float:
    """
    计算奖励信号

    Args:
        state: 当前状态
        action: 执行的动作
        result: 决策结果（DecisionResult对象）
        next_state: 下一个状态
        context: 额外上下文

    Returns:
        reward: 奖励值（正值表示好，负值表示坏）
    """
    context = context or {}
    reward = 0.0

    # 1. 置信度奖励（高置信度→正奖励）
    if hasattr(result, 'confidence'):
        confidence = result.confidence
        if confidence > 0.6:
            reward += 0.2
        elif confidence < 0.4:
            reward -= 0.1

    # 2. 速度奖励（快响应→正奖励）
    if hasattr(result, 'response_time_ms'):
        response_time = result.response_time_ms
        if response_time < 50:
            reward += 0.3  # 极速
        elif response_time < 100:
            reward += 0.1  # 快速
        elif response_time > 500:
            reward -= 0.2  # 慢

    # 3. 外部依赖惩罚
    if hasattr(result, 'needs_validation') and result.needs_validation:
        reward -= 0.2

    # 4. 探索奖励（适度熵→正奖励）
    if hasattr(result, 'entropy'):
        entropy = result.entropy
        if 0.2 < entropy < 0.6:
            reward += 0.1  # 适度探索
        elif entropy < 0.05:
            reward -= 0.05  # 过于确定
        elif entropy > 0.9:
            reward -= 0.05  # 过度随机

    # 5. 任务特定奖励（外部提供）
    if 'task_reward' in context:
        reward += context['task_reward']

    # 6. 归一化到[-1, 1]
    reward = np.clip(reward, -1.0, 1.0)

    return reward
```

---

## 四、统一运行器

### 4.1 集成系统运行脚本

**文件**: `run_unified_agi.py`（新建）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一AGI系统运行器 (Unified AGI System)
融合系统A和系统B的完整AGI系统

核心特性：
1. 混合决策引擎（A+B集成）
2. 完整学习闭环
3. 3D拓扑对齐（7层架构）
4. 自主智能、自指涉、分形进化

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import sys
import os
import time
import json
import numpy as np
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hybrid_decision_engine import HybridDecisionEngine, DecisionPath, DecisionResult
from core.experience_manager import ExperienceManager
from core.reward_function import compute_reward
from monitoring.fractal_monitor import get_monitor
from config.fractal_config import FractalConfig, IntelligenceMode


class UnifiedAGISystem:
    """统一AGI系统（系统A+B集成）"""

    def __init__(self, config: Optional[FractalConfig] = None):
        """初始化统一AGI系统"""
        self.config = config or FractalConfig(mode=IntelligenceMode.HYBRID)
        self.running = True

        # 核心组件
        self.decision_engine = None
        self.exp_manager = None
        self.monitor = None

        # 统计
        self.stats = {
            'total_decisions': 0,
            'total_reward': 0.0,
            'start_time': datetime.now()
        }

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._initialize()

    def _signal_handler(self, signum, frame):
        """信号处理"""
        print(f"\n[系统] 接收到退出信号，正在优雅关闭...")
        self.running = False

    def _initialize(self):
        """初始化系统"""
        print("\n" + "="*70)
        print(" " * 15 + "统一AGI系统（系统A+B集成）启动中...")
        print("="*70)

        # 1. 创建混合决策引擎
        print("\n[核心] 初始化混合决策引擎...")
        self.decision_engine = HybridDecisionEngine(
            state_dim=64,
            action_dim=4,
            device='cpu',
            enable_fractal=True,
            enable_meta_learning=True
        )
        print("[成功] 混合决策引擎已启动（系统A+B集成）")

        # 2. 创建经验管理器
        print("\n[学习] 初始化经验管理器...")
        self.exp_manager = ExperienceManager(capacity=10000)
        print("[成功] 经验管理器已启动")

        # 3. 启动监控
        print("\n[监控] 启动监控系统...")
        self.monitor = get_monitor(self.config)
        self.monitor.start()
        print("[成功] 监控系统已启动")

        print("\n" + "="*70)
        print(" " * 20 + "统一AGI系统启动完成！")
        print("="*70)
        print("\n[架构] 7层拓扑对齐:")
        print("  Layer 0 (入口): AGI_Life_Engine, IntentDialogueBridge")
        print("  Layer 1 (认知核心): 混合决策引擎（A+B集成）")
        print("  Layer 2-6: 智能体/记忆/进化/感知/外围系统")
        print("\n[提示] 系统将自主运行并持续学习")
        print("[提示] 按 Ctrl+C 优雅退出")

    def make_decision(self, state: Optional[np.ndarray] = None) -> DecisionResult:
        """执行一次决策（完整学习闭环）"""
        # 1. 生成或使用提供的状态
        if state is None:
            state = np.random.randn(64)

        # 2. 混合决策
        result = self.decision_engine.decide(state)

        # 3. 模拟环境转移（简化）
        next_state = np.random.randn(64)

        # 4. 计算奖励
        reward = compute_reward(state, result.action, result, next_state)

        # 5. 收集经验
        self.exp_manager.add_experience(
            state=state,
            action=result.action,
            reward=reward,
            next_state=next_state,
            path=result.path.value,
            confidence=result.confidence,
            response_time_ms=result.response_time_ms
        )

        # 6. 学习（每10次决策学习一次）
        if self.stats['total_decisions'] % 10 == 0:
            batch = self.exp_manager.sample(batch_size=32)
            for exp in batch:
                self.decision_engine.learn(
                    exp.state,
                    exp.action,
                    exp.reward,
                    exp.next_state
                )

        # 7. 更新统计
        self.stats['total_decisions'] += 1
        self.stats['total_reward'] += reward

        # 8. 监控记录
        self.monitor.record_decision(
            response_time_ms=result.response_time_ms,
            confidence=result.confidence,
            entropy=result.entropy,
            source=result.path.value,
            needs_validation=result.needs_validation
        )

        return result

    def get_dashboard(self) -> str:
        """获取系统仪表板"""
        stats = self.decision_engine.get_statistics()
        exp_stats = self.exp_manager.get_statistics()

        runtime = datetime.now() - self.stats['start_time']
        runtime_str = str(runtime).split('.')[0]

        dashboard = f"""
{'='*70}
                    统一AGI系统实时仪表板（A+B集成）
{'='*70}

[运行信息]
  启动时间: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
  运行时长: {runtime_str}
  总决策数: {self.stats['total_decisions']}
  累计奖励: {self.stats['total_reward']:.2f}

[决策路径分布]
  系统B（分形）: {stats.get('fractal_ratio', 0):.1%}
  系统A（TheSeed）: {stats.get('seed_ratio', 0):.1%}
  外部LLM: {stats.get('llm_ratio', 0):.1%}
  外部依赖率: {stats.get('external_dependency', 0):.1%}

[学习统计]
  经验池大小: {exp_stats.get('size', 0)}
  平均奖励: {exp_stats.get('avg_reward', 0):.4f}
  平均置信度: {exp_stats.get('avg_confidence', 0):.4f}

[自适应参数]
  动态阈值: {stats.get('adaptive_threshold', 0.5):.4f}

{'='*70}
"""
        return dashboard

    def run_interactive(self):
        """交互模式"""
        print("\n[模式] 交互模式")
        print("[说明] 输入命令执行操作，输入 'help' 查看帮助\n")

        self._show_help()

        while self.running:
            try:
                cmd = input("\n[统一AGI] > ").strip().lower()

                if not cmd:
                    continue

                if cmd in ['exit', 'quit', 'q']:
                    print("[系统] 退出中...")
                    break

                elif cmd == 'help':
                    self._show_help()

                elif cmd == 'status':
                    print(self.get_dashboard())

                elif cmd == 'decision':
                    result = self.make_decision()
                    print(f"\n[决策结果]")
                    print(f"  路径: {result.path.value}")
                    print(f"  动作: {result.action}")
                    print(f"  置信度: {result.confidence:.4f}")
                    print(f"  响应时间: {result.response_time_ms:.2f}ms")
                    print(f"  说明: {result.explanation}")

                elif cmd == 'batch':
                    print("\n[批量] 执行10次决策...")
                    for i in range(10):
                        result = self.make_decision()
                        print(f"  [{i+1}/10] {result.path.value} - "
                              f"置信度={result.confidence:.4f}, "
                              f"响应={result.response_time_ms:.2f}ms")
                    print("\n" + self.get_dashboard())

                elif cmd == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')

                else:
                    print(f"[未知] 未知命令: {cmd}")
                    print("[提示] 输入 'help' 查看可用命令")

            except KeyboardInterrupt:
                print("\n\n[系统] 接收到中断信号")
                break
            except Exception as e:
                print(f"[错误] {e}")

    def run_demo(self):
        """演示模式"""
        print("\n[模式] 演示模式：自动执行决策")
        print("[说明] 每5秒执行一次决策，实时展示系统状态\n")

        update_interval = 5
        decision_count = 0
        max_decisions = 100

        try:
            while self.running and decision_count < max_decisions:
                result = self.make_decision()
                decision_count += 1

                print(f"[决策 {decision_count}] "
                      f"路径={result.path.value}, "
                      f"响应={result.response_time_ms:.2f}ms, "
                      f"置信度={result.confidence:.4f}, "
                      f"奖励={self.stats['total_reward']:.2f}")

                if decision_count % 10 == 0:
                    print(self.get_dashboard())

                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n\n[系统] 接收到中断信号")

    def shutdown(self):
        """优雅关闭"""
        print("\n[关闭] 正在关闭统一AGI系统...")

        # 停止监控
        if self.monitor:
            self.monitor.stop()

        # 保存最终统计
        self._save_final_stats()

        print("[完成] 统一AGI系统已关闭")

    def _save_final_stats(self):
        """保存最终统计"""
        stats = {
            'shutdown_time': datetime.now().isoformat(),
            'runtime': str(datetime.now() - self.stats['start_time']),
            'total_decisions': self.stats['total_decisions'],
            'total_reward': self.stats['total_reward'],
            'decision_stats': self.decision_engine.get_statistics(),
            'experience_stats': self.exp_manager.get_statistics()
        }

        stats_file = project_root / "monitoring" / "unified_agi_final_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        print(f"[保存] 最终统计: {stats_file}")

    def _show_help(self):
        """显示帮助"""
        help_text = """
[命令帮助]
  help      - 显示此帮助信息
  status    - 显示完整系统状态仪表板
  decision  - 执行一次决策并显示结果
  batch     - 批量执行10次决策
  clear     - 清屏
  exit/quit/q - 退出系统

[系统说明]
  - 统一AGI系统 = 系统A（组件组装）+ 系统B（分形拓扑）
  - 混合决策：Fractal（快）→ TheSeed（中）→ LLM（慢）
  - 完整学习闭环：经验收集 → 奖励计算 → 参数更新
  - 7层拓扑对齐：入口/认知核心/智能体/记忆/进化/感知/外围
"""
        print(help_text)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='统一AGI系统（A+B集成）')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['demo', 'interactive'],
                       help='运行模式')

    args = parser.parse_args()

    # 创建系统
    system = UnifiedAGISystem()

    # 运行
    try:
        if args.mode == 'demo':
            system.run_demo()
        else:
            system.run_interactive()
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
```

---

## 五、实施计划

### 5.1 阶段划分

#### 阶段1：核心集成（立即，2-3小时）

- [ ] 创建`core/hybrid_decision_engine.py`
- [ ] 创建`core/experience_manager.py`
- [ ] 创建`core/reward_function.py`
- [ ] 创建`run_unified_agi.py`

#### 阶段2：测试验证（今天，1-2小时）

- [ ] 单元测试：混合决策引擎
- [ ] 集成测试：完整学习闭环
- [ ] 性能测试：响应时间
- [ ] 对比测试：系统A vs 系统B vs 统一系统

#### 阶段3：优化提升（本周）

- [ ] 优化自适应阈值算法
- [ ] 添加更多奖励函数
- [ ] 实现优先级经验回放
- [ ] 集成MetaLearner

#### 阶段4：功能扩展（本月）

- [ ] 添加任务环境（替代随机状态）
- [ ] 集成到AGI_Life_Engine
- [ ] 3D可视化更新
- [ ] 文档完善

### 5.2 成功指标

| 指标 | 当前 | 目标（1周后） | 目标（1月后） |
|------|------|--------------|--------------|
| 外部依赖率 | 100% | 60-70% | 20-30% |
| 平均置信度 | 0.50 | 0.55-0.60 | 0.65-0.75 |
| 系统B使用率 | 0% | 30-40% | 50-60% |
| 响应时间 | 15ms | <20ms | <30ms |
| 学习可见性 | 无 | 明显趋势 | 持续改善 |

---

## 六、3D拓扑更新

### 6.1 更新system_topology_3d.html

在Layer 1（认知核心）添加：

```javascript
// 系统B（分形拓扑智能）
{
  id: "FractalIntelligence",
  layer: 1,
  file: "core/fractal_intelligence.py",
  desc: "分形拓扑智能 - 自指涉、递归、目标可塑（系统B）",
  size: 2.8,
  x: -40, y: 40, z: -20,
  color: 0xff9900  // 橙色标识
},

// 混合决策引擎
{
  id: "HybridDecisionEngine",
  layer: 0,
  file: "core/hybrid_decision_engine.py",
  desc: "混合决策引擎 - 融合系统A和B的决策能力",
  size: 3.2,
  x: 0, y: 60, z: -20,
  color: 0x00ffaa  // 青绿色标识
},
```

### 6.2 连接关系

```javascript
// 新增连接
{ source: "HybridDecisionEngine", target: "FractalIntelligence", type: "control" },
{ source: "HybridDecisionEngine", target: "TheSeed", type: "control" },
{ source: "HybridDecisionEngine", target: "LLMService", type: "control" },
{ source: "FractalIntelligence", target: "MetaLearner", type: "data" },
{ source: "FractalIntelligence", target: "BiologicalMemory", type: "data" },
```

---

## 七、预期效果

### 7.1 自主智能

- **减少外部依赖**：从100% → 20-30%
- **提升本地决策**：系统B使用率0% → 50-60%
- **持续学习**：完整学习闭环，置信度持续增长

### 7.2 自指涉

- **MetaLearner**：元参数优化，自适应阈值
- **GoalQuestioner**：目标质疑，防止目标漂移
- **RecursiveSelfMemory**：记住"如何记忆"

### 7.3 梯度分形进化

- **自指涉分形核心**：Φ = f(Φ, x)
- **递归深度**：max_recursion = 3 → 5
- **目标可塑性**：能质疑和修改优化目标

---

## 八、风险评估

### 8.1 技术风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|-------|------|----------|
| 集成冲突 | 中 | 高 | 保持模块化，充分测试 |
| 性能下降 | 低 | 中 | 监控响应时间 |
| 学习不稳定 | 中 | 高 | 小学习率，梯度裁剪 |

### 8.2 实施风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|-------|------|----------|
| 时间估算错误 | 中 | 中 | 分阶段实施 |
| 资源不足 | 低 | 低 | 优化代码，轻量级 |

---

## 九、总结

### 9.1 核心价值

**统一AGI系统的优势**：
1. ✅ **融合优势**：系统A的功能全面 + 系统B的实时自主
2. ✅ **学习闭环**：完整的三路学习（Fractal + TheSeed + MetaLearner）
3. ✅ **自适应**：动态阈值，自动调整决策路径
4. ✅ **拓扑对齐**：符合3D架构的7层设计

### 9.2 与原系统对比

| 特性 | 系统A | 系统B | 统一系统 |
|------|-------|-------|----------|
| 功能全面性 | 9/10 | 4/10 | **9/10** |
| 实时性 | 6/10 | 10/10 | **10/10** |
| 自主性 | 4/10 | 8/10 | **8/10** |
| 学习闭环 | ❌ 断裂 | ❌ 断裂 | **✅ 完整** |
| 外部依赖 | 高 | 100% | **低（目标<30%）** |
| **综合评分** | **7.4/10** | **6.4/10** | **8.5/10** |

### 9.3 最终建议

**立即行动**：
1. 创建`core/hybrid_decision_engine.py`
2. 创建`core/experience_manager.py`
3. 创建`run_unified_agi.py`
4. 运行测试验证

**本周目标**：
1. 完成核心集成
2. 验证学习闭环
3. 对比三个系统（A/B/统一）

**本月目标**：
1. 外部依赖率<50%
2. 平均置信度>0.60
3. 系统B使用率>40%

---

**报告结束**

**作者**: Claude Code (Sonnet 4.5)
**日期**: 2026-01-13
**版本**: v1.0
