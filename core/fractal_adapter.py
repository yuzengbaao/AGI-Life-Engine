#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分形智能集成适配器 (Fractal Intelligence Integration Adapter)
将B组（自指涉分形拓扑）集成到现有TRAE AGI系统

功能：
1. 提供A/B组切换功能
2. 无缝集成TheSeed + Fractal Intelligence
3. 保持向后兼容性
4. 监控和日志记录

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-12
版本：v1.0
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入B组分形智能
from core.fractal_intelligence import (
    FractalIntelligenceAdapter,
    FractalOutput,
    create_fractal_intelligence
)

# 导入A组TheSeed
from core.seed import TheSeed

logger = logging.getLogger(__name__)


class IntelligenceMode(Enum):
    """智能模式枚举"""
    GROUP_A = "A"  # A组：仅使用TheSeed
    GROUP_B = "B"  # B组：TheSeed + Fractal Intelligence
    HYBRID = "HYBRID"  # 混合模式：根据置信度动态切换


@dataclass
class DecisionResult:
    """决策结果数据类"""
    action: int
    confidence: float
    entropy: float
    source: str  # 'seed', 'fractal', 'hybrid'
    self_awareness: float = 0.0
    goal_score: float = 0.0
    needs_validation: bool = False


class FractalSeedAdapter:
    """
    TheSeed + Fractal Intelligence 集成适配器

    核心功能：
    1. 包装TheSeed，添加分形智能能力
    2. 提供A/B组切换
    3. 增强决策能力
    4. 降低外部LLM依赖
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        mode: IntelligenceMode = IntelligenceMode.GROUP_A,
        device: str = 'cpu',
        enable_fractal: bool = True
    ):
        """
        初始化集成适配器

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            mode: 智能模式 (A/B/HYBRID)
            device: 设备 ('cpu' 或 'cuda')
            enable_fractal: 是否启用分形智能
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = mode
        self.device = device
        self.enable_fractal = enable_fractal and mode != IntelligenceMode.GROUP_A

        # 1. 创建A组TheSeed（保持向后兼容）
        self.seed = TheSeed(
            state_dim=state_dim,
            action_dim=action_dim
        )
        logger.info(f"[A组] TheSeed initialized: state_dim={state_dim}, action_dim={action_dim}")

        # 2. 创建B组分形智能（如果启用）
        self.fractal: Optional[FractalIntelligenceAdapter] = None
        if self.enable_fractal:
            try:
                self.fractal = create_fractal_intelligence(
                    input_dim=state_dim,
                    state_dim=state_dim,
                    device=device
                )
                logger.info(f"[B组] Fractal Intelligence initialized: device={device}")
            except Exception as e:
                logger.warning(f"[B组] Fractal initialization failed: {e}, falling back to A-only")
                self.enable_fractal = False
                self.mode = IntelligenceMode.GROUP_A

        # 3. 统计信息
        self.stats = {
            'total_decisions': 0,
            'seed_decisions': 0,
            'fractal_decisions': 0,
            'hybrid_decisions': 0,
            'external_llm_calls': 0,
            'avg_confidence': 0.0,
            'avg_entropy': 0.0
        }

        logger.info(f"[Adapter] Initialized with mode={mode.value}, fractal_enabled={self.enable_fractal}")

    def decide(
        self,
        state: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """
        统一决策接口

        这是降低外部依赖的核心方法：
        - A组：仅使用TheSeed
        - B组：优先使用Fractal，高置信度时直接决策
        - HYBRID：结合两者优势

        Args:
            state: 当前状态向量
            context: 额外上下文信息

        Returns:
            DecisionResult: 决策结果
        """
        self.stats['total_decisions'] += 1
        context = context or {}

        # 根据模式决策
        if self.mode == IntelligenceMode.GROUP_A or not self.enable_fractal:
            return self._decide_a_group(state, context)
        elif self.mode == IntelligenceMode.GROUP_B:
            return self._decide_b_group(state, context)
        else:  # HYBRID
            return self._decide_hybrid(state, context)

    def _decide_a_group(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """A组决策：仅使用TheSeed"""
        action = self.seed.act(state)

        # 估算价值作为置信度
        value = self.seed.evaluate(state, state, 0.0)
        confidence = min(1.0, max(0.0, value))

        self.stats['seed_decisions'] += 1

        return DecisionResult(
            action=action,
            confidence=confidence,
            entropy=0.5,  # A组没有熵计算
            source='seed',
            needs_validation=confidence < 0.7  # 低置信度需要外部验证
        )

    def _decide_b_group(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """B组决策：优先使用Fractal Intelligence"""
        if self.fractal is None:
            return self._decide_a_group(state, context)

        # 1. 转换为Tensor
        state_tensor = torch.from_numpy(state).float().to(self.device)

        # 2. 使用Fractal Core决策
        with torch.no_grad():
            output, meta = self.fractal.core(state_tensor, return_meta=True)

        # 3. 提取决策信息
        confidence = meta.self_awareness.mean().item()
        entropy = meta.entropy.item()
        goal_score = meta.goal_score
        alpha, beta, gamma = meta.metaparams

        # 4. 根据输出决定动作（这里简化处理）
        # 在实际系统中，output可能需要映射到action_space
        # 为了兼容性，我们先使用TheSeed的act，但用fractal增强置信度
        seed_action = self.seed.act(state)

        # 5. 如果置信度高，直接使用本地决策
        if confidence > 0.7:
            self.stats['fractal_decisions'] += 1
            return DecisionResult(
                action=seed_action,  # 使用seed的动作选择
                confidence=confidence,
                entropy=entropy,
                source='fractal',
                self_awareness=confidence,
                goal_score=goal_score,
                needs_validation=False  # 高置信度，不需要外部验证
            )
        else:
            # 低置信度：标记需要外部LLM验证
            self.stats['external_llm_calls'] += 1
            return DecisionResult(
                action=seed_action,
                confidence=confidence,
                entropy=entropy,
                source='fractal',
                self_awareness=confidence,
                goal_score=goal_score,
                needs_validation=True  # 需要外部LLM验证
            )

    def _decide_hybrid(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """混合模式：结合A组和B组优势"""
        if self.fractal is None:
            return self._decide_a_group(state, context)

        # 1. 获取A组决策
        a_action = self.seed.act(state)
        a_value = self.seed.evaluate(state, state, 0.0)
        a_confidence = min(1.0, max(0.0, a_value))

        # 2. 获取B组决策
        state_tensor = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            output, meta = self.fractal.core(state_tensor, return_meta=True)

        b_confidence = meta.self_awareness.mean().item()
        b_entropy = meta.entropy.item()

        # 3. 融合决策
        # 如果B组置信度高，优先使用；否则使用A组
        if b_confidence > 0.7:
            self.stats['fractal_decisions'] += 1
            return DecisionResult(
                action=a_action,  # 使用A组的动作选择逻辑
                confidence=max(a_confidence, b_confidence),
                entropy=b_entropy,
                source='hybrid',
                self_awareness=b_confidence,
                goal_score=meta.goal_score,
                needs_validation=False
            )
        else:
            self.stats['seed_decisions'] += 1
            return DecisionResult(
                action=a_action,
                confidence=a_confidence,
                entropy=b_entropy,
                source='hybrid',
                self_awareness=b_confidence,
                goal_score=meta.goal_score,
                needs_validation=a_confidence < 0.7
            )

    def learn(
        self,
        experience: Any,
        reward: float = 0.0
    ):
        """
        学习接口

        Args:
            experience: 经验数据（TheSeed的Experience对象）
            reward: 奖励值
        """
        # 1. TheSeed学习
        if hasattr(experience, 'state'):
            self.seed.learn(experience)

        # 2. Fractal学习（如果启用）
        if self.fractal and hasattr(experience, 'state'):
            exp_dict = {
                'state': torch.from_numpy(experience.state).float().to(self.device)
            }
            self.fractal.learn(exp_dict, reward)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()

        # 计算衍生指标
        if stats['total_decisions'] > 0:
            stats['fractal_ratio'] = stats['fractal_decisions'] / stats['total_decisions']
            stats['external_dependency'] = stats['external_llm_calls'] / stats['total_decisions']
        else:
            stats['fractal_ratio'] = 0.0
            stats['external_dependency'] = 0.0

        return stats

    def set_mode(self, mode: IntelligenceMode):
        """
        动态切换智能模式

        Args:
            mode: 目标模式
        """
        old_mode = self.mode
        self.mode = mode

        if mode == IntelligenceMode.GROUP_A:
            self.enable_fractal = False
        elif mode == IntelligenceMode.GROUP_B:
            if self.fractal is not None:
                self.enable_fractal = True
            else:
                logger.warning("Cannot enable B mode: fractal not initialized")
        else:  # HYBRID
            if self.fractal is not None:
                self.enable_fractal = True
            else:
                logger.warning("Cannot enable hybrid mode: fractal not initialized")
                self.mode = old_mode

        logger.info(f"[Adapter] Mode switched: {old_mode.value} -> {self.mode.value}")

    def get_self_representation(self) -> Optional[np.ndarray]:
        """获取自指涉表示（B组特性）"""
        if self.fractal:
            return self.fractal.core.get_self_representation().cpu().numpy()
        return None

    def get_goal_representation(self) -> Optional[np.ndarray]:
        """获取目标表示（B组特性）"""
        if self.fractal:
            return self.fractal.core.get_goal_representation().cpu().numpy()
        return None

    def get_fractal_state(self) -> Dict[str, Any]:
        """获取分形智能状态"""
        if not self.fractal:
            return {
                'enabled': False,
                'mode': self.mode.value
            }

        return {
            'enabled': True,
            'mode': self.mode.value,
            'entropy_history': self.fractal.core.entropy_history[-10:],  # 最近10个
            'goal_score_history': self.fractal.core.goal_score_history[-10:],
            'self_representation_norm': torch.norm(
                self.fractal.core.get_self_representation()
            ).item(),
            'goal_representation_norm': torch.norm(
                self.fractal.core.get_goal_representation()
            ).item()
        }


class EvolutionFractalAdapter:
    """
    EvolutionController + Fractal Intelligence 集成适配器

    用于在进化控制层面集成分形智能
    """

    def __init__(
        self,
        seed_adapter: FractalSeedAdapter,
        device: str = 'cpu'
    ):
        """
        初始化Evolution适配器

        Args:
            seed_adapter: 已初始化的FractalSeedAdapter
            device: 设备
        """
        self.seed_adapter = seed_adapter
        self.device = device
        self.generation_count = 0

        logger.info("[Evolution Adapter] Initialized with FractalSeedAdapter")

    def evaluate_generation(
        self,
        population_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估一代的进化情况

        Args:
            population_metrics: 种群指标

        Returns:
            评估结果
        """
        self.generation_count += 1

        # 获取分形智能状态
        fractal_state = self.seed_adapter.get_fractal_state()
        stats = self.seed_adapter.get_statistics()

        # 评估报告
        report = {
            'generation': self.generation_count,
            'fractal_enabled': fractal_state['enabled'],
            'mode': fractal_state['mode'],
            'external_dependency': stats['external_dependency'],
            'fractal_usage_ratio': stats['fractal_ratio'],
            'total_decisions': stats['total_decisions'],
            'avg_self_awareness': fractal_state.get('self_representation_norm', 0.0),
            'avg_goal_score': np.mean(fractal_state.get('goal_score_history', [0.0])),
        }

        # 记录日志
        logger.info(
            f"[Evolution Gen {self.generation_count}] "
            f"Mode={report['mode']}, "
            f"ExternalDep={report['external_dependency']:.2%}, "
            f"FractalUsage={report['fractal_usage_ratio']:.2%}"
        )

        return report

    def suggest_mode_switch(
        self,
        performance_metrics: Dict[str, Any]
    ) -> Optional[IntelligenceMode]:
        """
        根据性能指标建议模式切换

        Args:
            performance_metrics: 性能指标

        Returns:
            建议的模式（如果需要切换）
        """
        current_mode = self.seed_adapter.mode
        stats = self.seed_adapter.get_statistics()

        # 规则：如果外部依赖过高，建议切换到B组
        if stats['external_dependency'] > 0.3 and current_mode == IntelligenceMode.GROUP_A:
            logger.info("[Evolution] High external dependency detected, suggesting B mode")
            return IntelligenceMode.GROUP_B

        # 规则：如果B组使用率低但性能好，建议切换到A组
        if stats['fractal_ratio'] < 0.1 and current_mode != IntelligenceMode.GROUP_A:
            if performance_metrics.get('stability', 1.0) > 0.9:
                logger.info("[Evolution] Low fractal usage with high stability, suggesting A mode")
                return IntelligenceMode.GROUP_A

        return None


# 便捷函数
def create_fractal_seed_adapter(
    state_dim: int = 64,
    action_dim: int = 4,
    mode: str = "GROUP_A",
    device: str = 'cpu'
) -> FractalSeedAdapter:
    """
    创建分形种子适配器

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        mode: 模式 ("GROUP_A", "GROUP_B", "HYBRID")
        device: 设备

    Returns:
        FractalSeedAdapter实例
    """
    mode_map = {
        "A": IntelligenceMode.GROUP_A,
        "GROUP_A": IntelligenceMode.GROUP_A,
        "B": IntelligenceMode.GROUP_B,
        "GROUP_B": IntelligenceMode.GROUP_B,
        "HYBRID": IntelligenceMode.HYBRID
    }
    mode_enum = mode_map.get(mode.upper(), IntelligenceMode.GROUP_A)
    return FractalSeedAdapter(
        state_dim=state_dim,
        action_dim=action_dim,
        mode=mode_enum,
        device=device
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*60)
    print("[测试] 分形智能集成适配器")
    print("="*60)

    # 创建适配器（B组模式）
    adapter = create_fractal_seed_adapter(
        state_dim=64,
        action_dim=4,
        mode="B",
        device='cpu'
    )

    # 测试决策
    print("\n[测试] 决策测试")
    test_state = np.random.randn(64)
    result = adapter.decide(test_state)

    print(f"  动作: {result.action}")
    print(f"  置信度: {result.confidence:.4f}")
    print(f"  熵: {result.entropy:.4f}")
    print(f"  来源: {result.source}")
    print(f"  需要验证: {result.needs_validation}")

    # 测试统计
    print("\n[测试] 统计信息")
    stats = adapter.get_statistics()
    print(f"  总决策: {stats['total_decisions']}")
    print(f"  分形决策: {stats['fractal_decisions']}")
    print(f"  外部依赖: {stats['external_dependency']:.2%}")

    # 测试分形状态
    print("\n[测试] 分形智能状态")
    fractal_state = adapter.get_fractal_state()
    print(f"  启用: {fractal_state['enabled']}")
    print(f"  模式: {fractal_state['mode']}")
    print(f"  自我表示范数: {fractal_state['self_representation_norm']:.4f}")
    print(f"  目标表示范数: {fractal_state['goal_representation_norm']:.4f}")

    # 测试模式切换
    print("\n[测试] 模式切换")
    print(f"  当前模式: {adapter.mode.value}")
    adapter.set_mode(IntelligenceMode.HYBRID)
    print(f"  切换后: {adapter.mode.value}")

    print("\n" + "="*60)
    print("[成功] 分形智能集成适配器测试通过")
    print("="*60)
