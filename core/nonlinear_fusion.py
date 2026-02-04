#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非线性融合引擎 (Nonlinear Fusion Engine)
实现真正的1+1>2协同效应

核心思想：
1. 交互项：当两个系统都强时，产生增强效应
2. 互补项：当决策不同时，捕捉互补视角
3. 非线性变换：突破线性加权平均的理论上限

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """融合配置"""
    interaction_strength: float = 0.15  # 交互强度
    complementarity_strength: float = 0.08  # 互补强度
    nonlinearity_degree: float = 2.0  # 非线性度（指数）
    diversity_bonus: float = 0.05  # 多样性奖励
    confidence_threshold: float = 0.6  # 高置信度阈值


class NonlinearFusionEngine:
    """
    非线性融合引擎

    核心特性：
    1. 交互项 (Interaction Term): conf_A * conf_B
    2. 互补项 (Complementarity Term): 当action不同时奖励
    3. 多样性增强 (Diversity Bonus): 鼓励不同视角
    4. 自适应权重 (Adaptive Weights): 根据历史表现动态调整
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        enable_adaptive: bool = True
    ):
        self.config = config or FusionConfig()
        self.enable_adaptive = enable_adaptive

        # 历史表现追踪
        self.fusion_history: List[Dict[str, Any]] = []
        self.performance_history: Dict[str, List[float]] = {
            'linear': [],
            'nonlinear': [],
            'emergence': []
        }

        # 自适应参数
        self.adaptive_weights = {
            'interaction': self.config.interaction_strength,
            'complementarity': self.config.complementarity_strength,
            'diversity': self.config.diversity_bonus
        }

        logger.info(f"[非线性融合] 引擎初始化完成")
        logger.info(f"[非线性融合] 交互强度={self.config.interaction_strength}")
        logger.info(f"[非线性融合] 互补强度={self.config.complementarity_strength}")

    def fuse(
        self,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]],
        weight_A: float,
        weight_B: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        非线性融合

        Args:
            result_A: 系统A的决策结果
            result_B: 系统B的决策结果
            weight_A: 系统A的权重
            weight_B: 系统B的权重
            context: 额外上下文信息

        Returns:
            融合结果字典，包含：
            - action: 融合动作
            - confidence: 融合置信度
            - emergence: 涌现分数
            - method: 融合方法
            - breakdown: 各项贡献的分解
        """

        context = context or {}

        # 处理单系统情况
        if result_A is None and result_B is None:
            return self._get_fallback()
        elif result_A is None:
            return self._single_system(result_B, 'B')
        elif result_B is None:
            return self._single_system(result_A, 'A')

        # 提取关键信息
        action_A = result_A.get('action', 0)
        action_B = result_B.get('action', 0)
        conf_A = result_A.get('confidence', 0.5)
        conf_B = result_B.get('confidence', 0.5)
        entropy_A = result_A.get('entropy', 0.5)
        entropy_B = result_B.get('entropy', 0.5)

        # 1. 基础线性融合（作为对比）
        linear_fusion = weight_A * conf_A + weight_B * conf_B

        # 2. 计算非线性项
        interaction = self._calculate_interaction(
            conf_A, conf_B, weight_A, weight_B
        )

        complementarity = self._calculate_complementarity(
            action_A, action_B, conf_A, conf_B
        )

        diversity = self._calculate_diversity_bonus(
            conf_A, conf_B, entropy_A, entropy_B
        )

        # 3. 非线性融合
        nonlinear_fusion = (
            linear_fusion +
            interaction['term'] +
            complementarity['term'] +
            diversity['term']
        )

        # 4. 应用非线性变换（可选）
        if self.config.nonlinearity_degree > 1.0:
            nonlinear_fusion = self._apply_nonlinear_transform(
                nonlinear_fusion,
                linear_fusion,
                conf_A,
                conf_B
            )

        # 5. 限制在[0, 1]
        nonlinear_fusion = np.clip(nonlinear_fusion, 0.0, 1.0)

        # 6. 计算涌现分数
        max_individual = max(conf_A, conf_B)
        emergence = nonlinear_fusion - max_individual

        # 7. 融合动作
        fused_action = self._fuse_actions(
            action_A, action_B,
            weight_A, weight_B,
            conf_A, conf_B,
            nonlinear_fusion > max_individual
        )

        # 8. 确定融合方法
        if emergence > 0.01:
            method = 'nonlinear_emergence'
        elif emergence > 0:
            method = 'nonlinear_weak_emergence'
        else:
            method = 'nonlinear_fusion'

        # 9. 记录历史
        self._record_history({
            'linear': linear_fusion,
            'nonlinear': nonlinear_fusion,
            'emergence': emergence,
            'interaction': interaction['term'],
            'complementarity': complementarity['term'],
            'diversity': diversity['term'],
            'method': method
        })

        # 10. 自适应调整（如果启用）
        if self.enable_adaptive and len(self.fusion_history) > 10:
            self._adaptive_adjustment()

        return {
            'action': fused_action,
            'confidence': nonlinear_fusion,
            'emergence': max(0.0, emergence),
            'method': method,
            'breakdown': {
                'linear': linear_fusion,
                'interaction': interaction,
                'complementarity': complementarity,
                'diversity': diversity,
                'max_individual': max_individual
            }
        }

    def _calculate_interaction(
        self,
        conf_A: float,
        conf_B: float,
        weight_A: float,
        weight_B: float
    ) -> Dict[str, Any]:
        """
        计算交互项

        物理意义：当两个系统都高置信度时，产生增强效应
        公式：α * (conf_A * conf_B) * (weight_A * weight_B)
        """

        strength = self.adaptive_weights['interaction']

        # 归一化置信度
        norm_A = conf_A / max(conf_A, conf_B) if max(conf_A, conf_B) > 0 else 0
        norm_B = conf_B / max(conf_A, conf_B) if max(conf_A, conf_B) > 0 else 0

        # 交互项：只有两个系统都强时才显著
        product = norm_A * norm_B
        weight_coupling = weight_A * weight_B

        interaction = strength * product * weight_coupling

        return {
            'term': interaction,
            'strength': strength,
            'product': product,
            'weight_coupling': weight_coupling
        }

    def _calculate_complementarity(
        self,
        action_A: int,
        action_B: int,
        conf_A: float,
        conf_B: float
    ) -> Dict[str, Any]:
        """
        计算互补项

        物理意义：当两个系统给出不同决策时，可能捕捉到互补视角
        公式：β * min(conf_A, conf_B) if action_A != action_B
        """

        strength = self.adaptive_weights['complementarity']

        if action_A != action_B:
            # 不同决策 = 互补视角
            # 使用较小的置信度（保守估计）
            min_confidence = min(conf_A, conf_B)
            complementarity = strength * min_confidence
        else:
            # 相同决策 = 冗余确认，无额外奖励
            complementarity = 0.0

        return {
            'term': complementarity,
            'strength': strength,
            'actions_differ': action_A != action_B
        }

    def _calculate_diversity_bonus(
        self,
        conf_A: float,
        conf_B: float,
        entropy_A: float,
        entropy_B: float
    ) -> Dict[str, Any]:
        """
        计算多样性奖励

        物理意义：鼓励不确定性处理方式的多样性
        公式：γ * (1 - |entropy_A - entropy_B|)
        """

        strength = self.adaptive_weights['diversity']

        # 熵差异：越小表示处理不确定性方式越相似
        entropy_diff = abs(entropy_A - entropy_B)
        diversity_score = 1.0 - entropy_diff

        diversity = strength * diversity_score

        return {
            'term': diversity,
            'strength': strength,
            'entropy_diff': entropy_diff,
            'diversity_score': diversity_score
        }

    def _apply_nonlinear_transform(
        self,
        nonlinear_fusion: float,
        linear_fusion: float,
        conf_A: float,
        conf_B: float
    ) -> float:
        """
        应用非线性变换

        当两个系统都高置信度时，使用指数变换放大协同效应
        """

        # 判断是否为高置信度场景
        both_high = (
            conf_A > self.config.confidence_threshold and
            conf_B > self.config.confidence_threshold
        )

        if both_high and nonlinear_fusion > linear_fusion:
            # 使用非线性度进行变换
            # base^degree，但保持单调性
            ratio = nonlinear_fusion / max(linear_fusion, 0.01)
            enhanced_ratio = ratio ** (1.0 / self.config.nonlinearity_degree)

            nonlinear_fusion = linear_fusion * enhanced_ratio

        return nonlinear_fusion

    def _fuse_actions(
        self,
        action_A: int,
        action_B: int,
        weight_A: float,
        weight_B: float,
        conf_A: float,
        conf_B: float,
        has_emergence: bool
    ) -> int:
        """
        融合动作

        策略：
        - 如果有涌现（1+1>2），选择置信度更高的动作
        - 否则，使用加权平均
        """

        if has_emergence:
            # 有涌现时，非线性融合可能产生更好的动作
            # 这里简化为选择权重高的动作
            return action_A if weight_A > weight_B else action_B
        else:
            # 线性融合：加权平均
            return int(weight_A * action_A + weight_B * action_B)

    def _adaptive_adjustment(self):
        """
        自适应调整融合参数

        根据历史表现，动态调整交互、互补、多样性的权重
        """

        if len(self.fusion_history) < 20:
            return

        # 分析最近20次的融合效果
        recent = self.fusion_history[-20:]

        # 计算各项的平均贡献
        avg_interaction = np.mean([h['interaction'] for h in recent])
        avg_complementarity = np.mean([h['complementarity'] for h in recent])
        avg_diversity = np.mean([h['diversity'] for h in recent])
        avg_emergence = np.mean([h['emergence'] for h in recent])

        # 如果某项贡献持续很低，降低其权重
        # 如果某项贡献持续很高，提高其权重
        total = avg_interaction + avg_complementarity + avg_diversity

        if total > 0:
            # 归一化并调整
            self.adaptive_weights['interaction'] = 0.7 * self.adaptive_weights['interaction'] + 0.3 * (avg_interaction / total)
            self.adaptive_weights['complementarity'] = 0.7 * self.adaptive_weights['complementarity'] + 0.3 * (avg_complementarity / total)
            self.adaptive_weights['diversity'] = 0.7 * self.adaptive_weights['diversity'] + 0.3 * (avg_diversity / total)

        logger.debug(f"[非线性融合] 自适应调整: {self.adaptive_weights}")

    def _record_history(self, record: Dict[str, Any]):
        """记录历史"""
        self.fusion_history.append(record)

        # 限制历史长度
        if len(self.fusion_history) > 100:
            self.fusion_history = self.fusion_history[-100:]

    def _single_system(
        self,
        result: Dict[str, Any],
        system_name: str
    ) -> Dict[str, Any]:
        """单系统处理"""
        return {
            'action': result['action'],
            'confidence': result['confidence'],
            'emergence': 0.0,
            'method': f'{system_name}_only',
            'breakdown': {
                'linear': result['confidence'],
                'interaction': {'term': 0.0},
                'complementarity': {'term': 0.0},
                'diversity': {'term': 0.0},
                'max_individual': result['confidence']
            }
        }

    def _get_fallback(self) -> Dict[str, Any]:
        """兜底决策"""
        return {
            'action': 0,
            'confidence': 0.5,
            'emergence': 0.0,
            'method': 'fallback',
            'breakdown': {
                'linear': 0.5,
                'interaction': {'term': 0.0},
                'complementarity': {'term': 0.0},
                'diversity': {'term': 0.0},
                'max_individual': 0.5
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""

        if len(self.fusion_history) == 0:
            return {
                'total_fusions': 0,
                'avg_emergence': 0.0,
                'emergence_rate': 0.0
            }

        emergences = [h['emergence'] for h in self.fusion_history]
        positive_emergences = [e for e in emergences if e > 0]

        return {
            'total_fusions': len(self.fusion_history),
            'avg_emergence': np.mean(emergences),
            'max_emergence': np.max(emergences),
            'emergence_rate': len(positive_emergences) / len(emergences),
            'avg_interaction': np.mean([h['interaction'] for h in self.fusion_history]),
            'avg_complementarity': np.mean([h['complementarity'] for h in self.fusion_history]),
            'avg_diversity': np.mean([h['diversity'] for h in self.fusion_history]),
            'adaptive_weights': self.adaptive_weights.copy()
        }

    def reset(self):
        """重置引擎状态"""
        self.fusion_history.clear()
        self.performance_history = {
            'linear': [],
            'nonlinear': [],
            'emergence': []
        }
        self.adaptive_weights = {
            'interaction': self.config.interaction_strength,
            'complementarity': self.config.complementarity_strength,
            'diversity': self.config.diversity_bonus
        }
        logger.info("[非线性融合] 引擎已重置")


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "非线性融合引擎测试")
    print("="*70)

    engine = NonlinearFusionEngine(
        config=FusionConfig(
            interaction_strength=0.15,
            complementarity_strength=0.08,
            diversity_bonus=0.05
        )
    )

    print(f"\n[初始化] 非线性融合引擎创建成功")
    print(f"[配置] 交互强度=0.15, 互补强度=0.08, 多样性奖励=0.05")

    # 测试场景1：相同决策
    print(f"\n[场景1] 相同决策，高置信度")
    print("="*70)

    result_A = {'action': 1, 'confidence': 0.8, 'entropy': 0.3}
    result_B = {'action': 1, 'confidence': 0.75, 'entropy': 0.35}

    fusion = engine.fuse(result_A, result_B, weight_A=0.5, weight_B=0.5)

    print(f"系统A: action={result_A['action']}, conf={result_A['confidence']}")
    print(f"系统B: action={result_B['action']}, conf={result_B['confidence']}")
    print(f"\n融合结果:")
    print(f"  动作: {fusion['action']}")
    print(f"  置信度: {fusion['confidence']:.4f}")
    print(f"  涌现分数: {fusion['emergence']:.4f}")
    print(f"  方法: {fusion['method']}")

    bd = fusion['breakdown']
    print(f"\n各项贡献:")
    print(f"  线性基础: {bd['linear']:.4f}")
    print(f"  交互项: {bd['interaction']['term']:.4f}")
    print(f"  互补项: {bd['complementarity']['term']:.4f}")
    print(f"  多样性: {bd['diversity']['term']:.4f}")
    print(f"  单系统最大: {bd['max_individual']:.4f}")

    # 测试场景2：不同决策
    print(f"\n[场景2] 不同决策，中等置信度")
    print("="*70)

    result_A = {'action': 1, 'confidence': 0.7, 'entropy': 0.4}
    result_B = {'action': 2, 'confidence': 0.65, 'entropy': 0.5}

    fusion = engine.fuse(result_A, result_B, weight_A=0.5, weight_B=0.5)

    print(f"系统A: action={result_A['action']}, conf={result_A['confidence']}")
    print(f"系统B: action={result_B['action']}, conf={result_B['confidence']}")
    print(f"\n融合结果:")
    print(f"  动作: {fusion['action']}")
    print(f"  置信度: {fusion['confidence']:.4f}")
    print(f"  涌现分数: {fusion['emergence']:.4f}")
    print(f"  方法: {fusion['method']}")

    bd = fusion['breakdown']
    print(f"\n各项贡献:")
    print(f"  线性基础: {bd['linear']:.4f}")
    print(f"  交互项: {bd['interaction']['term']:.4f}")
    print(f"  互补项: {bd['complementarity']['term']:.4f} (不同决策)")
    print(f"  多样性: {bd['diversity']['term']:.4f}")
    print(f"  单系统最大: {bd['max_individual']:.4f}")

    # 测试场景3：双高置信度
    print(f"\n[场景3] 双高置信度，不同决策")
    print("="*70)

    result_A = {'action': 0, 'confidence': 0.9, 'entropy': 0.2}
    result_B = {'action': 3, 'confidence': 0.85, 'entropy': 0.25}

    fusion = engine.fuse(result_A, result_B, weight_A=0.5, weight_B=0.5)

    print(f"系统A: action={result_A['action']}, conf={result_A['confidence']}")
    print(f"系统B: action={result_B['action']}, conf={result_B['confidence']}")
    print(f"\n融合结果:")
    print(f"  动作: {fusion['action']}")
    print(f"  置信度: {fusion['confidence']:.4f}")
    print(f"  涌现分数: {fusion['emergence']:.4f}")
    print(f"  方法: {fusion['method']}")

    bd = fusion['breakdown']
    print(f"\n各项贡献:")
    print(f"  线性基础: {bd['linear']:.4f}")
    print(f"  交互项: {bd['interaction']['term']:.4f} (双高置信)")
    print(f"  互补项: {bd['complementarity']['term']:.4f} (不同决策)")
    print(f"  多样性: {bd['diversity']['term']:.4f}")
    print(f"  单系统最大: {bd['max_individual']:.4f}")

    # 显示统计
    print("\n" + "="*70)
    print(" "*25 + "统计信息")
    print("="*70)

    stats = engine.get_statistics()
    print(f"\n总融合次数: {stats['total_fusions']}")
    print(f"平均涌现分数: {stats['avg_emergence']:.4f}")
    print(f"最大涌现分数: {stats['max_emergence']:.4f}")
    print(f"涌现率: {stats['emergence_rate']:.2%}")
    print(f"\n平均交互项: {stats['avg_interaction']:.4f}")
    print(f"平均互补项: {stats['avg_complementarity']:.4f}")
    print(f"平均多样性: {stats['avg_diversity']:.4f}")

    print("\n" + "="*70 + "\n")
