#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Action Space - 动态动作空间生成器
=========================================

功能：
1. 基于任务需求动态扩展动作空间维度
2. 复合动作生成
3. 分层动作空间（基础/复合/抽象/元）
4. 动作有效性验证

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ActionLevel(Enum):
    """动作层级"""
    PRIMITIVE = "primitive"      # 基础动作（原始动作）
    COMPOSITE = "composite"      # 复合动作（基础动作组合）
    ABSTRACT = "abstract"        # 抽象动作（复合动作组合）
    META = "meta"                # 元动作（抽象动作组合）


@dataclass
class ActionVector:
    """动作向量"""
    vector: np.ndarray
    level: ActionLevel
    source_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicActionSpace:
    """
    动态动作空间生成器

    基于任务需求动态扩展动作空间维度
    """

    def __init__(self, base_action_dim: int = 4):
        """
        初始化动态动作空间

        Args:
            base_action_dim: 基础动作维度（默认4）
        """
        self.base_action_dim = base_action_dim
        self.current_action_dim = base_action_dim

        # 分层动作空间
        self.action_levels = {
            ActionLevel.PRIMITIVE: base_action_dim,      # 4维基础
            ActionLevel.COMPOSITE: base_action_dim * 3,  # 12维复合
            ActionLevel.ABSTRACT: base_action_dim * 9,    # 36维抽象
            ActionLevel.META: base_action_dim * 27       # 108维元
        }

        # 基础动作矩阵
        self.base_actions = np.eye(base_action_dim)

        # 动作历史
        self.action_history: List[ActionVector] = []

        logger.info(
            f"[动作空间] 初始化: "
            f"基础={base_action_dim}维, "
            f"复合={self.action_levels[ActionLevel.COMPOSITE]}维"
        )

    def expand_action_space(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        基于任务需求动态扩展动作空间

        Args:
            context: 任务上下文

        Returns:
            扩展后的动作空间
        """
        if context is None:
            context = {}

        novelty_required = context.get('novelty_required', 0.0)
        complexity = context.get('task_complexity', 0.5)

        # 动态扩展因子
        if novelty_required > 0.8 and complexity > 0.7:
            expansion_factor = 3.0  # 高新颖性+高复杂度→3倍扩展
        elif novelty_required > 0.5:
            expansion_factor = 2.0  # 中等新颖性→2倍扩展
        else:
            expansion_factor = 1.5  # 保守扩展

        # 计算新维度
        old_dim = self.current_action_dim
        new_dim = int(self.base_action_dim * expansion_factor)
        new_dim = max(new_dim, 4)  # 至少4维

        # 生成复合动作
        expanded_actions = self._generate_composite_actions(
            old_dim=old_dim,
            new_dim=new_dim
        )

        self.current_action_dim = new_dim

        logger.info(
            f"[动作空间] 扩展: "
            f"{old_dim}维 → {new_dim}维 "
            f"(因子={expansion_factor:.1f})"
        )

        return expanded_actions

    def _generate_composite_actions(
        self,
        old_dim: int,
        new_dim: int
    ) -> np.ndarray:
        """
        生成复合动作

        Args:
            old_dim: 原始维度
            new_dim: 目标维度

        Returns:
            复合动作矩阵
        """
        # 基础动作保持（前old_dim维）
        base_actions = np.eye(old_dim)

        # 扩展部分：生成复合动作
        num_new_actions = new_dim - old_dim
        if num_new_actions <= 0:
            return base_actions

        composite_actions = []

        for i in range(num_new_actions):
            # 随机组合基础动作
            weights = np.random.dirichlet(np.ones(old_dim))
            composite_action = np.dot(weights, base_actions)

            # 添加轻微随机噪声
            noise = np.random.normal(0, 0.1, size=old_dim)
            composite_action = composite_action + noise

            # 归一化
            composite_action = composite_action / (np.linalg.norm(composite_action) + 1e-8)

            composite_actions.append(composite_action)

        # 合并基础动作和复合动作
        if len(composite_actions) > 0:
            return np.vstack([base_actions, composite_actions])
        else:
            return base_actions

    def get_action_space(self, level: ActionLevel) -> int:
        """
        获取指定层级的动作空间维度

        Args:
            level: 动作层级

        Returns:
            动作空间维度
        """
        return self.action_levels.get(level, self.base_action_dim)

    def sample_action(
        self,
        level: ActionLevel = ActionLevel.PRIMITIVE,
        context: Optional[Dict[str, Any]] = None
    ) -> ActionVector:
        """
        采样动作

        Args:
            level: 动作层级
            context: 上下文

        Returns:
            动作向量
        """
        dim = self.get_action_space(level)

        # 根据层级生成动作
        if level == ActionLevel.PRIMITIVE:
            # 基础动作：单位向量
            idx = random.randint(0, self.base_action_dim - 1)
            vector = np.zeros(self.base_action_dim)
            vector[idx] = 1.0
            source = [f"primitive_{idx}"]

        elif level == ActionLevel.COMPOSITE:
            # 复合动作：基础动作加权和
            weights = np.random.dirichlet(np.ones(self.base_action_dim))
            vector = weights
            source = [f"composite_weights"]

        elif level == ActionLevel.ABSTRACT:
            # 抽象动作：复合动作组合
            composite1 = np.random.dirichlet(np.ones(self.base_action_dim))
            composite2 = np.random.dirichlet(np.ones(self.base_action_dim))
            vector = (composite1 + composite2) / 2
            source = ["abstract_composition"]

        else:  # META
            # 元动作：多个抽象动作的组合
            abstract_actions = [
                np.random.dirichlet(np.ones(self.base_action_dim))
                for _ in range(3)
            ]
            vector = np.mean(abstract_actions, axis=0)
            source = ["meta_combination"]

        action = ActionVector(
            vector=vector,
            level=level,
            source_components=source
        )

        # 记录历史
        self.action_history.append(action)

        return action

    def promote_action(
        self,
        action: ActionVector,
        from_level: ActionLevel,
        to_level: ActionLevel
    ) -> ActionVector:
        """
        将动作从低层级提升到高层级

        Args:
            action: 原始动作
            from_level: 原始层级
            to_level: 目标层级

        Returns:
        提升后的动作
        """
        level_mapping = {
            ActionLevel.PRIMITIVE: 0,
            ActionLevel.COMPOSITE: 1,
            ActionLevel.ABSTRACT: 2,
            ActionLevel.META: 3
        }

        from_idx = level_mapping[from_level]
        to_idx = level_mapping[to_level]

        if to_idx <= from_idx:
            return action  # 已经是高层级或相同层级

        # 提升动作：组合多个副本
        promoted_vectors = [action.vector.copy() for _ in range(to_idx)]

        # 添加轻微变化
        for i in range(1, len(promoted_vectors)):
            noise = np.random.normal(0, 0.05, size=action.vector.shape)
            promoted_vectors[i] = promoted_vectors[i] + noise

        # 合并
        promoted_vector = np.mean(promoted_vectors, axis=0)
        promoted_vector = promoted_vector / (np.linalg.norm(promoted_vector) + 1e-8)

        return ActionVector(
            vector=promoted_vector,
            level=to_level,
            source_components=action.source_components + [f"promoted_from_{from_level.value}"],
            metadata=action.metadata.copy()
        )

    def validate_action(
        self,
        action: ActionVector,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        验证动作有效性

        Args:
            action: 动作向量
            constraints: 约束条件

        Returns:
            (是否有效, 错误信息)
        """
        # 检查向量范数
        norm = np.linalg.norm(action.vector)
        if norm > 10.0:
            return False, f"向量范数过大: {norm:.2f}"

        if norm < 0.01:
            return False, f"向量范数过小: {norm:.4f}"

        # 检查维度
        if len(action.vector) != self.current_action_dim:
            return False, f"维度不匹配: {len(action.vector)} != {self.current_action_dim}"

        # 检查数值有效性
        if np.any(np.isnan(action.vector)):
            return False, "包含NaN值"

        if np.any(np.isinf(action.vector)):
            return False, "包含无穷大值"

        return True, "有效"

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.action_history:
            return {
                'base_dim': self.base_action_dim,
                'current_dim': self.current_action_dim,
                'total_actions': 0,
                'by_level': {}
            }

        level_counts = {}
        for action in self.action_history:
            level = action.level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            'base_dim': self.base_action_dim,
            'current_dim': self.current_action_dim,
            'total_actions': len(self.action_history),
            'by_level': level_counts,
            'expansion_factor': self.current_action_dim / self.base_action_dim
        }


class HierarchicalActionSpace:
    """
    分层动作空间

    支持基础/复合/抽象/元四个层级
    """

    def __init__(self, base_dim: int = 4):
        """
        初始化分层动作空间

        Args:
            base_dim: 基础维度
        """
        self.base_dim = base_dim

        self.action_levels = {
            ActionLevel.PRIMITIVE: base_dim,     # 4维
            ActionLevel.COMPOSITE: base_dim * 3,   # 12维
            ActionLevel.ABSTRACT: base_dim * 9,     # 36维
            ActionLevel.META: base_dim * 27        # 108维
        }

        logger.info(
            f"[分层动作空间] 初始化: "
            f"基础={base_dim}维, "
            f"复合={self.action_levels[ActionLevel.COMPOSITE]}维, "
            f"抽象={self.action_levels[ActionLevel.ABSTRACT]}维, "
            f"元={self.action_levels[ActionLevel.META]}维"
        )

    def get_action_space(self, level: ActionLevel) -> int:
        """获取指定层级的动作空间维度"""
        return self.action_levels.get(level, self.base_dim)

    def get_promotion_path(
        self,
        from_level: ActionLevel,
        to_level: ActionLevel
    ) -> List[ActionLevel]:
        """
        获取提升路径

        Args:
            from_level: 起始层级
            to_level: 目标层级

        Returns:
            经过的层级列表
        """
        level_order = [
            ActionLevel.PRIMITIVE,
            ActionLevel.COMPOSITE,
            ActionLevel.ABSTRACT,
            ActionLevel.META
        ]

        try:
            start_idx = level_order.index(from_level)
            end_idx = level_order.index(to_level)
            return level_order[start_idx:end_idx + 1]
        except ValueError:
            return [from_level]

    def calculate_effective_dimension(self, context: Dict[str, Any]) -> int:
        """
        计算有效动作维度（基于上下文）

        Args:
            context: 任务上下文

        Returns:
            有效维度
        """
        base_dim = self.base_dim

        # 新颖性要求
        novelty = context.get('novelty_required', 0.5)
        if novelty > 0.8:
            multiplier = 3.0
        elif novelty > 0.5:
            multiplier = 2.0
        else:
            multiplier = 1.5

        effective_dim = int(base_dim * multiplier)

        # 限制在元层级
        max_dim = self.action_levels[ActionLevel.META]
        effective_dim = min(effective_dim, max_dim)

        return effective_dim


# 全局单例
_global_action_space: Optional[DynamicActionSpace] = None
_global_hierarchical: Optional[HierarchicalActionSpace] = None


def get_dynamic_action_space() -> DynamicActionSpace:
    """获取全局动态动作空间"""
    global _global_action_space
    if _global_action_space is None:
        _global_action_space = DynamicActionSpace()
    return _global_action_space


def get_hierarchical_action_space() -> HierarchicalActionSpace:
    """获取全局分层动作空间"""
    global _global_hierarchical
    if _global_hierarchical is None:
        _global_hierarchical = HierarchicalActionSpace()
    return _global_hierarchical
