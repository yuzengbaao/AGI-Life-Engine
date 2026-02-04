#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主目标生成系统 (Autonomous Goal Generation System)
====================================================

功能: 实现完全自主的目标设定，超越GoalQuestioner建议模式
版本: 1.0.0 (2026-01-19)

核心创新:
1. 内在价值函数 (Intrinsic Value Function) - 替代外部价值
2. 机会识别引擎 (Opportunity Recognition) - 自动发现目标
3. 目标层级构建 (Goal Hierarchy) - 递归分解
4. 自主性评估 (Autonomy Assessment) - 衡量目标自主程度
5. 价值函数内在化 (Value Internalization) - 长期价值学习

参考理论:
- Self-Determination Theory (SDT) - 自我决定理论
- Intrinsic Motivation - 内在动机
- Goal Setting Theory - 目标设定理论
- Value Alignment Research - 价值对齐研究
"""

import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ValueSourceType(Enum):
    """价值来源类型"""
    INTRINSIC = "intrinsic"       # 内在价值（好奇心、胜任感）
    EXTRINSIC = "extrinsic"       # 外在价值（任务、奖励）
    SOCIAL = "social"            # 社会价值（认可、协作）
    EPISTEMIC = "epistemic"       # 认知价值（知识、真理）
    CREATIVE = "creative"        # 创造价值（新颖性、创新）


@dataclass
class ValueSignal:
    """价值信号"""
    source: ValueSourceType
    magnitude: float  # 价值强度 (0-1)
    direction: str    # "approach" or "avoid"
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def compute_value(self) -> float:
        """计算综合价值"""
        if self.direction == "approach":
            return self.magnitude
        else:
            return -self.magnitude


@dataclass
class Goal:
    """目标"""
    goal_id: str
    description: str
    value: float              # 目标价值 (0-1)
    autonomy: float           # 自主性 (0-1)
    source: str               # 来源 (intrinsic/extrinsic/social)
    priority: int             # 优先级 (1-10)
    sub_goals: List['Goal'] = field(default_factory=list)
    parent_goal: Optional['Goal'] = None
    status: str = "pending"    # pending, in_progress, completed, abandoned
    created_at: float = field(default_factory=time.time)

    def add_sub_goal(self, sub_goal: 'Goal'):
        """添加子目标"""
        sub_goal.parent_goal = self
        self.sub_goals.append(sub_goal)

    def get_depth(self) -> int:
        """获取目标深度（递归）"""
        if not self.sub_goals:
            return 1
        return 1 + max(g.get_depth() for g in self.sub_goals)

    def get_total_value(self) -> float:
        """获取总价值（包含子目标）"""
        sub_values = sum(g.get_total_value() for g in self.sub_goals)
        return self.value + sub_values * 0.3  # 子目标价值折算


@dataclass
class Opportunity:
    """机会（潜在目标）"""
    opportunity_id: str
    description: str
    expected_value: float
    confidence: float        # 置信度 (0-1)
    required_resources: List[str]
    feasibility: float       # 可行性 (0-1)
    urgency: float          # 紧迫性 (0-1)
    novelty: float          # 新颖性 (0-1)

    def compute_opportunity_score(self) -> float:
        """计算机会得分"""
        # 加权组合
        score = (
            0.3 * self.expected_value +
            0.2 * self.confidence +
            0.2 * self.feasibility +
            0.15 * self.urgency +
            0.15 * self.novelty
        )
        return score


class IntrinsicValueFunction:
    """
    内在价值函数

    基于Self-Determination Theory (SDT):
    - Autonomy (自主性): 感到行为自主可控
    - Competence (胜任感): 感到能力提升
    - Relatedness (关联性): 感到与系统目标关联
    """

    def __init__(self):
        # 价值权重（可学习）
        self.value_weights = {
            'curiosity': 0.35,      # 好奇心驱动
            'competence': 0.30,     # 能力提升
            'autonomy': 0.20,       # 自主性
            'creativity': 0.15      # 创造性
        }

        # 历史价值记录（用于学习）
        self.value_history: deque = deque(maxlen=1000)

        # 价值函数参数
        self.curiosity_decay = 0.95      # 好奇心衰减
        self.competence_threshold = 0.6   # 胜任感阈值
        self.autonomy_threshold = 0.5     # 自主性阈值

    def compute_value(self, state: Dict[str, Any]) -> float:
        """
        计算内在价值

        Args:
            state: 系统状态，包含：
                - curiosity: 当前好奇心 (0-1)
                - competence: 当前能力感 (0-1)
                - autonomy: 当前自主性 (0-1)
                - creativity: 创造性得分 (0-1)
                - uncertainty: 不确定性 (0-1)
                - novelty: 新颖性 (0-1)

        Returns:
            float: 内在价值 (0-1)
        """
        # 提取状态特征
        curiosity = state.get('curiosity', 0.5)
        competence = state.get('competence', 0.5)
        autonomy = state.get('autonomy', 0.5)
        creativity = state.get('creativity', 0.5)
        uncertainty = state.get('uncertainty', 0.5)
        novelty = state.get('novelty', 0.5)

        # 1. 好奇心价值：追求新颖和不确定性
        curiosity_value = (
            curiosity *
            (1 + novelty) *
            (1 + uncertainty)
        )

        # 2. 胜任感价值：追求能力提升
        competence_value = competence
        if competence > self.competence_threshold:
            competence_value *= 1.2  # 超越阈值的胜任感更宝贵

        # 3. 自主性价值：追求自主控制
        autonomy_value = autonomy
        if autonomy < self.autonomy_threshold:
            autonomy_value *= 0.7  # 低自主性降低价值

        # 4. 创造性价值：追求新颖性
        creativity_value = (
            creativity *
            (1 + novelty)
        )

        # 加权组合
        intrinsic_value = (
            self.value_weights['curiosity'] * curiosity_value +
            self.value_weights['competence'] * competence_value +
            self.value_weights['autonomy'] * autonomy_value +
            self.value_weights['creativity'] * creativity_value
        )

        # 记录历史
        self.value_history.append({
            'timestamp': time.time(),
            'value': intrinsic_value,
            'state': state
        })

        return intrinsic_value

    def update_weights(self, recent_outcomes: List[Dict]):
        """
        根据最近结果更新价值权重（学习）

        Args:
            recent_outcomes: 最近的行动结果列表
                - value_type: 价值类型 (curiosity/competence/autonomy/creativity)
                - outcome: 结果好坏 (0-1)
        """
        for outcome in recent_outcomes:
            value_type = outcome.get('value_type', 'curiosity')
            result = outcome.get('outcome', 0.5)

            # 简单的强化学习更新
            if value_type in self.value_weights:
                # 成功 -> 增加权重
                # 失败 -> 减少权重
                adjustment = 0.01 * (result - 0.5)
                self.value_weights[value_type] = np.clip(
                    self.value_weights[value_type] + adjustment,
                    0.1, 0.5  # 保持在合理范围
                )

        # 归一化权重
        total = sum(self.value_weights.values())
        for key in self.value_weights:
            self.value_weights[key] /= total

        logger.debug(f"价值权重更新: {self.value_weights}")


class OpportunityRecognitionEngine:
    """机会识别引擎"""

    def __init__(self, value_function: IntrinsicValueFunction):
        self.value_function = value_function
        self.opportunities: List[Opportunity] = []

    def identify_opportunities(self,
                                state: Dict[str, Any],
                                context: Dict[str, Any]) -> List[Opportunity]:
        """
        识别机会（潜在目标）

        Args:
            state: 当前系统状态
            context: 上下文信息

        Returns:
            识别出的机会列表
        """
        opportunities = []

        # 1. 好奇心驱动机会：探索未知领域
        if state.get('curiosity', 0) > 0.6:
            opportunities.append(Opportunity(
                opportunity_id=f"curiosity_explore_{int(time.time())}",
                description="探索未知领域以获得新知识",
                expected_value=self._compute_curiosity_value(state),
                confidence=0.7,
                required_resources=["attention", "time"],
                feasibility=0.8,
                urgency=0.3,
                novelty=0.9
            ))

        # 2. 能力提升机会：挑战略高于当前水平的任务
        competence = state.get('competence', 0.5)
        if competence < 0.9:
            opportunities.append(Opportunity(
                opportunity_id=f"competence_growth_{int(time.time())}",
                description=f"挑战当前能力边界（当前{competence:.2%}）",
                expected_value=self._compute_growth_value(state),
                confidence=0.8,
                required_resources=["effort", "learning"],
                feasibility=0.7,
                urgency=0.5,
                novelty=0.4
            ))

        # 3. 创造性机会：生成新洞察或理论
        if state.get('creativity', 0) > 0.7:
            opportunities.append(Opportunity(
                opportunity_id=f"creative_insight_{int(time.time())}",
                description="生成原创洞察或理论假设",
                expected_value=0.85,
                confidence=0.6,
                required_resources=["deep_reasoning", "knowledge"],
                feasibility=0.6,
                urgency=0.4,
                novelty=0.95
            ))

        # 4. 系统优化机会：改进系统性能
        if state.get('entropy', 0) > 0.6:
            opportunities.append(Opportunity(
                opportunity_id=f"system_optimize_{int(time.time())}",
                description="优化系统以降低熵值",
                expected_value=0.75,
                confidence=0.8,
                required_resources=["analysis", "modification"],
                feasibility=0.7,
                urgency=0.7,
                novelty=0.3
            ))

        # 5. 协作机会：多智能体协作
        if context.get('multi_agent_available', False):
            opportunities.append(Opportunity(
                opportunity_id=f"collaboration_{int(time.time())}",
                description="与其他智能体协作完成复杂任务",
                expected_value=0.70,
                confidence=0.6,
                required_resources=["communication", "coordination"],
                feasibility=0.5,
                urgency=0.4,
                novelty=0.6
            ))

        return opportunities

    def _compute_curiosity_value(self, state: Dict) -> float:
        """计算好奇心驱动价值"""
        curiosity = state.get('curiosity', 0.5)
        novelty = state.get('novelty', 0.5)
        return curiosity * (1 + novelty)

    def _compute_growth_value(self, state: Dict) -> float:
        """计算成长价值"""
        competence = state.get('competence', 0.5)
        # 边际效应：能力越低，成长价值越高
        return (1 - competence) * 1.2


class AutonomousGoalGenerator:
    """
    自主目标生成器

    核心功能：
    1. 识别机会（潜在目标）
    2. 评估机会价值
    3. 选择最佳目标
    4. 构建目标层级
    """

    def __init__(self):
        self.value_function = IntrinsicValueFunction()
        self.opportunity_engine = OpportunityRecognitionEngine(self.value_function)
        self.goal_history: List[Goal] = []

        # 目标生成统计
        self.stats = {
            'goals_generated': 0,
            'intrinsic_goals': 0,
            'extrinsic_goals': 0,
            'avg_goal_value': 0.0,
            'avg_autonomy': 0.0
        }

        logger.info("🎯 AutonomousGoalGenerator initialized")

    def generate_goal(self,
                      state: Dict[str, Any],
                      context: Dict[str, Any]) -> Optional[Goal]:
        """
        自主生成目标

        Args:
            state: 当前系统状态
            context: 上下文信息

        Returns:
            生成的目标，若无合适机会则返回None
        """
        # 1. 识别机会
        opportunities = self.opportunity_engine.identify_opportunities(state, context)

        if not opportunities:
            logger.debug("未识别到任何机会")
            return None

        # 2. 评估机会
        scored_opportunities = []
        for opp in opportunities:
            score = opp.compute_opportunity_score()
            scored_opportunities.append((opp, score))

        # 3. 选择最佳机会
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)
        best_opportunity, best_score = scored_opportunities[0]

        # 如果最佳机会得分太低，不生成目标
        if best_score < 0.4:
            logger.debug(f"最佳机会得分过低: {best_score:.2f}")
            return None

        # 4. 生成目标
        goal = Goal(
            goal_id=f"goal_{int(time.time() * 1000)}",
            description=best_opportunity.description,
            value=best_opportunity.expected_value,
            autonomy=0.8,  # 自主生成的目标，高自主性
            source="intrinsic",
            priority=int(best_opportunity.urgency * 10)
        )

        # 5. 记录统计
        self.stats['goals_generated'] += 1
        self.stats['intrinsic_goals'] += 1
        self.stats['avg_goal_value'] = (
            (self.stats['avg_goal_value'] * (self.stats['goals_generated'] - 1) + goal.value) /
            self.stats['goals_generated']
        )
        self.stats['avg_autonomy'] = (
            (self.stats['avg_autonomy'] * (self.stats['goals_generated'] - 1) + goal.autonomy) /
            self.stats['goals_generated']
        )

        self.goal_history.append(goal)

        logger.info(f"🎯 自主生成目标: {goal.description} (价值={goal.value:.2f}, 自主性={goal.autonomy:.2f})")

        return goal

    def generate_goal_hierarchy(self,
                               root_goal: Goal,
                               max_depth: int = 3) -> Goal:
        """
        构建目标层级（递归分解）

        Args:
            root_goal: 根目标
            max_depth: 最大层级深度

        Returns:
            构建好的目标层级树
        """
        if max_depth <= 0 or root_goal.get_depth() >= max_depth:
            return root_goal

        # 根据目标类型生成子目标
        sub_goals = self._decompose_goal(root_goal)

        for sub_goal_desc in sub_goals:
            sub_goal = Goal(
                goal_id=f"subgoal_{int(time.time() * 1000)}_{len(root_goal.sub_goals)}",
                description=sub_goal_desc['description'],
                value=sub_goal_desc['value'] * 0.8,  # 子目标价值略低于父目标
                autonomy=root_goal.autonomy * 0.9,
                source="intrinsic",
                priority=max(1, root_goal.priority - 1)
            )

            # 递归构建子目标层级
            self.generate_goal_hierarchy(sub_goal, max_depth - 1)
            root_goal.add_sub_goal(sub_goal)

        return root_goal

    def _decompose_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """
        分解目标为子目标

        Args:
            goal: 要分解的目标

        Returns:
            子目标描述列表
        """
        # 根据目标类型进行分解
        sub_goals = []

        if "探索" in goal.description:
            sub_goals = [
                {"description": "确定探索方向", "value": 0.8},
                {"description": "收集相关信息", "value": 0.7},
                {"description": "分析与整合", "value": 0.9}
            ]

        elif "优化" in goal.description:
            sub_goals = [
                {"description": "识别优化目标", "value": 0.7},
                {"description": "分析当前瓶颈", "value": 0.8},
                {"description": "设计优化方案", "value": 0.9},
                {"description": "实施优化", "value": 0.8}
            ]

        elif "洞察" in goal.description or "理论" in goal.description:
            sub_goals = [
                {"description": "收集相关数据", "value": 0.6},
                {"description": "深度推理分析", "value": 0.9},
                {"description": "生成假设", "value": 0.95},
                {"description": "验证假设", "value": 0.8}
            ]

        elif "协作" in goal.description:
            sub_goals = [
                {"description": "识别协作伙伴", "value": 0.7},
                {"description": "定义协作任务", "value": 0.8},
                {"description": "建立通信", "value": 0.6},
                {"description": "执行协作", "value": 0.9}
            ]

        elif "挑战" in goal.description or "成长" in goal.description:
            sub_goals = [
                {"description": "评估当前能力", "value": 0.6},
                {"description": "选择挑战任务", "value": 0.8},
                {"description": "执行任务", "value": 0.9},
                {"description": "反思与总结", "value": 0.85}
            ]

        else:
            # 通用分解
            sub_goals = [
                {"description": "明确目标要求", "value": 0.7},
                {"description": "制定执行计划", "value": 0.75},
                {"description": "执行计划", "value": 0.8},
                {"description": "验证结果", "value": 0.75}
            ]

        return sub_goals

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'goal_history_length': len(self.goal_history),
            'intrinsic_ratio': (
                self.stats['intrinsic_goals'] / max(1, self.stats['goals_generated'])
            )
        }


# ==================== 辅助函数 ====================

def print_goal_tree(goal: Goal, indent: int):
    """打印目标树"""
    prefix = "  " * indent
    print(f"{prefix}● {goal.description}")
    print(f"{prefix}  价值={goal.value:.2f}, 自主性={goal.autonomy:.2f}")

    for sub_goal in goal.sub_goals:
        print_goal_tree(sub_goal, indent + 1)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("自主目标生成系统测试")
    print("=" * 70)

    # 创建生成器
    generator = AutonomousGoalGenerator()

    # 模拟系统状态
    test_state = {
        'curiosity': 0.75,
        'competence': 0.60,
        'autonomy': 0.50,
        'creativity': 0.80,
        'uncertainty': 0.65,
        'novelty': 0.70,
        'entropy': 0.65
    }

    test_context = {
        'multi_agent_available': False
    }

    print("\n测试1: 自主目标生成")
    print("-" * 70)

    # 生成目标
    goal = generator.generate_goal(test_state, test_context)

    if goal:
        print(f"✅ 目标生成成功")
        print(f"   描述: {goal.description}")
        print(f"   价值: {goal.value:.2f}")
        print(f"   自主性: {goal.autonomy:.2f}")
        print(f"   来源: {goal.source}")
        print(f"   优先级: {goal.priority}")

    print("\n测试2: 目标层级构建")
    print("-" * 70)

    if goal:
        # 构建目标层级
        hierarchy = generator.generate_goal_hierarchy(goal, max_depth=2)

        print(f"✅ 目标层级构建完成 (深度: {hierarchy.get_depth()})")
        print("\n目标层级树:")
        print_goal_tree(hierarchy, indent=0)

    print("\n测试3: 统计信息")
    print("-" * 70)

    stats = generator.get_statistics()
    print(f"生成目标数: {stats['goals_generated']}")
    print(f"内在目标数: {stats['intrinsic_goals']}")
    print(f"平均目标价值: {stats['avg_goal_value']:.2f}")
    print(f"平均自主性: {stats['avg_autonomy']:.2f}")
    print(f"内在目标比例: {stats['intrinsic_ratio']:.2%}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")


def print_goal_tree(goal: Goal, indent: int):
    """打印目标树"""
    prefix = "  " * indent
    print(f"{prefix}● {goal.description}")
    print(f"{prefix}  价值={goal.value:.2f}, 自主性={goal.autonomy:.2f}")

    for sub_goal in goal.sub_goals:
        print_goal_tree(sub_goal, indent + 1)
