#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层级目标系统（Hierarchical Goal Manager）
=========================================

功能：实现层级化的目标管理，支持自主目标设定
基于：层级任务网络 (HTN) + 目标层级理论

核心能力：
1. 目标层级结构（终身/长期/中期/短期/即时）
2. 目标分解与规划
3. 目标冲突检测
4. 自主目标生成

版本: 1.0.0
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


class GoalLevel(Enum):
    """目标层级"""
    LIFETIME = "lifetime"       # 终身目标（年）
    LONG_TERM = "long_term"     # 长期目标（月）
    MEDIUM_TERM = "medium_term" # 中期目标（周）
    SHORT_TERM = "short_term"   # 短期目标（日）
    IMMEDIATE = "immediate"     # 即时目标（Tick）


class GoalStatus(Enum):
    """目标状态"""
    PENDING = "pending"         # 待处理
    ACTIVE = "active"           # 进行中
    BLOCKED = "blocked"         # 被阻塞
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 已失败
    CANCELLED = "cancelled"     # 已取消


@dataclass
class Goal:
    """目标"""
    id: str
    name: str
    level: GoalLevel
    status: GoalStatus
    description: str
    priority: float  # 优先级 [0, 1]
    created_at: float
    updated_at: float
    parent_id: Optional[str] = None  # 父目标ID
    children: List[str] = field(default_factory=list)  # 子目标ID列表
    progress: float = 0.0  # 进度 [0, 1]
    success_criteria: Optional[str] = None  # 成功标准
    deadline: Optional[float] = None  # 截止时间
    metadata: Dict[str, Any] = field(default_factory=dict)
    _goal_type: Optional[str] = None  # 向后兼容：旧的goal_type字段

    def __repr__(self):
        return f"Goal({self.name}, level={self.level.value}, status={self.status.value}, progress={self.progress:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value if hasattr(self.level, "value") else str(self.level),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "description": self.description,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_id": self.parent_id,
            "children": self.children,
            "progress": self.progress,
            "success_criteria": self.success_criteria,
            "deadline": self.deadline,
            "metadata": self.metadata
        }

    @property
    def goal_type(self):
        """向后兼容：返回旧的goal_type枚举值"""
        if self._goal_type:
            return self._goal_type

        # 将新的GoalLevel映射到旧的GoalType值
        level_to_type = {
            GoalLevel.LIFETIME: 'LIFETIME',
            GoalLevel.LONG_TERM: 'LONG_TERM',
            GoalLevel.MEDIUM_TERM: 'MEDIUM_TERM',
            GoalLevel.SHORT_TERM: 'SHORT_TERM',
            GoalLevel.IMMEDIATE: 'IMMEDIATE',
        }
        # 创建一个简单的枚举类用于兼容
        class LegacyGoalType:
            value = level_to_type.get(self.level, 'CUSTOM')

        return LegacyGoalType()


@dataclass
class GoalConflict:
    """目标冲突"""
    conflict_id: str
    goal1_id: str
    goal2_id: str
    conflict_type: str  # resource, temporal, logical
    severity: float  # 严重程度 [0, 1]
    description: str
    detected_at: float


class HierarchicalGoalManager:
    """
    层级目标管理器

    核心功能：
    1. 维护目标层级结构
    2. 分解高層目标到低层
    3. 检测目标冲突
    4. 自主生成新目标
    """

    def __init__(self, max_active_goals: int = 10):
        """
        初始化目标管理器

        Args:
            max_active_goals: 最大同时活动目标数
        """
        self.max_active_goals = max_active_goals

        # 目标存储
        self.goals: Dict[str, Goal] = {}

        # 目标层级索引
        self.hierarchy: Dict[GoalLevel, List[str]] = {
            level: [] for level in GoalLevel
        }

        # 冲突记录
        self.conflicts: List[GoalConflict] = []

        # 统计信息
        self.stats = {
            'total_goals_created': 0,
            'total_goals_completed': 0,
            'total_conflicts_detected': 0,
            'total_goals_generated': 0
        }

    def create_goal(self, name=None, level=None, description: str = "",
                    priority: float = 0.5, parent_id: Optional[str] = None,
                    success_criteria: Optional[str] = None,
                    deadline: Optional[float] = None, **kwargs) -> Goal:
        """
        创建新目标（支持新旧两种API格式）

        新API格式：
            create_goal(name, level, description, priority=0.5, ...)

        旧API格式（向后兼容）：
            create_goal(description, goal_type=GoalType.CUSTOM, priority="critical")

        Args:
            name: 目标名称（新API）
            level: 目标层级（新API）
            description: 目标描述
            priority: 优先级（支持字符串或浮点数）
            parent_id: 父目标ID
            success_criteria: 成功标准
            deadline: 截止时间
            **kwargs: 其他参数（支持旧API的goal_type等）

        Returns:
            创建的目标
        """
        # 检测是否为旧API调用
        # 如果description是第一个位置参数且name和level都没有明确指定
        if 'goal_type' in kwargs or (isinstance(priority, str) and priority in ['critical', 'high', 'medium', 'low']):
            # 旧API调用模式
            # 提取goal_type并从kwargs中移除，避免重复传递
            goal_type = kwargs.pop('goal_type', None)
            return self._create_goal_legacy(description, goal_type, priority, success_criteria, **kwargs)

        # 新API调用模式
        if name is None:
            name = description[:30] if description else "unnamed_goal"
        if level is None:
            level = GoalLevel.SHORT_TERM

        # 处理字符串优先级
        if isinstance(priority, str):
            priority_mapping = {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.5,
                'low': 0.2,
            }
            priority = priority_mapping.get(priority.lower(), 0.5)

        return self._create_goal_new(name, level, description, priority, parent_id, success_criteria, deadline)

    def _create_goal_new(self, name: str, level: GoalLevel, description: str,
                        priority: float, parent_id: Optional[str],
                        success_criteria: Optional[str], deadline: Optional[float]) -> Goal:
        """新API的内部实现"""
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        current_time = time.time()

        goal = Goal(
            id=goal_id,
            name=name,
            level=level,
            status=GoalStatus.PENDING,
            description=description,
            priority=priority,
            created_at=current_time,
            updated_at=current_time,
            parent_id=parent_id,
            success_criteria=success_criteria,
            deadline=deadline
        )

        # 添加到存储
        self.goals[goal_id] = goal
        self.hierarchy[level].append(goal_id)

        # 如果有父目标，添加到父目标的children
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].children.append(goal_id)

        self.stats['total_goals_created'] += 1

        print(f"  [GoalManager] Created goal: {goal}")

        return goal

    def _create_goal_legacy(self, description: str, goal_type, priority, success_criteria, **kwargs) -> Goal:
        """旧API的内部实现"""
        # 映射旧GoalType到新GoalLevel
        level_mapping = {
            'lifetime': GoalLevel.LIFETIME,
            'long_term': GoalLevel.LONG_TERM,
            'medium_term': GoalLevel.MEDIUM_TERM,
            'short_term': GoalLevel.SHORT_TERM,
            'immediate': GoalLevel.IMMEDIATE,
            # 旧的GoalType映射
            'OBSERVATION': GoalLevel.SHORT_TERM,
            'CUSTOM': GoalLevel.IMMEDIATE,
            'EXPLORATION': GoalLevel.SHORT_TERM,
        }

        # 如果goal_type是枚举，获取其值
        goal_type_str = goal_type.value if hasattr(goal_type, 'value') else str(goal_type) if goal_type else 'CUSTOM'

        # 确定目标层级
        level = level_mapping.get(goal_type_str, GoalLevel.IMMEDIATE)

        # 映射优先级字符串到浮点数
        priority_mapping = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2,
        }

        if isinstance(priority, str):
            priority_float = priority_mapping.get(priority.lower(), 0.5)
        else:
            priority_float = float(priority)

        # 从description生成name（取前30个字符）
        name = description.replace("User Command: ", "")[:30]

        # 调用新API创建目标
        return self._create_goal_new(name, level, description, priority_float, None, success_criteria, None)

    # ============ 兼容性方法 ============
    
    def start_goal(self, goal: Goal) -> bool:
        """Compatibility method for starting a goal"""
        if goal.id not in self.goals:
            self.goals[goal.id] = goal
        return self.activate_goal(goal.id)

    def abandon_goal(self, goal: Goal, reason: str = "Abandoned") -> bool:
        """Compatibility method for abandoning a goal"""
        if not goal or goal.id not in self.goals:
            return False
        
        goal.status = GoalStatus.CANCELLED
        goal.metadata['abandon_reason'] = reason
        print(f"  [GoalManager] Abandoned goal: {goal.name} ({reason})")
        return True

    def get_current_goal(self) -> Optional[Goal]:
        """Compatibility method to get the single most important active goal"""
        active = self.get_active_goals()
        if not active:
            return None
        # get_active_goals returns sorted by priority, so first is best
        return active[0]

    def create_goal_legacy(self, description: str, goal_type=None, priority="medium",
                          success_criteria=None, **kwargs) -> Goal:
        """
        向后兼容的目标创建方法（支持旧API）

        旧API调用格式：
            create_goal(description, goal_type=GoalType.CUSTOM, priority="critical")

        新API调用格式：
            create_goal(name, level=GoalLevel.SHORT_TERM, description, priority=0.8)

        Args:
            description: 目标描述
            goal_type: 旧的GoalType枚举（映射到新的GoalLevel）
            priority: 优先级字符串或浮点数
            success_criteria: 成功标准

        Returns:
            创建的目标
        """
        # 映射旧GoalType到新GoalLevel
        level_mapping = {
            'lifetime': GoalLevel.LIFETIME,
            'long_term': GoalLevel.LONG_TERM,
            'medium_term': GoalLevel.MEDIUM_TERM,
            'short_term': GoalLevel.SHORT_TERM,
            'immediate': GoalLevel.IMMEDIATE,
            # 旧的GoalType映射
            'OBSERVATION': GoalLevel.SHORT_TERM,
            'CUSTOM': GoalLevel.IMMEDIATE,
            'EXPLORATION': GoalLevel.SHORT_TERM,
        }

        # 如果goal_type是枚举，获取其值
        goal_type_str = goal_type.value if hasattr(goal_type, 'value') else str(goal_type)

        # 确定目标层级
        if goal_type_str in level_mapping:
            level = level_mapping[goal_type_str]
        else:
            # 默认使用SHORT_TERM
            level = GoalLevel.SHORT_TERM

        # 映射优先级字符串到浮点数
        priority_mapping = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2,
        }

        if isinstance(priority, str):
            priority_float = priority_mapping.get(priority.lower(), 0.5)
        else:
            priority_float = float(priority)

        # 从description生成name（取前30个字符）
        name = description.replace("User Command: ", "")[:30]

        # 调用新的create_goal方法
        return self.create_goal(
            name=name,
            level=level,
            description=description,
            priority=priority_float,
            success_criteria=success_criteria
        )

    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """
        分解目标为子目标

        Args:
            goal_id: 要分解的目标ID

        Returns:
            子目标列表
        """
        if goal_id not in self.goals:
            return []

        parent_goal = self.goals[goal_id]

        # 如果已经分解过，返回现有子目标
        if parent_goal.children:
            return [self.goals[cid] for cid in parent_goal.children if cid in self.goals]

        # 根据目标层级生成子目标
        subgoals = self._generate_subgoals(parent_goal)

        # 添加子目标
        for subgoal in subgoals:
            self.goals[subgoal.id] = subgoal
            self.hierarchy[subgoal.level].append(subgoal.id)
            parent_goal.children.append(subgoal.id)

        print(f"  [GoalManager] Decomposed {parent_goal.name} into {len(subgoals)} subgoals")

        return subgoals

    def _generate_subgoals(self, parent_goal: Goal) -> List[Goal]:
        """生成子目标"""
        subgoals = []

        # 根据父目标层级决定子目标层级
        level_mapping = {
            GoalLevel.LIFETIME: GoalLevel.LONG_TERM,
            GoalLevel.LONG_TERM: GoalLevel.MEDIUM_TERM,
            GoalLevel.MEDIUM_TERM: GoalLevel.SHORT_TERM,
            GoalLevel.SHORT_TERM: GoalLevel.IMMEDIATE,
            GoalLevel.IMMEDIATE: None  # 即时目标不再分解
        }

        child_level = level_mapping.get(parent_goal.level)
        if not child_level:
            return []

        # 简化的子目标生成模板
        templates = {
            GoalLevel.LONG_TERM: [
                ("建立知识库", "收集和组织领域知识"),
                ("提升能力", "增强核心技能和算法"),
                ("优化架构", "改进系统设计和实现")
            ],
            GoalLevel.MEDIUM_TERM: [
                ("实施模块", "开发具体功能模块"),
                ("测试验证", "验证系统功能和性能"),
                ("文档记录", "编写技术文档")
            ],
            GoalLevel.SHORT_TERM: [
                ("任务规划", "制定具体执行计划"),
                ("资源准备", "准备所需资源"),
                ("进度跟踪", "监控任务进度")
            ],
            GoalLevel.IMMEDIATE: [
                ("执行动作", "执行当前操作"),
                ("状态检查", "检查当前状态"),
                ("结果记录", "记录执行结果")
            ]
        }

        template = templates.get(child_level, [])
        for i, (name, desc) in enumerate(template):
            subgoal = Goal(
                id=f"goal_{uuid.uuid4().hex[:8]}",
                name=f"{parent_goal.name}-{name}",
                level=child_level,
                status=GoalStatus.PENDING,
                description=f"{desc} (for {parent_goal.name})",
                priority=parent_goal.priority * 0.9,  # 子目标优先级略低
                created_at=time.time(),
                updated_at=time.time(),
                parent_id=parent_goal.id
            )
            subgoals.append(subgoal)

        return subgoals

    def activate_goal(self, goal_id: str) -> bool:
        """
        激活目标

        Args:
            goal_id: 目标ID

        Returns:
            是否成功激活
        """
        if goal_id not in self.goals:
            return False

        goal = self.goals[goal_id]

        # 检查是否超过最大活动目标数
        active_count = sum(1 for g in self.goals.values() if g.status == GoalStatus.ACTIVE)
        if active_count >= self.max_active_goals:
            print(f"  [GoalManager] Warning: Max active goals ({self.max_active_goals}) reached")
            return False

        # 激活目标
        goal.status = GoalStatus.ACTIVE
        goal.updated_at = time.time()

        print(f"  [GoalManager] Activated goal: {goal.name}")

        return True

    def complete_goal(self, goal_id: str, success: bool = True) -> bool:
        """
        完成目标

        Args:
            goal_id: 目标ID
            success: 是否成功

        Returns:
            是否成功标记完成
        """
        if goal_id not in self.goals:
            return False

        goal = self.goals[goal_id]

        if success:
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            self.stats['total_goals_completed'] += 1
            print(f"  [GoalManager] Completed goal: {goal.name}")
        else:
            goal.status = GoalStatus.FAILED
            print(f"  [GoalManager] Failed goal: {goal.name}")

        goal.updated_at = time.time()

        # 更新父目标进度
        self._update_parent_progress(goal_id)

        return True

    def _update_parent_progress(self, goal_id: str):
        """更新父目标进度"""
        goal = self.goals.get(goal_id)
        if not goal or not goal.parent_id:
            return

        parent = self.goals.get(goal.parent_id)
        if not parent:
            return

        # 计算所有子目标的平均进度
        sibling_progress = []
        for child_id in parent.children:
            if child_id in self.goals:
                child = self.goals[child_id]
                if child.status == GoalStatus.COMPLETED:
                    sibling_progress.append(1.0)
                elif child.status == GoalStatus.FAILED:
                    sibling_progress.append(0.0)
                else:
                    sibling_progress.append(child.progress)

        if sibling_progress:
            parent.progress = np.mean(sibling_progress)
            parent.updated_at = time.time()

        # 递归更新更高层
        self._update_parent_progress(parent.id)

    def detect_conflicts(self) -> List[GoalConflict]:
        """
        检测目标冲突

        Returns:
            检测到的冲突列表
        """
        conflicts = []
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]

        # 检查每对活动目标
        for i, goal1 in enumerate(active_goals):
            for goal2 in active_goals[i+1:]:
                conflict = self._check_pairwise_conflict(goal1, goal2)
                if conflict:
                    conflicts.append(conflict)

        self.conflicts.extend(conflicts)
        self.stats['total_conflicts_detected'] += len(conflicts)

        if conflicts:
            print(f"  [GoalManager] Detected {len(conflicts)} conflicts")

        return conflicts

    def _check_pairwise_conflict(self, goal1: Goal, goal2: Goal) -> Optional[GoalConflict]:
        """检查两个目标之间的冲突"""
        # 资源冲突（高优先级导致）
        if goal1.priority > 0.8 and goal2.priority > 0.8:
            # 两个高优先级目标可能冲突
            if goal1.level == goal2.level and goal1.level != GoalLevel.IMMEDIATE:
                return GoalConflict(
                    conflict_id=f"conf_{uuid.uuid4().hex[:8]}",
                    goal1_id=goal1.id,
                    goal2_id=goal2.id,
                    conflict_type="resource",
                    severity=0.7,
                    description=f"High-priority goals at same level: {goal1.name} vs {goal2.name}",
                    detected_at=time.time()
                )

        # 时间冲突（检查截止时间）
        if goal1.deadline and goal2.deadline:
            time_diff = abs(goal1.deadline - goal2.deadline)
            if time_diff < 3600:  # 1小时内
                return GoalConflict(
                    conflict_id=f"conf_{uuid.uuid4().hex[:8]}",
                    goal1_id=goal1.id,
                    goal2_id=goal2.id,
                    conflict_type="temporal",
                    severity=0.5,
                    description=f"Competing deadlines: {goal1.name} vs {goal2.name}",
                    detected_at=time.time()
                )

        return None

    def generate_autonomous_goal(self, context: Dict[str, Any]) -> Optional[Goal]:
        """
        自主生成新目标

        Args:
            context: 当前上下文

        Returns:
            生成的新目标
        """
        self.stats['total_goals_generated'] += 1

        # 分析上下文，生成合适的目标
        entropy = context.get('entropy', 0.5)
        curiosity = context.get('curiosity', 0.5)
        performance = context.get('performance', 0.5)

        # 高熵 -> 探索目标
        if entropy > 0.7:
            goal = self.create_goal(
                name="reduce_uncertainty",
                level=GoalLevel.SHORT_TERM,
                description="降低系统不确定性，提升预测准确性",
                priority=entropy,
                success_criteria="entropy < 0.5"
            )
            return goal

        # 低性能 -> 改进目标
        if performance < 0.5:
            goal = self.create_goal(
                name="improve_performance",
                level=GoalLevel.MEDIUM_TERM,
                description="提升系统性能指标",
                priority=1.0 - performance,
                success_criteria="performance > 0.7"
            )
            return goal

        # 高好奇心 -> 探索目标
        if curiosity > 0.8:
            goal = self.create_goal(
                name="explore_new_knowledge",
                level=GoalLevel.LONG_TERM,
                description="探索新领域知识",
                priority=curiosity * 0.7,
                success_criteria="discover >= 5 new insights"
            )
            return goal

        return None

    def get_active_goals(self, level: Optional[GoalLevel] = None) -> List[Goal]:
        """
        获取活动目标

        Args:
            level: 目标层级（None表示所有层级）

        Returns:
            活动目标列表
        """
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]

        if level:
            active_goals = [g for g in active_goals if g.level == level]

        # 按优先级排序
        active_goals.sort(key=lambda g: g.priority, reverse=True)

        return active_goals

    def get_goal_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取目标树

        Args:
            root_id: 根目标ID（None表示所有目标）

        Returns:
            目标树结构
        """
        if root_id:
            return self._build_subtree(root_id)

        # 找到所有根目标（没有父目标的目标）
        root_goals = [g for g in self.goals.values() if g.parent_id is None]

        return {
            root.id: self._build_subtree(root.id)
            for root in root_goals
        }

    def _build_subtree(self, goal_id: str) -> Dict[str, Any]:
        """构建子树"""
        if goal_id not in self.goals:
            return {}

        goal = self.goals[goal_id]

        subtree = {
            'id': goal.id,
            'name': goal.name,
            'level': goal.level.value,
            'status': goal.status.value,
            'priority': goal.priority,
            'progress': goal.progress,
            'children': {}
        }

        for child_id in goal.children:
            subtree['children'][child_id] = self._build_subtree(child_id)

        return subtree

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        level_counts = {}
        status_counts = {}

        for goal in self.goals.values():
            level_counts[goal.level.value] = level_counts.get(goal.level.value, 0) + 1
            status_counts[goal.status.value] = status_counts.get(goal.status.value, 0) + 1

        active_goals = self.get_active_goals()

        return {
            'total_goals': len(self.goals),
            'by_level': level_counts,
            'by_status': status_counts,
            'active_goals': len(active_goals),
            'avg_priority': np.mean([g.priority for g in active_goals]) if active_goals else 0,
            'active_conflicts': len([c for c in self.conflicts if c.detected_at > time.time() - 3600]),
            'stats': self.stats
        }


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("层级目标管理器测试")
    print("=" * 60)

    # 创建目标管理器
    manager = HierarchicalGoalManager(max_active_goals=10)

    # 测试1: 创建目标
    print("\n[测试1] 创建目标层级")
    print("-" * 60)

    lifetime_goal = manager.create_goal(
        name="become_intelligent",
        level=GoalLevel.LIFETIME,
        description="实现通用人工智能",
        priority=1.0
    )

    # 测试2: 目标分解
    print("\n[测试2] 目标分解")
    print("-" * 60)

    subgoals = manager.decompose_goal(lifetime_goal.id)
    print(f"  分解为 {len(subgoals)} 个长期目标")

    # 继续分解
    for subgoal in subgoals:
        if subgoal.level == GoalLevel.LONG_TERM:
            manager.decompose_goal(subgoal.id)

    # 测试3: 激活目标
    print("\n[测试3] 激活目标")
    print("-" * 60)

    manager.activate_goal(lifetime_goal.id)
    active = manager.get_active_goals()
    for goal in active[:3]:
        print(f"  {goal}")

    # 测试4: 冲突检测
    print("\n[测试4] 冲突检测")
    print("-" * 60)

    manager.create_goal(
        name="competing_goal_1",
        level=GoalLevel.SHORT_TERM,
        description="竞争目标1",
        priority=0.9
    )

    manager.create_goal(
        name="competing_goal_2",
        level=GoalLevel.SHORT_TERM,
        description="竞争目标2",
        priority=0.9
    )

    manager.activate_goal("competing_goal_1")
    manager.activate_goal("competing_goal_2")

    conflicts = manager.detect_conflicts()
    for conflict in conflicts:
        print(f"  {conflict.conflict_type}: {conflict.description}")

    # 测试5: 自主目标生成
    print("\n[测试5] 自主目标生成")
    print("-" * 60)

    context = {
        'entropy': 0.8,
        'curiosity': 0.6,
        'performance': 0.4
    }

    new_goal = manager.generate_autonomous_goal(context)
    if new_goal:
        print(f"  自主生成: {new_goal}")

    # 测试6: 摘要
    print("\n[测试6] 状态摘要")
    print("-" * 60)

    summary = manager.get_summary()
    for key, value in summary.items():
        if key != 'stats':
            print(f"  {key}: {value}")
        else:
            print(f"  stats:")
            for k, v in value.items():
                print(f"    {k}: {v}")

    # 测试7: 目标树
    print("\n[测试7] 目标树结构")
    print("-" * 60)

    tree = manager.get_goal_tree()
    import json
    print(json.dumps(tree, indent=2, ensure_ascii=False)[:500])
