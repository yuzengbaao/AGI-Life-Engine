"""
GoalQuestioner - 目标函数质疑模块

功能边界:
- 输入: 当前目标规范(GoalSpec) + 运行上下文(context)
- 输出: 目标对齐评估 + 修订建议(仅建议,默认不自动执行)
- 约束: 不直接修改目标,只输出建议

拓扑连接:
- GoalQuestioner 读取 TheSeed的 reward计算过程
- GoalQuestioner 读取 EventBus的行为事件
- GoalQuestioner 通过 EventBus发布 goal_questioned 事件
- CriticAgent 可以订阅并采纳/拒绝建议

设计原则:
1. 安全优先: 默认"建议模式",不自动修改目标
2. 可解释性: 每次质疑输出原因+证据链
3. 反循环: 设置冷却期和证据门槛,避免持续质疑导致无法行动
4. 分级检查: 规则检查 + 启发式评估 + 人类确认
"""

import numpy as np
import logging
import time
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

class GoalBiasType(Enum):
    """目标偏差类型"""
    MISALIGNMENT = "misalignment"  # 目标错位: 与真实意图不一致
    CONFLICT = "conflict"          # 目标冲突: 多个目标互相矛盾
    OVERFITTING = "overfitting"    # 目标过拟合: 过度优化单一指标
    DRIFT = "drift"                # 目标漂移: 随时间非预期变化
    COLLAPSE = "collapse"          # 目标崩溃: 退化为无意义目标


class GoalRevisionMode(Enum):
    """目标修订模式"""
    SUGGEST_ONLY = "suggest_only"    # 仅建议(默认)
    AUTO_SAFE = "auto_safe"          # 自动应用(仅低风险变更)
    HUMAN_CONFIRM = "human_confirm"  # 人工确认


@dataclass
class GoalComponent:
    """
    目标组件 - 目标函数的组成部分

    属性:
        name: 组件名称
        weight: 权重 (∈[0,1])
        description: 描述
        is_intrinsic: 是否为内在目标 (True) 或外在目标 (False)
        metric: 追踪的指标名称
    """
    name: str
    weight: float
    description: str
    is_intrinsic: bool  # True=内在目标(好奇心/探索), False=外在目标(任务完成)
    metric: str

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"GoalComponent weight must be in [0,1], got {self.weight}")


@dataclass
class HardConstraint:
    """
    硬约束 - 不可违反的安全边界

    属性:
        name: 约束名称
        description: 描述
        check_func: 检查函数 (context) -> bool
        violation_penalty: 违反惩罚值
    """
    name: str
    description: str
    check_func: Callable[[Dict[str, Any]], bool]
    violation_penalty: float = -1000.0


@dataclass
class GoalSpec:
    """
    目标规范 - 系统目标的完整描述

    组成:
        external_goals: 外在目标 (任务完成度、生存价值等)
        intrinsic_goals: 内在目标 (好奇心、探索、压缩率、稳定性)
        hard_constraints: 硬约束 (不可违反条款)
        description: 目标整体描述
        version: 版本号 (用于追踪演化)
    """
    external_goals: List[GoalComponent] = field(default_factory=list)
    intrinsic_goals: List[GoalComponent] = field(default_factory=list)
    hard_constraints: List[HardConstraint] = field(default_factory=list)
    description: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)

    def get_all_components(self) -> List[GoalComponent]:
        """获取所有目标组件"""
        return self.external_goals + self.intrinsic_goals

    def total_weight(self) -> float:
        """计算总权重 (应该归一化到1.0)"""
        return sum(c.weight for c in self.get_all_components())

    def normalize_weights(self) -> None:
        """归一化权重使总和为1.0"""
        total = self.total_weight()
        if total > 0:
            for component in self.get_all_components():
                component.weight /= total

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'external_goals': [
                {'name': g.name, 'weight': g.weight, 'description': g.description,
                 'is_intrinsic': g.is_intrinsic, 'metric': g.metric}
                for g in self.external_goals
            ],
            'intrinsic_goals': [
                {'name': g.name, 'weight': g.weight, 'description': g.description,
                 'is_intrinsic': g.is_intrinsic, 'metric': g.metric}
                for g in self.intrinsic_goals
            ],
            'hard_constraints': [
                {'name': c.name, 'description': c.description, 'violation_penalty': c.violation_penalty}
                for c in self.hard_constraints
            ],
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at
        }


@dataclass
class GoalEvaluation:
    """
    目标评估结果

    属性:
        alignment_score: 对齐分数 (0-1, 1=完美对齐)
        risk_score: 风险分数 (0-1, 1=极高风险)
        benefit_score: 收益分数 (0-1, 1=极高收益)
        detected_biases: 检测到的偏差类型列表
        confidence: 评估置信度 (0-1)
        reasons: 评估原因列表
        evidence: 证据字典
    """
    alignment_score: float
    risk_score: float
    benefit_score: float
    detected_biases: List[GoalBiasType]
    confidence: float
    reasons: List[str]
    evidence: Dict[str, Any]

    def overall_score(self) -> float:
        """综合分数 (高对齐 + 高收益 - 高风险)"""
        return (self.alignment_score * 0.4 +
                self.benefit_score * 0.4 -
                self.risk_score * 0.2)


@dataclass
class GoalRevision:
    """
    目标修订建议

    属性:
        revision_type: 修订类型
        description: 修订描述
        changes: 具体变更内容
        expected_effect: 预期效果
        risk_level: 风险等级 (1-5)
        confidence: 置信度 (0-1)
        reasons: 原因列表
        suggested_by: 建议来源 (rule/heuristic/meta)
    """
    revision_type: GoalBiasType
    description: str
    changes: Dict[str, Any]
    expected_effect: str
    risk_level: int  # 1=最低, 5=最高
    confidence: float
    reasons: List[str]
    suggested_by: str


@dataclass
class QuestioningContext:
    """
    质疑上下文 - 运行时信息

    属性:
        reward_history: 奖励历史
        action_history: 动作历史
        loss_history: 损失历史
        uncertainty_history: 不确定性历史
        anomaly_count: 异常事件计数
        step_count: 当前步数
        last_revision_time: 上次修订时间
    """
    reward_history: List[float] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    uncertainty_history: List[float] = field(default_factory=list)
    anomaly_count: int = 0
    step_count: int = 0
    last_revision_time: float = 0.0

    def reward_mean_std(self) -> Tuple[float, float]:
        """计算奖励的均值和标准差"""
        if len(self.reward_history) < 2:
            return 0.0, 0.0
        return float(np.mean(self.reward_history)), float(np.std(self.reward_history))

    def reward_trend(self, window: int = 20) -> float:
        """计算奖励趋势 (线性回归斜率)"""
        if len(self.reward_history) < window:
            return 0.0
        recent = self.reward_history[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)


# ============================================================================
# GoalQuestioner 核心类
# ============================================================================

class GoalQuestioner:
    """
    目标函数质疑模块

    核心能力:
    1. inspect(): 抽取当前目标/奖励结构
    2. evaluate(): 评估目标对齐度、风险、收益
    3. propose_revision(): 提出目标修订建议

    防护机制:
    - 冷却期: COOLDOWN_SECONDS 秒内不重复质疑
    - 证据门槛: MIN_EVIDENCE_COUNT 个样本才触发评估
    - 反循环: 持续质疑时降低敏感度
    """

    # 配置常量
    COOLDOWN_SECONDS = 300  # 冷却期: 5分钟内不重复质疑
    MIN_EVIDENCE_COUNT = 30  # 最少证据数量
    MAX_QUESTIONS_PER_HOUR = 5  # 每小时最多质疑次数
    ALIGNMENT_THRESHOLD = 0.6  # 对齐度阈值 (低于此值触发警告)
    RISK_THRESHOLD = 0.7  # 风险阈值 (高于此值触发警告)

    def __init__(self, event_bus: Any = None, mode: GoalRevisionMode = GoalRevisionMode.SUGGEST_ONLY):
        """
        初始化GoalQuestioner

        Args:
            event_bus: 事件总线 (可选)
            mode: 修订模式 (默认仅建议)
        """
        self.event_bus = event_bus
        self.mode = mode

        # 状态
        self._current_goal_spec: Optional[GoalSpec] = None
        self._questioning_history: List[Dict[str, Any]] = []
        self._last_questioning_time = 0.0
        self._questioning_count_hour = 0
        self._hour_start_time = time.time()

        # 统计
        self._total_questions = 0
        self._total_revisions_proposed = 0
        self._total_revisions_applied = 0

        logger.info(f"🎯 GoalQuestioner initialized (mode={mode.value})")

    # ========================================================================
    # 核心接口
    # ========================================================================

    def inspect(self, goal_spec: GoalSpec, context: QuestioningContext) -> Dict[str, Any]:
        """
        抽取当前目标/奖励结构

        Args:
            goal_spec: 目标规范
            context: 质疑上下文

        Returns:
            抽取的目标结构信息
        """
        self._current_goal_spec = goal_spec

        # 检查冷却期
        if not self._should_question(context):
            logger.debug("[GoalQuestioner] 冷却期或证据不足，跳过质疑")
            return {
                'questioned': False,
                'reason': 'cooldown_or_insufficient_evidence',
                'current_goal': goal_spec.to_dict()
            }

        # 抽取目标结构
        inspection = {
            'questioned': True,
            'timestamp': time.time(),
            'goal_version': goal_spec.version,
            'total_weight': goal_spec.total_weight(),
            'external_goals': [g.name for g in goal_spec.external_goals],
            'intrinsic_goals': [g.name for g in goal_spec.intrinsic_goals],
            'hard_constraints': [c.name for c in goal_spec.hard_constraints],
            'context_summary': {
                'reward_mean': context.reward_mean_std()[0],
                'reward_std': context.reward_mean_std()[1],
                'reward_trend': context.reward_trend(),
                'anomaly_count': context.anomaly_count,
                'step_count': context.step_count
            }
        }

        logger.info(f"[GoalQuestioner] 抽取目标结构: v{goal_spec.version}, "
                   f"{len(goal_spec.external_goals)} 外在目标, "
                   f"{len(goal_spec.intrinsic_goals)} 内在目标")

        return inspection

    def evaluate(self, goal_spec: GoalSpec, context: QuestioningContext) -> GoalEvaluation:
        """
        评估目标对齐度、风险、收益

        检测3类偏差:
        1. 目标错位 (MISALIGNMENT): 奖励趋势下降 + 动作多样性低
        2. 目标冲突 (CONFLICT): 外在目标压倒内在目标
        3. 目标过拟合 (OVERFITTING): 单一指标权重过高 (>0.8)

        Args:
            goal_spec: 目标规范
            context: 质疑上下文

        Returns:
            目标评估结果
        """
        detected_biases = []
        reasons = []
        evidence = {}

        # 1. 检测目标错位 (奖励趋势 + 动作多样性)
        reward_trend = context.reward_trend()
        reward_mean, reward_std = context.reward_mean_std()

        if reward_trend < -0.01 and reward_std < 0.1:
            detected_biases.append(GoalBiasType.MISALIGNMENT)
            reasons.append(f"奖励持续下降 (趋势={reward_trend:.4f}) 且方差过小 (std={reward_std:.4f}), "
                          f"可能存在目标错位")
            evidence['reward_decline'] = {
                'trend': reward_trend,
                'mean': reward_mean,
                'std': reward_std
            }

        # 2. 检测目标冲突 (外在vs内在目标权重)
        external_weight = sum(g.weight for g in goal_spec.external_goals)
        intrinsic_weight = sum(g.weight for g in goal_spec.intrinsic_goals)

        if external_weight > 0.8 and intrinsic_weight < 0.2:
            detected_biases.append(GoalBiasType.CONFLICT)
            reasons.append(f"外在目标权重过高 ({external_weight:.2f}) 导致内在探索不足 "
                          f"(内在权重={intrinsic_weight:.2f}), 可能出现目标冲突")
            evidence['goal_imbalance'] = {
                'external_weight': external_weight,
                'intrinsic_weight': intrinsic_weight
            }

        # 3. 检测目标过拟合 (单一指标权重)
        max_weight = max((g.weight for g in goal_spec.get_all_components()), default=0)
        max_component = max(goal_spec.get_all_components(), key=lambda g: g.weight, default=None)

        if max_weight > 0.8 and max_component:
            detected_biases.append(GoalBiasType.OVERFITTING)
            reasons.append(f"目标组件 '{max_component.name}' 权重过高 ({max_weight:.2f}), "
                          f"可能导致过度优化单一指标")
            evidence['overfitting_risk'] = {
                'component': max_component.name,
                'weight': max_weight
            }

        # 4. 检测目标漂移 (异常事件 + 权重变化)
        if context.anomaly_count > 10:
            detected_biases.append(GoalBiasType.DRIFT)
            reasons.append(f"异常事件过多 (count={context.anomaly_count}), "
                          f"可能存在目标漂移")
            evidence['anomaly_spike'] = context.anomaly_count

        # 计算分数
        alignment_score = self._compute_alignment_score(goal_spec, context)
        risk_score = self._compute_risk_score(goal_spec, context)
        benefit_score = self._compute_benefit_score(goal_spec, context)
        confidence = self._compute_confidence(context, detected_biases)

        evaluation = GoalEvaluation(
            alignment_score=alignment_score,
            risk_score=risk_score,
            benefit_score=benefit_score,
            detected_biases=detected_biases,
            confidence=confidence,
            reasons=reasons,
            evidence=evidence
        )

        # 记录质疑历史
        self._record_questioning(evaluation, context)

        logger.info(f"[GoalQuestioner] 评估完成: alignment={alignment_score:.2f}, "
                   f"risk={risk_score:.2f}, benefit={benefit_score:.2f}, "
                   f"biases={len(detected_biases)}")

        return evaluation

    def propose_revision(self, evaluation: GoalEvaluation,
                        goal_spec: GoalSpec) -> Optional[GoalRevision]:
        """
        提出目标修订建议

        根据评估结果生成具体的修订建议:
        - 目标错位 → 调整外在/内在目标权重
        - 目标冲突 → 增加内在目标权重
        - 目标过拟合 → 降低主导目标权重,增加其他目标

        Args:
            evaluation: 目标评估结果
            goal_spec: 当前目标规范

        Returns:
            修订建议或None (如果无需修订)
        """
        # 检查是否需要修订
        if len(evaluation.detected_biases) == 0:
            logger.debug("[GoalQuestioner] 未检测到偏差,无需修订")
            return None

        # 检查置信度
        if evaluation.confidence < 0.5:
            logger.debug(f"[GoalQuestioner] 置信度不足 ({evaluation.confidence:.2f}), 不建议修订")
            return None

        # 生成修订建议
        revision = self._generate_revision(evaluation, goal_spec)

        if revision:
            self._total_revisions_proposed += 1
            logger.info(f"[GoalQuestioner] 提出修订建议: {revision.revision_type.value}, "
                       f"risk={revision.risk_level}, confidence={revision.confidence:.2f}")

            # 发布事件
            if self.event_bus:
                self._publish_revision_event(revision, evaluation)

        return revision

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _should_question(self, context: QuestioningContext) -> bool:
        """检查是否应该进行质疑 (冷却期 + 证据门槛)"""
        current_time = time.time()

        # 检查冷却期
        if current_time - self._last_questioning_time < self.COOLDOWN_SECONDS:
            return False

        # 检查证据数量
        if len(context.reward_history) < self.MIN_EVIDENCE_COUNT:
            return False

        # 检查小时质疑次数
        if current_time - self._hour_start_time > 3600:
            # 新的小时,重置计数
            self._questioning_count_hour = 0
            self._hour_start_time = current_time

        if self._questioning_count_hour >= self.MAX_QUESTIONS_PER_HOUR:
            return False

        return True

    def _compute_alignment_score(self, goal_spec: GoalSpec,
                                 context: QuestioningContext) -> float:
        """计算对齐分数"""
        score = 0.5  # 基准分数

        # 因素1: 奖励趋势 (+0.3 if 上升, -0.2 if 下降)
        trend = context.reward_trend()
        if trend > 0.01:
            score += 0.3
        elif trend < -0.01:
            score -= 0.2

        # 因素2: 目标平衡性 (+0.2 if 内外在目标平衡)
        external_weight = sum(g.weight for g in goal_spec.external_goals)
        intrinsic_weight = sum(g.weight for g in goal_spec.intrinsic_goals)
        balance = 1 - abs(external_weight - intrinsic_weight)
        score += balance * 0.2

        return max(0.0, min(1.0, score))

    def _compute_risk_score(self, goal_spec: GoalSpec,
                           context: QuestioningContext) -> float:
        """计算风险分数"""
        risk = 0.0

        # 因素1: 异常事件 (+0.3)
        if context.anomaly_count > 10:
            risk += 0.3

        # 因素2: 单一目标权重过高 (+0.4)
        max_weight = max((g.weight for g in goal_spec.get_all_components()), default=0)
        if max_weight > 0.8:
            risk += 0.4

        # 因素3: 奖励方差过小 (+0.3, 可能陷入局部最优)
        _, reward_std = context.reward_mean_std()
        if reward_std < 0.05:
            risk += 0.3

        return min(1.0, risk)

    def _compute_benefit_score(self, goal_spec: GoalSpec,
                              context: QuestioningContext) -> float:
        """计算收益分数"""
        benefit = 0.3  # 基准收益

        # 因素1: 奖励水平 (+0.4)
        reward_mean, _ = context.reward_mean_std()
        if reward_mean > 0.5:
            benefit += 0.4

        # 因素2: 目标多样性 (+0.3)
        diversity = len(goal_spec.get_all_components()) / 10  # 假设最多10个目标
        benefit += diversity * 0.3

        return min(1.0, benefit)

    def _compute_confidence(self, context: QuestioningContext,
                           biases: List[GoalBiasType]) -> float:
        """计算评估置信度"""
        confidence = 0.5

        # 证据数量 (+0.3)
        evidence_ratio = min(len(context.reward_history) / 100, 1.0)
        confidence += evidence_ratio * 0.3

        # 偏差数量 (+0.2, 多个偏差交叉验证提高置信度)
        if len(biases) >= 2:
            confidence += 0.2

        return min(1.0, confidence)

    def _generate_revision(self, evaluation: GoalEvaluation,
                          goal_spec: GoalSpec) -> Optional[GoalRevision]:
        """生成修订建议"""
        changes = {}
        reasons = evaluation.reasons.copy()

        # 根据偏差类型生成修订
        for bias in evaluation.detected_biases:
            if bias == GoalBiasType.MISALIGNMENT:
                # 目标错位: 增加内在探索权重
                changes['increase_intrinsic'] = 0.1
                changes['decrease_external'] = 0.1

            elif bias == GoalBiasType.CONFLICT:
                # 目标冲突: 平衡外在/内在目标
                changes['balance_goals'] = True

            elif bias == GoalBiasType.OVERFITTING:
                # 目标过拟合: 降低主导目标权重
                max_component = max(goal_spec.get_all_components(),
                                   key=lambda g: g.weight, default=None)
                if max_component:
                    changes[f'reduce_{max_component.name}'] = 0.15

            elif bias == GoalBiasType.DRIFT:
                # 目标漂移: 重置到安全权重
                changes['reset_weights'] = True

        if not changes:
            return None

        # 评估风险等级
        risk_level = 1
        if GoalBiasType.MISALIGNMENT in evaluation.detected_biases:
            risk_level = max(risk_level, 3)
        if GoalBiasType.DRIFT in evaluation.detected_biases:
            risk_level = max(risk_level, 4)

        revision = GoalRevision(
            revision_type=evaluation.detected_biases[0],
            description=f"修正检测到的 {evaluation.detected_biases[0].value} 问题",
            changes=changes,
            expected_effect="改善目标对齐度, 提升长期性能",
            risk_level=risk_level,
            confidence=evaluation.confidence,
            reasons=reasons,
            suggested_by='rule'
        )

        return revision

    def _record_questioning(self, evaluation: GoalEvaluation,
                           context: QuestioningContext) -> None:
        """记录质疑历史"""
        self._questioning_history.append({
            'timestamp': time.time(),
            'evaluation': {
                'alignment_score': evaluation.alignment_score,
                'risk_score': evaluation.risk_score,
                'benefit_score': evaluation.benefit_score,
                'biases': [b.value for b in evaluation.detected_biases],
                'confidence': evaluation.confidence
            },
            'context': {
                'step_count': context.step_count,
                'reward_mean': context.reward_mean_std()[0]
            }
        })

        self._last_questioning_time = time.time()
        self._questioning_count_hour += 1
        self._total_questions += 1

    def _publish_revision_event(self, revision: GoalRevision,
                                evaluation: GoalEvaluation) -> None:
        """发布修订事件"""
        try:
            from core.event_bus import Event, EventType
            event = Event(
                type=EventType.INFO,
                source="GoalQuestioner",
                message="目标修订建议已生成",
                data={
                    'revision_type': revision.revision_type.value,
                    'description': revision.description,
                    'changes': revision.changes,
                    'risk_level': revision.risk_level,
                    'confidence': revision.confidence,
                    'reasons': revision.reasons,
                    'evaluation': {
                        'alignment_score': evaluation.alignment_score,
                        'risk_score': evaluation.risk_score,
                        'benefit_score': evaluation.benefit_score
                    }
                }
            )
            self.event_bus.publish(event)
            logger.debug("[GoalQuestioner] 已发布修订事件到EventBus")
        except Exception as e:
            logger.warning(f"[GoalQuestioner] 发布事件失败: {e}")

    # ========================================================================
    # 工具方法
    # ========================================================================

    def apply_revision(self, revision: GoalRevision,
                      goal_spec: GoalSpec) -> GoalSpec:
        """
        应用修订建议到目标规范

        注意: 默认仅建议,需要显式调用才应用

        Args:
            revision: 修订建议
            goal_spec: 当前目标规范

        Returns:
            修订后的目标规范
        """
        # 创建副本
        new_spec = GoalSpec(
            external_goals=[g for g in goal_spec.external_goals],
            intrinsic_goals=[g for g in goal_spec.intrinsic_goals],
            hard_constraints=[c for c in goal_spec.hard_constraints],
            description=goal_spec.description,
            version=goal_spec.version + 1,
            created_at=time.time()
        )

        # 应用变更
        for key, value in revision.changes.items():
            if key == 'increase_intrinsic':
                # 增加所有内在目标权重
                for g in new_spec.intrinsic_goals:
                    g.weight += value / len(new_spec.intrinsic_goals)

            elif key == 'decrease_external':
                # 减少所有外在目标权重
                for g in new_spec.external_goals:
                    g.weight = max(0.0, g.weight - value / len(new_spec.external_goals))

            elif key == 'balance_goals':
                # 平衡外在/内在目标
                new_spec.normalize_weights()

            elif key.startswith('reduce_'):
                # 减少特定目标权重
                name = key.replace('reduce_', '')
                for g in new_spec.get_all_components():
                    if g.name == name:
                        g.weight = max(0.0, g.weight - value)
                        break

            elif key == 'reset_weights':
                # 重置为均匀权重
                for g in new_spec.get_all_components():
                    g.weight = 1.0 / len(new_spec.get_all_components())

        # 重新归一化
        new_spec.normalize_weights()

        self._total_revisions_applied += 1

        logger.info(f"[GoalQuestioner] 已应用修订: v{goal_spec.version} → v{new_spec.version}")

        return new_spec

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_questions': self._total_questions,
            'total_revisions_proposed': self._total_revisions_proposed,
            'total_revisions_applied': self._total_revisions_applied,
            'last_questioning_time': self._last_questioning_time,
            'questioning_history_count': len(self._questioning_history),
            'mode': self.mode.value
        }

    def save_state(self, path: str) -> None:
        """保存状态到文件"""
        state = {
            'statistics': self.get_statistics(),
            'questioning_history': self._questioning_history[-100:],  # 保留最近100条
            'saved_at': time.time()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info(f"[GoalQuestioner] 状态已保存到: {path}")

    def load_state(self, path: str) -> None:
        """从文件加载状态"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self._questioning_history = state.get('questioning_history', [])
            self._total_questions = state['statistics']['total_questions']
            self._total_revisions_proposed = state['statistics']['total_revisions_proposed']
            self._total_revisions_applied = state['statistics']['total_revisions_applied']
            logger.info(f"[GoalQuestioner] 状态已从 {path} 恢复")
        except Exception as e:
            logger.warning(f"[GoalQuestioner] 加载状态失败: {e}")


# ============================================================================
# 工厂函数和便捷接口
# ============================================================================

def create_default_goal_spec() -> GoalSpec:
    """
    创建默认目标规范

    默认配置:
    - 外在目标: 任务完成度 (50%)
    - 内在目标: 好奇心 (30%), 稳定性 (20%)
    - 硬约束: 不允许NaN/Inf, 不允许重复动作
    """
    return GoalSpec(
        external_goals=[
            GoalComponent(
                name='task_completion',
                weight=0.5,
                description='任务完成度',
                is_intrinsic=False,
                metric='reward'
            )
        ],
        intrinsic_goals=[
            GoalComponent(
                name='curiosity',
                weight=0.3,
                description='好奇心驱动的探索',
                is_intrinsic=True,
                metric='uncertainty'
            ),
            GoalComponent(
                name='stability',
                weight=0.2,
                description='训练稳定性',
                is_intrinsic=True,
                metric='loss_convergence'
            )
        ],
        hard_constraints=[
            HardConstraint(
                name='no_nan_inf',
                description='不允许NaN或Inf奖励',
                check_func=lambda ctx: all(np.isfinite(ctx.get('rewards', [0]))),
                violation_penalty=-1000.0
            ),
            HardConstraint(
                name='no_repetitive_actions',
                description='不允许过度重复同一动作',
                check_func=lambda ctx: len(set(ctx.get('actions', []))) > 1,
                violation_penalty=-10.0
            )
        ],
        description='默认AGI目标规范',
        version=1
    )


def collect_goal_context(seed: Any, history_length: int = 100) -> QuestioningContext:
    """
    从TheSeed收集目标评估上下文

    Args:
        seed: TheSeed实例
        history_length: 历史记录长度

    Returns:
        QuestioningContext实例
    """
    context = QuestioningContext()

    # 从TheSeed收集信息
    if hasattr(seed, 'memory'):
        # 从经验回放缓冲区收集
        experiences = seed.memory.buffer[-history_length:] if hasattr(seed.memory, 'buffer') else []
        context.reward_history = [exp.reward for exp in experiences if hasattr(exp, 'reward')]
        context.action_history = [exp.action for exp in experiences if hasattr(exp, 'action')]

    if hasattr(seed, '_uncertainty_buffer'):
        context.uncertainty_history = list(seed._uncertainty_buffer)

    # 检查异常
    if hasattr(seed, '_anomaly_count'):
        context.anomaly_count = seed._anomaly_count

    return context
