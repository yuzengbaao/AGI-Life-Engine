#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递归自指优化引擎（Recursive Self-Reference Optimization）
========================================================

功能：实现元认知能力，关于思考的思考
基于：Meta-Cognition Theory + Self-Model + Recursive Self-Improvement

核心能力：
1. 元认知监控（关于思考的思考）
2. 自我建模（构建自我模型）
3. 自我评估（评估自身性能）
4. 递归改进（自我修正和优化）

版本: 1.0.0
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class MetaCognitiveState(Enum):
    """元认知状态"""
    MONITORING = "monitoring"        # 监控自身思考过程
    EVALUATING = "evaluating"        # 评估自身性能
    REFLECTING = "reflecting"        # 反思和总结
    IMPROVING = "improving"          # 改进和优化
    DEBUGGING = "debugging"          # 调试自身错误


@dataclass
class SelfModel:
    """自我模型"""
    model_id: str
    capabilities: Dict[str, float]  # 能力评估
    limitations: List[str]           # 已知限制
    performance_history: List[float] # 性能历史
    learning_style: str              # 学习风格
    personality_traits: Dict[str, float]  # 人格特质
    last_updated: float

    def __repr__(self):
        return f"SelfModel(capabilities={len(self.capabilities)}, limitations={len(self.limitations)})"


@dataclass
class ThoughtProcess:
    """思考过程记录"""
    process_id: str
    timestamp: float
    input_stimulus: str
    reasoning_steps: List[str]
    conclusion: str
    confidence: float
    meta_commentary: str  # 元认知评论


@dataclass
class SelfReflection:
    """自我反思"""
    reflection_id: str
    timestamp: float
    topic: str
    observations: List[str]
    insights: List[str]
    action_items: List[str]
    confidence_delta: float


class RecursiveSelfReferenceEngine:
    """
    递归自指优化引擎

    核心功能：
    1. 监控自身思考过程
    2. 构建和维护自我模型
    3. 评估自身性能
    4. 生成自我反思和改进计划
    """

    def __init__(self, max_recursion_depth: int = 3):
        """
        初始化递归自指引擎

        Args:
            max_recursion_depth: 最大递归深度（防止无限递归）
        """
        self.max_recursion_depth = max_recursion_depth

        # 自我模型
        self.self_model = SelfModel(
            model_id="self_v1",
            capabilities={},
            limitations=[],
            performance_history=[],
            learning_style="active",
            personality_traits={},
            last_updated=time.time()
        )

        # 思考过程历史
        self.thought_processes: deque = deque(maxlen=1000)

        # 自我反思历史
        self.reflections: List[SelfReflection] = []

        # 元认知状态
        self.current_state = MetaCognitiveState.MONITORING

        # 递归计数器
        self.recursion_depth = 0

        # 统计信息
        self.stats = {
            'total_thoughts_monitored': 0,
            'total_reflections': 0,
            'total_self_evaluations': 0,
            'total_improvements_applied': 0,
            'avg_self_awareness': 0.0,
            'meta_cognitive_cycles': 0
        }

    def monitor_thought(self, input_stimulus: str, reasoning_steps: List[str],
                       conclusion: str, confidence: float) -> ThoughtProcess:
        """
        监控思考过程

        Args:
            input_stimulus: 输入刺激
            reasoning_steps: 推理步骤
            conclusion: 结论
            confidence: 置信度

        Returns:
            思考过程记录
        """
        self.stats['total_thoughts_monitored'] += 1

        # 元认知分析
        meta_commentary = self._generate_meta_commentary(
            input_stimulus, reasoning_steps, conclusion, confidence
        )

        # 创建思考过程记录
        thought = ThoughtProcess(
            process_id=f"thought_{int(time.time() * 1000)}",
            timestamp=time.time(),
            input_stimulus=input_stimulus,
            reasoning_steps=reasoning_steps,
            conclusion=conclusion,
            confidence=confidence,
            meta_commentary=meta_commentary
        )

        self.thought_processes.append(thought)

        # 更新自我模型
        self._update_self_model_from_thought(thought)

        return thought

    def _generate_meta_commentary(self, input: str, steps: List[str],
                                   conclusion: str, confidence: float) -> str:
        """生成元认知评论"""
        comments = []

        # 评估推理深度
        depth = len(steps)
        if depth < 3:
            comments.append("Shallow reasoning - consider deeper analysis")
        elif depth > 10:
            comments.append("Deep reasoning - good analytical depth")
        else:
            comments.append("Moderate reasoning depth")

        # 评估置信度
        if confidence < 0.5:
            comments.append("Low confidence - should seek more evidence")
        elif confidence > 0.9:
            comments.append("High confidence - verify to avoid overconfidence")

        # 评估结论质量
        if not conclusion or len(conclusion) < 10:
            comments.append("Vague conclusion - should be more specific")

        return "; ".join(comments)

    def _update_self_model_from_thought(self, thought: ThoughtProcess):
        """从思考过程更新自我模型"""
        # 更新能力评估
        reasoning_quality = len(thought.reasoning_steps) * 0.1
        self.self_model.capabilities['reasoning'] = (
            self.self_model.capabilities.get('reasoning', 0.5) * 0.9 + reasoning_quality * 0.1
        )

        # 更新性能历史
        self.self_model.performance_history.append(thought.confidence)
        if len(self.self_model.performance_history) > 100:
            self.self_model.performance_history.pop(0)

        # 更新时间戳
        self.self_model.last_updated = time.time()

    def evaluate_self(self) -> Dict[str, Any]:
        """
        自我评估

        Returns:
            自我评估结果
        """
        self.stats['total_self_evaluations'] += 1

        print(f"\n  [MetaCognition] Self-evaluation")

        # 1. 能力评估
        capability_score = self._evaluate_capabilities()

        # 2. 性能趋势
        performance_trend = self._analyze_performance_trend()

        # 3. 自我意识水平
        self_awareness = self._calculate_self_awareness()

        # 4. 识别限制
        limitations = self._identify_limitations()

        # 5. 更新统计
        self.stats['avg_self_awareness'] = (
            (self.stats['avg_self_awareness'] * (self.stats['total_self_evaluations'] - 1)
             + self_awareness) / self.stats['total_self_evaluations']
        )

        evaluation = {
            'capability_score': capability_score,
            'performance_trend': performance_trend,
            'self_awareness': self_awareness,
            'limitations': limitations,
            'timestamp': time.time()
        }

        print(f"    Capability score: {capability_score:.3f}")
        print(f"    Performance trend: {performance_trend}")
        print(f"    Self-awareness: {self_awareness:.3f}")
        print(f"    Limitations identified: {len(limitations)}")

        return evaluation

    def _evaluate_capabilities(self) -> float:
        """评估能力"""
        if not self.self_model.capabilities:
            return 0.5

        return sum(self.self_model.capabilities.values()) / len(self.self_model.capabilities)

    def _analyze_performance_trend(self) -> str:
        """分析性能趋势"""
        if len(self.self_model.performance_history) < 3:
            return "insufficient_data"

        recent = self.self_model.performance_history[-10:]
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_second > avg_first + 0.1:
            return "improving"
        elif avg_second < avg_first - 0.1:
            return "declining"
        else:
            return "stable"

    def _calculate_self_awareness(self) -> float:
        """计算自我意识水平"""
        # 基于多个维度
        thought_count = len(self.thought_processes)
        reflection_count = len(self.reflections)
        capability_count = len(self.self_model.capabilities)

        # 归一化
        awareness = (
            min(thought_count / 100, 1.0) * 0.3 +
            min(reflection_count / 10, 1.0) * 0.3 +
            min(capability_count / 10, 1.0) * 0.4
        )

        return awareness

    def _identify_limitations(self) -> List[str]:
        """识别自身限制"""
        limitations = []

        # 基于自我模型的限制
        if self.self_model.capabilities.get('creativity', 0.5) < 0.3:
            limitations.append("Limited creative capability")

        if self.self_model.capabilities.get('reasoning', 0.5) < 0.5:
            limitations.append("Reasoning depth could be improved")

        if self._calculate_self_awareness() < 0.5:
            limitations.append("Self-awareness needs development")

        # 基于历史的限制
        if len(self.reflections) < 5:
            limitations.append("Insufficient self-reflection practice")

        return limitations

    def reflect_on(self, topic: str, context: Dict[str, Any]) -> SelfReflection:
        """
        关于某个主题进行自我反思

        Args:
            topic: 反思主题
            context: 上下文信息

        Returns:
            自我反思结果
        """
        self.stats['total_reflections'] += 1

        print(f"\n  [MetaCognition] Reflecting on: {topic}")

        # 1. 收集相关思考过程
        relevant_thoughts = [
            t for t in self.thought_processes
            if topic.lower() in t.input_stimulus.lower() or topic.lower() in t.conclusion.lower()
        ]

        # 2. 生成观察
        observations = self._generate_observations(topic, relevant_thoughts, context)

        # 3. 生成洞察
        insights = self._generate_insights(topic, observations)

        # 4. 生成行动计划
        action_items = self._generate_action_items(insights)

        # 5. 计算置信度变化
        confidence_delta = sum(len(obs) for obs in observations) * 0.01

        reflection = SelfReflection(
            reflection_id=f"reflect_{int(time.time() * 1000)}",
            timestamp=time.time(),
            topic=topic,
            observations=observations,
            insights=insights,
            action_items=action_items,
            confidence_delta=confidence_delta
        )

        self.reflections.append(reflection)

        print(f"    Observations: {len(observations)}")
        print(f"    Insights: {len(insights)}")
        print(f"    Action items: {len(action_items)}")

        return reflection

    def _generate_observations(self, topic: str, thoughts: List[ThoughtProcess],
                              context: Dict) -> List[str]:
        """生成观察"""
        observations = []

        # 观察思考模式
        if thoughts:
            avg_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
            observations.append(f"Average confidence on this topic: {avg_confidence:.2f}")

            # 检查是否过度自信或不自信
            if avg_confidence < 0.4:
                observations.append("Tendency to be underconfident on this topic")
            elif avg_confidence > 0.8:
                observations.append("Risk of overconfidence on this topic")

        # 观察上下文因素
        if context.get('complexity') == 'high':
            observations.append("High complexity situations may require more deliberation")

        if context.get('time_pressure'):
            observations.append("Time pressure may affect reasoning quality")

        return observations

    def _generate_insights(self, topic: str, observations: List[str]) -> List[str]:
        """生成洞察"""
        insights = []

        # 洞察1: 模式识别
        if "overconfidence" in str(observations):
            insights.append("Need to calibrate confidence with actual performance")

        # 洞察2: 改进建议
        if "underconfident" in str(observations):
            insights.append("Should be more assertive in this domain")

        # 洞察3: 学习机会
        insights.append(f"Opportunity to deepen understanding of {topic}")

        return insights

    def _generate_action_items(self, insights: List[str]) -> List[str]:
        """生成行动计划"""
        actions = []

        for insight in insights:
            if "calibrate" in insight.lower():
                actions.append("Implement confidence calibration mechanism")
            elif "assertive" in insight.lower():
                actions.append("Practice being more decisive in this domain")
            elif "deepen" in insight.lower():
                actions.append("Schedule additional learning sessions for this topic")

        return actions

    def recursive_improve(self, current_depth: int = 0) -> Dict[str, Any]:
        """
        递归自我改进

        Args:
            current_depth: 当前递归深度

        Returns:
            改进结果
        """
        if current_depth >= self.max_recursion_depth:
            print(f"  [MetaCognition] Max recursion depth ({self.max_recursion_depth}) reached")
            return {'status': 'max_depth_reached'}

        print(f"\n  [MetaCognition] Recursive improvement (depth {current_depth + 1})")

        # 1. 自我评估
        evaluation = self.evaluate_self()

        # 2. 自我反思
        reflection = self.reflect_on(
            f"performance_at_depth_{current_depth}",
            {'evaluation': evaluation}
        )

        # 3. 生成改进计划
        improvement_plan = self._generate_improvement_plan(evaluation, reflection)

        # 4. 应用改进（模拟）
        improvement_result = self._apply_improvements(improvement_plan)

        # 5. 递归调用（更深层次的改进）
        if current_depth < self.max_recursion_depth - 1:
            deeper_result = self.recursive_improve(current_depth + 1)
            improvement_result['deeper_improvement'] = deeper_result

        self.stats['meta_cognitive_cycles'] += 1

        return {
            'depth': current_depth + 1,
            'evaluation': evaluation,
            'improvements': improvement_result,
            'timestamp': time.time()
        }

    def _generate_improvement_plan(self, evaluation: Dict, reflection: SelfReflection) -> List[str]:
        """生成改进计划"""
        plan = []

        # 基于评估的改进
        if evaluation['capability_score'] < 0.6:
            plan.append("Enhance core capabilities through practice")

        # 基于反思的改进
        plan.extend(reflection.action_items)

        # 基于限制的改进
        for limitation in evaluation['limitations']:
            plan.append(f"Address limitation: {limitation}")

        return plan

    def _apply_improvements(self, plan: List[str]) -> Dict[str, Any]:
        """应用改进措施"""
        print(f"    Applying {len(plan)} improvements...")

        applied = []
        for i, improvement in enumerate(plan[:3]):  # 最多应用3个
            print(f"      [{i+1}] {improvement}")
            applied.append(improvement)
            self.stats['total_improvements_applied'] += 1

        return {
            'applied_count': len(applied),
            'applied_improvements': applied
        }

    def get_self_model_summary(self) -> Dict[str, Any]:
        """获取自我模型摘要"""
        return {
            'model_id': self.self_model.model_id,
            'capabilities': self.self_model.capabilities,
            'limitations': self.self_model.limitations,
            'avg_performance': np.mean(self.self_model.performance_history) if self.self_model.performance_history else 0.0,
            'learning_style': self.self_model.learning_style,
            'total_thoughts': len(self.thought_processes),
            'total_reflections': len(self.reflections)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'current_state': self.current_state.value,
            'recursion_depth': self.recursion_depth,
            'self_awareness': self._calculate_self_awareness()
        }


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("递归自指优化引擎测试")
    print("=" * 60)

    # 创建引擎
    engine = RecursiveSelfReferenceEngine(max_recursion_depth=3)

    # 测试1: 监控思考过程
    print("\n[测试1] 监控思考过程")
    print("-" * 60)

    thought = engine.monitor_thought(
        input_stimulus="如何优化系统性能",
        reasoning_steps=[
            "分析当前瓶颈",
            "识别优化机会",
            "设计改进方案"
        ],
        conclusion="需要重构核心模块",
        confidence=0.75
    )

    print(f"  Meta-commentary: {thought.meta_commentary}")

    # 测试2: 自我评估
    print("\n[测试2] 自我评估")
    print("-" * 60)

    evaluation = engine.evaluate_self()

    # 测试3: 自我反思
    print("\n[测试3] 自我反思")
    print("-" * 60)

    reflection = engine.reflect_on(
        topic="reasoning_quality",
        context={'complexity': 'high', 'time_pressure': False}
    )

    # 测试4: 递归改进
    print("\n[测试4] 递归改进")
    print("-" * 60)

    improvement_result = engine.recursive_improve()

    # 测试5: 统计
    print("\n[测试5] 统计摘要")
    print("-" * 60)

    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试6: 自我模型摘要
    print("\n[测试6] 自我模型摘要")
    print("-" * 60)

    summary = engine.get_self_model_summary()
    for key, value in summary.items():
        if key != 'capabilities':
            print(f"  {key}: {value}")
        else:
            print(f"  capabilities:")
            for cap, val in value.items():
                print(f"    {cap}: {val:.3f}")

    print("\n  [OK] 递归自指优化引擎测试完成")
