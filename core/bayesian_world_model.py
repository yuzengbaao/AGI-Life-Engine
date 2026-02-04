#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一世界模型（Unified World Model）
====================================

功能：贝叶斯世界模型，实现统一的预测与干预
基于：Judea Pearl的因果推理 + 贝叶斯推理

核心能力：
1. 贝叶斯信念更新 P(H|E) = P(E|H)*P(H)/P(E)
2. 干预预测 do(X=x)
3. 统一状态表征
4. 不确定性量化

版本: 1.0.0
"""

import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


class StateVariable(Enum):
    """状态变量类型"""
    DISCRETE = "discrete"    # 离散变量
    CONTINUOUS = "continuous" # 连续变量
    BINARY = "binary"        # 二元变量


@dataclass
class BeliefState:
    """信念状态"""
    variable: str
    value: Any
    probability: float  # P(variable=value)
    confidence: float   # 置信度 [0, 1]
    last_update: float
    evidence_count: int = 0

    def __repr__(self):
        return f"Belief({self.variable}={self.value}, p={self.probability:.3f}, conf={self.confidence:.2f})"


@dataclass
class CausalLink:
    """因果链接"""
    cause: str
    effect: str
    strength: float  # 因果强度 [0, 1]
    mechanism: str   # 机制描述
    confidence: float  # 置信度

    def __repr__(self):
        return f"Causal({self.cause} -> {self.effect}, strength={self.strength:.2f})"


@dataclass
class Intervention:
    """干预"""
    variable: str
    value: Any
    timestamp: float
    effect_prediction: Dict[str, float] = field(default_factory=dict)
    actual_effect: Dict[str, float] = field(default_factory=dict)


class BayesianWorldModel:
    """
    贝叶斯世界模型

    核心功能：
    1. 维护信念状态（贝叶斯更新）
    2. 存储因果关系
    3. 预测干预效果（do-calculus）
    4. 生成解释
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        初始化世界模型

        Args:
            learning_rate: 学习率（控制信念更新速度）
        """
        self.learning_rate = learning_rate

        # 信念状态存储
        self.beliefs: Dict[str, BeliefState] = {}

        # 因果关系网络
        self.causal_graph: Dict[str, List[CausalLink]] = defaultdict(list)

        # 干预历史
        self.interventions: List[Intervention] = []

        # 统计信息
        self.stats = {
            'total_observations': 0,
            'total_interventions': 0,
            'belief_updates': 0,
            'predictions_made': 0,
            'prediction_accuracy': 0.0
        }

    def observe(self, variable: str, value: Any, confidence: float = 0.8) -> BeliefState:
        """
        观察变量，更新信念

        贝叶斯更新: P(H|E) = P(E|H)*P(H)/P(E)

        Args:
            variable: 变量名
            value: 观察值
            confidence: 观察置信度

        Returns:
            更新后的信念状态
        """
        self.stats['total_observations'] += 1

        current_time = time.time()

        # 如果变量不存在，创建新信念
        if variable not in self.beliefs:
            self.beliefs[variable] = BeliefState(
                variable=variable,
                value=value,
                probability=0.5,  # 初始概率
                confidence=confidence,
                last_update=current_time,
                evidence_count=1
            )
            return self.beliefs[variable]

        # 获取当前信念
        belief = self.beliefs[variable]

        # 贝叶斯更新
        # P(H|E) = P(E|H)*P(H)/P(E)
        # 简化版本：基于似然性更新

        # 计算似然性 P(E|H)
        likelihood = self._compute_likelihood(belief.value, value)

        # 更新概率
        prior = belief.probability
        posterior = (likelihood * prior) / ((likelihood * prior) + (1 - likelihood) * (1 - prior))

        # 更新信念
        belief.probability = posterior
        belief.value = value  # 更新为最新观察值
        belief.confidence = min(belief.confidence + self.learning_rate * confidence, 1.0)
        belief.last_update = current_time
        belief.evidence_count += 1

        self.stats['belief_updates'] += 1

        return belief

    def _compute_likelihood(self, old_value: Any, new_value: Any) -> float:
        """
        计算似然性 P(E|H)

        简化版本：基于值相似性
        """
        if old_value == new_value:
            return 0.9  # 高似然性
        elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            # 连续值：计算相似度
            diff = abs(old_value - new_value)
            max_val = max(abs(old_value), abs(new_value), 1.0)
            similarity = 1.0 - (diff / max_val)
            return max(0.1, similarity)
        else:
            return 0.3  # 低似然性

    def add_causal_link(self, cause: str, effect: str, strength: float = 0.5,
                        mechanism: str = "unknown", confidence: float = 0.5):
        """
        添加因果链接

        Args:
            cause: 原因变量
            effect: 结果变量
            strength: 因果强度 [0, 1]
            mechanism: 机制描述
            confidence: 置信度
        """
        link = CausalLink(
            cause=cause,
            effect=effect,
            strength=strength,
            mechanism=mechanism,
            confidence=confidence
        )

        self.causal_graph[cause].append(link)

        print(f"  [WorldModel] Causal link added: {link}")

    def get_belief(self, variable: str) -> Optional[BeliefState]:
        """获取变量信念"""
        return self.beliefs.get(variable)

    def predict(self, query: str, context: Optional[Dict] = None) -> Tuple[Any, float]:
        """
        预测变量值

        Args:
            query: 查询变量
            context: 上下文信息

        Returns:
            (predicted_value, confidence)
        """
        self.stats['predictions_made'] += 1

        # 如果有直接信念，返回
        if query in self.beliefs:
            belief = self.beliefs[query]
            return belief.value, belief.confidence

        # 如果没有直接信念，尝试通过因果链推理
        if context:
            return self._predict_via_causal_chain(query, context)

        # 无法预测
        return None, 0.0

    def _predict_via_causal_chain(self, query: str, context: Dict) -> Tuple[Any, float]:
        """通过因果链预测"""
        # 查找影响query的原因
        possible_causes = []
        for cause, links in self.causal_graph.items():
            for link in links:
                if link.effect == query and cause in context:
                    possible_causes.append((link, context[cause]))

        if not possible_causes:
            return None, 0.0

        # 简化的预测：基于最强的因果链接
        best_link, cause_value = max(possible_causes, key=lambda x: x[0].strength)

        # 预测值 = 原因值 * 因果强度
        if isinstance(cause_value, (int, float)):
            predicted = cause_value * best_link.strength
            confidence = best_link.confidence * best_link.strength
            return predicted, confidence
        else:
            return cause_value, best_link.confidence * best_link.strength

    def intervene(self, variable: str, value: Any) -> Intervention:
        """
        执行干预 do(X=x)

        干预与观察不同，它会切断所有入边并固定变量值

        Args:
            variable: 干预变量
            value: 干预值

        Returns:
            干预对象
        """
        self.stats['total_interventions'] += 1

        # 创建干预
        intervention = Intervention(
            variable=variable,
            value=value,
            timestamp=time.time()
        )

        # 预测干预效果
        intervention.effect_prediction = self._predict_intervention_effects(variable, value)

        # 执行干预：固定变量值
        if variable in self.beliefs:
            self.beliefs[variable].value = value
            self.beliefs[variable].probability = 1.0  # 干预后确定性为1
            self.beliefs[variable].confidence = 1.0
        else:
            self.beliefs[variable] = BeliefState(
                variable=variable,
                value=value,
                probability=1.0,
                confidence=1.0,
                last_update=time.time(),
                evidence_count=1
            )

        # 记录干预
        self.interventions.append(intervention)

        print(f"  [WorldModel] Intervention: do({variable}={value})")
        print(f"    Predicted effects: {intervention.effect_prediction}")

        return intervention

    def _predict_intervention_effects(self, variable: str, value: Any) -> Dict[str, float]:
        """预测干预效果"""
        effects = {}

        # 查找所有直接效果
        if variable in self.causal_graph:
            for link in self.causal_graph[variable]:
                # 简化的效果预测
                if isinstance(value, (int, float)):
                    effect_value = value * link.strength
                else:
                    effect_value = value

                effects[link.effect] = {
                    'value': effect_value,
                    'confidence': link.confidence * link.strength
                }

        return effects

    def explain(self, query: str) -> str:
        """
        解释推理过程

        Args:
            query: 查询内容

        Returns:
            解释文本
        """
        lines = [f"[WorldModel] Explanation for: {query}\n"]

        # 当前信念
        if query in self.beliefs:
            belief = self.beliefs[query]
            lines.append(f"Current belief:")
            lines.append(f"  Value: {belief.value}")
            lines.append(f"  Probability: {belief.probability:.3f}")
            lines.append(f"  Confidence: {belief.confidence:.3f}")
            lines.append(f"  Evidence count: {belief.evidence_count}")
            lines.append("")

        # 因果关系
        causes = []
        effects = []

        for cause, links in self.causal_graph.items():
            for link in links:
                if link.effect == query:
                    causes.append(f"  - {cause} (strength={link.strength:.2f}, {link.mechanism})")
                elif link.cause == query:
                    effects.append(f"  - {link.effect} (strength={link.strength:.2f}, {link.mechanism})")

        if causes:
            lines.append("Causes:")
            lines.extend(causes)
            lines.append("")

        if effects:
            lines.append("Effects:")
            lines.extend(effects)
            lines.append("")

        # 干预历史
        relevant_interventions = [inv for inv in self.interventions if inv.variable == query]
        if relevant_interventions:
            lines.append(f"Recent interventions ({len(relevant_interventions)}):")
            for inv in relevant_interventions[-3:]:  # 最近3次
                lines.append(f"  - do({inv.variable}={inv.value}) at {inv.timestamp}")

        return "\n".join(lines)

    def get_state_summary(self) -> Dict[str, Any]:
        """获取世界模型状态摘要"""
        return {
            'total_beliefs': len(self.beliefs),
            'total_causal_links': sum(len(links) for links in self.causal_graph.values()),
            'total_interventions': len(self.interventions),
            'avg_belief_confidence': np.mean([b.confidence for b in self.beliefs.values()]) if self.beliefs else 0.0,
            'high_confidence_beliefs': sum(1 for b in self.beliefs.values() if b.confidence > 0.8),
            'stats': self.stats
        }

    def get_all_beliefs(self) -> Dict[str, BeliefState]:
        """获取所有信念"""
        return self.beliefs.copy()

    def get_causal_graph(self) -> Dict[str, List[CausalLink]]:
        """获取因果图"""
        return dict(self.causal_graph)

    def simulate(self, steps: int = 10) -> List[Dict[str, Any]]:
        """
        模拟世界演化

        Args:
            steps: 模拟步数

        Returns:
            模拟历史
        """
        history = []

        for step in range(steps):
            # 随机选择一个变量进行更新
            if not self.beliefs:
                break

            variable = random.choice(list(self.beliefs.keys()))
            current_belief = self.beliefs[variable]

            # 模拟观察（添加噪声）
            if isinstance(current_belief.value, (int, float)):
                noise = random.gauss(0, 0.1 * abs(current_belief.value))
                new_value = current_belief.value + noise
            else:
                new_value = current_belief.value

            # 更新信念
            updated_belief = self.observe(variable, new_value, confidence=0.7)

            history.append({
                'step': step,
                'variable': variable,
                'old_value': current_belief.value,
                'new_value': new_value,
                'belief': updated_belief
            })

        return history


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("贝叶斯世界模型测试")
    print("=" * 60)

    # 创建世界模型
    world = BayesianWorldModel(learning_rate=0.1)

    # 测试1: 观察更新
    print("\n[测试1] 贝叶斯信念更新")
    print("-" * 60)

    beliefs = []
    for i in range(5):
        value = 0.5 + random.gauss(0, 0.1)
        belief = world.observe("temperature", value, confidence=0.8)
        beliefs.append(belief)
        print(f"  观察[{i}]: temperature={value:.3f} -> P={belief.probability:.3f}, conf={belief.confidence:.3f}")

    # 测试2: 因果关系
    print("\n[测试2] 因果关系学习")
    print("-" * 60)

    world.add_causal_link("temperature", "ice_melting", strength=0.8, mechanism="热力学")
    world.add_causal_link("pressure", "boiling_point", strength=0.6, mechanism="物理定律")

    # 观察原因变量
    world.observe("temperature", 80, confidence=0.9)

    # 预测结果
    predicted, conf = world.predict("ice_melting", context={"temperature": 80})
    print(f"  预测: ice_melting={predicted:.2f}, confidence={conf:.2f}")

    # 测试3: 干预
    print("\n[测试3] 干预预测 do-calculus")
    print("-" * 60)

    intervention = world.intervene("temperature", 100)
    print(f"  干预: do(temperature=100)")
    print(f"  预测效果: {intervention.effect_prediction}")

    # 测试4: 解释
    print("\n[测试4] 解释生成")
    print("-" * 60)

    explanation = world.explain("temperature")
    print(explanation)

    # 测试5: 状态摘要
    print("\n[测试5] 状态摘要")
    print("-" * 60)

    summary = world.get_state_summary()
    for key, value in summary.items():
        if key != 'stats':
            print(f"  {key}: {value}")
        else:
            print(f"  stats:")
            for k, v in value.items():
                print(f"    {k}: {v}")

    # 测试6: 模拟演化
    print("\n[测试6] 世界演化模拟")
    print("-" * 60)

    history = world.simulate(steps=5)
    for entry in history:
        print(f"  Step {entry['step']}: {entry['variable']} {entry['old_value']:.2f} -> {entry['new_value']:.2f}")
