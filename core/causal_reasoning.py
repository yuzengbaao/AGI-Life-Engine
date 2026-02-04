#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果推理引擎（Causal Reasoning Engine）
======================================

功能：实现真正的因果理解，而不只是模式匹配
基于：Judea Pearl的因果推理理论

核心能力：
1. 从观察中推断因果
2. 反事实推理
3. 干预预测
4. 解释生成

版本: 1.0.0
"""

import time
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class Event:
    """事件"""
    id: str
    type: str
    timestamp: float
    properties: Dict[str, Any]
    cause: Optional[str] = None  # 原因事件ID
    effect: Optional[str] = None  # 结果事件ID


@dataclass
class CausalRelation:
    """因果关系"""
    cause: str  # 原因事件ID
    effect: str  # 结果事件ID
    strength: float  # 强度 [0, 1]
    confidence: float  # 置信度 [0, 1]
    mechanism: Optional[str] = None  # 机制描述


class CausalGraph:
    """因果图"""

    def __init__(self):
        self.nodes = {}  # 事件节点
        self.edges = {}  # 因果边
        self.observations = []  # 观察历史

    def add_event(self, event: Event):
        """添加事件"""
        self.nodes[event.id] = event
        self.observations.append(event)

    def add_causal_relation(self, relation: CausalRelation):
        """添加因果关系"""
        key = (relation.cause, relation.effect)
        self.edges[key] = relation

    def get_causes(self, event_id: str) -> List[str]:
        """获取事件的所有原因"""
        return [edge.cause for edge in self.edges.values()
                if edge.effect == event_id]

    def get_effects(self, event_id: str) -> List[str]:
        """获取事件的所有结果"""
        return [edge.effect for edge in self.edges.values()
                if edge.cause == event_id]

    def find_path(self, start: str, end: str) -> List[str]:
        """查找因果路径"""
        visited = set()
        path = []

        def dfs(current):
            if current in visited:
                return False
            visited.add(current)
            path.append(current)

            if current == end:
                return True

            for neighbor in self.get_effects(current):
                if dfs(neighbor):
                    return True

            path.pop()
            return False

        dfs(start)
        return path if path and path[-1] == end else []


class CausalReasoningEngine:
    """
    因果推理引擎

    核心方法：
    1. infer_causality() - 从观察推断因果
    2. predict_intervention() - 预测干预效果
    3. explain() - 解释推理过程
    """

    def __init__(self):
        self.graph = CausalGraph()
        self.hypotheses = []
        self.confidence_threshold = 0.7

    def infer_causality(self, events: List[Event]) -> CausalGraph:
        """
        从事件序列推断因果关系

        使用三条标准（来自Judea Pearl）：
        1. 时间顺序：原因必须在结果之前
        2. 协变性：原因变化时结果也变化
        3. 排除混淆：控制混杂变量
        """
        print(f"\n  [CausalEngine] [ANALYZE] 分析 {len(events)} 个事件")

        # 1. 时间顺序检查
        temporal_relations = self._check_temporal_precedence(events)
        print(f"    时间关系: 发现 {len(temporal_relations)} 个候选")

        # 2. 协变性检查
        covariation_relations = self._check_covariation(events, temporal_relations)
        print(f"    协变关系: {len(covariation_relations)} 个通过")

        # 3. 排除混淆
        final_relations = self._check_confounding(events, covariation_relations)
        print(f"    因果关系: {len(final_relations)} 个确认")

        # 构建因果图
        for relation in final_relations:
            self.graph.add_causal_relation(relation)

        return self.graph

    def _check_temporal_precedence(self, events: List[Event]) -> List[Tuple[str, str]]:
        """检查时间顺序"""
        relations = []

        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # 对于每对事件，检查时间顺序
        for i, e1 in enumerate(sorted_events):
            for e2 in sorted_events[i+1:]:
                # e1 在 e2 之前
                time_gap = e2.timestamp - e1.timestamp

                # 只考虑合理的时间窗口（1分钟内）
                if 0 < time_gap < 60:
                    relations.append((e1.id, e2.id))

        return relations

    def _check_covariation(self, events: List[Event],
                          candidates: List[Tuple[str, str]]) -> List[CausalRelation]:
        """检查协变性"""
        relations = []

        event_dict = {e.id: e for e in events}

        for cause_id, effect_id in candidates:
            if cause_id not in event_dict or effect_id not in event_dict:
                continue

            cause = event_dict[cause_id]
            effect = event_dict[effect_id]

            # 计算协变强度
            strength = self._calculate_covariation_strength(cause, effect, events)

            if strength > 0.3:  # 阈值
                relations.append(CausalRelation(
                    cause=cause_id,
                    effect=effect_id,
                    strength=strength,
                    confidence=0.5,  # 初始置信度
                    mechanism="协变观察"
                ))

        return relations

    def _calculate_covariation_strength(self, cause: Event, effect: Event,
                                       all_events: List[Event]) -> float:
        """
        计算协变强度

        简化版本：基于类型共现
        """
        # 统计cause类型出现后effect类型出现的频率
        cause_type = cause.type
        effect_type = effect.type

        cause_count = sum(1 for e in all_events if e.type == cause_type)
        effect_after_count = 0

        for i, e in enumerate(all_events):
            if e.type == cause_type:
                # 检查后续事件
                for later_e in all_events[i+1:i+5]:  # 5个事件窗口
                    if later_e.type == effect_type:
                        effect_after_count += 1
                        break

        if cause_count == 0:
            return 0.0

        # 协变强度 = effect出现频率
        return effect_after_count / cause_count

    def _check_confounding(self, events: List[Event],
                          relations: List[CausalRelation]) -> List[CausalRelation]:
        """排除混淆因素"""
        final_relations = []

        for relation in relations:
            # 检查是否有混杂变量
            is_confounded = False

            cause = next(e for e in events if e.id == relation.cause)
            effect = next(e for e in events if e.id == relation.effect)

            # 检查是否同时受第三个因素影响
            for other in events:
                if other.id not in [relation.cause, relation.effect]:
                    if self._is_confounder(other, cause, effect, events):
                        is_confounded = True
                        break

            if not is_confounded:
                # 提升置信度
                relation.confidence = min(relation.confidence + 0.3, 1.0)
                final_relations.append(relation)

        return final_relations

    def _is_confounder(self, potential_confounder: Event,
                      cause: Event, effect: Event,
                      all_events: List[Event]) -> bool:
        """检查是否是混杂因素"""
        # 简化版本：检查是否同时影响cause和effect
        # 在真实系统中，这需要更复杂的条件独立测试

        # 这里只检查时间顺序
        return (potential_confounder.timestamp < cause.timestamp and
                potential_confounder.timestamp < effect.timestamp)

    def predict_intervention(self, intervention: Dict[str, Any]) -> Dict[str, float]:
        """
        预测干预效果：do(X=x)

        使用do-calculus
        """
        print(f"\n  [CausalEngine] [PREDICT] 预测干预: {intervention}")

        # 原始预测（无干预）
        baseline = self._predict_baseline()

        # 执行干预（修改因果图）
        modified_graph = self._do_intervention(intervention)

        # 干预后的预测
        post_intervention = self._predict_with_graph(modified_graph)

        # 计算因果效应
        effect = self._calculate_causal_effect(baseline, post_intervention)

        print(f"    基线预测: {baseline}")
        print(f"    干预预测: {post_intervention}")
        print(f"    因果效应: {effect:.3f}")

        return {
            'baseline': baseline,
            'intervention': intervention,
            'post_intervention': post_intervention,
            'effect': effect
        }

    def _predict_baseline(self) -> Dict[str, float]:
        """基线预测（当前系统状态）"""
        # 简化版本：基于最近事件预测
        predictions = {}

        if not self.graph.observations:
            return predictions

        # 获取最近事件
        recent = self.graph.observations[-1]

        # 预测其直接结果
        effects = self.graph.get_effects(recent.id)
        for effect_id in effects:
            if effect_id in self.graph.nodes:
                effect = self.graph.nodes[effect_id]
                predictions[effect_id] = 0.7  # 简化的概率

        return predictions

    def _do_intervention(self, intervention: Dict[str, Any]) -> CausalGraph:
        """执行do-算子"""
        # 创建修改后的因果图
        new_graph = CausalGraph()
        new_graph.nodes = self.graph.nodes.copy()

        # 复制边，但删除被干预变量的入边
        for (cause, effect), relation in self.graph.edges.items():
            if cause not in intervention:
                new_graph.edges[(cause, effect)] = relation

        # 固定被干预变量的值
        for var, value in intervention.items():
            if var in new_graph.nodes:
                new_graph.nodes[var].properties['intervened'] = True
                new_graph.nodes[var].properties['value'] = value

        return new_graph

    def _predict_with_graph(self, graph: CausalGraph) -> Dict[str, float]:
        """在指定因果图上预测"""
        # 简化版本：返回可达的节点
        predictions = {}

        if not graph.observations:
            return predictions

        recent = graph.observations[-1]

        # 广度优先搜索
        visited = set()
        queue = [recent.id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            # 添加直接结果
            for effect_id in graph.get_effects(current):
                if effect_id not in visited:
                    predictions[effect_id] = 0.5  # 基础概率
                    queue.append(effect_id)

        return predictions

    def _calculate_causal_effect(self, baseline: Dict[str, float],
                                post_intervention: Dict[str, float]) -> float:
        """计算因果效应"""
        if not baseline or not post_intervention:
            return 0.0

        # 平均效应
        effects = []
        for key in baseline:
            if key in post_intervention:
                effects.append(abs(post_intervention[key] - baseline[key]))

        return np.mean(effects) if effects else 0.0

    def explain_reasoning(self, question: str) -> str:
        """
        解释推理过程

        返回因果路径和机制
        """
        print(f"\n  [CausalEngine] [EXPLAIN] 解释: {question}")

        # 解析问题（简化版本）
        # 在真实系统中，需要NLP解析

        # 查找相关的因果路径
        relevant_paths = []

        for event_id in self.graph.nodes:
            event = self.graph.nodes[event_id]

            # 简单的关键词匹配
            if any(keyword in event.type.lower() for keyword in question.lower().split()):
                # 查找这个事件的原因链
                causes = self.graph.get_causes(event_id)
                for cause_id in causes:
                    path = self.graph.find_path(cause_id, event_id)
                    if path:
                        relevant_paths.append(path)

        if not relevant_paths:
            return "无法找到相关因果路径"

        # 构建解释
        explanation = "因果推理路径:\n"

        for i, path in enumerate(relevant_paths[:3], 1):  # 最多3条路径
            explanation += f"\n路径{i}:\n"
            for j, event_id in enumerate(path):
                if event_id in self.graph.nodes:
                    event = self.graph.nodes[event_id]
                    explanation += f"  {j}. {event.type} ({event.timestamp})\n"

                    # 添加机制
                    if j < len(path) - 1:
                        for edge in self.graph.edges.values():
                            if edge.cause == event_id and edge.effect == path[j+1]:
                                if edge.mechanism:
                                    explanation += f"     ↓ {edge.mechanism} (强度={edge.strength:.2f})\n"

        return explanation

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_events': len(self.graph.nodes),
            'causal_relations': len(self.graph.edges),
            'observations': len(self.graph.observations),
            'avg_confidence': np.mean([e.confidence for e in self.graph.edges.values()]) if self.graph.edges else 0.0
        }


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("因果推理引擎测试")
    print("=" * 60)

    # 创建引擎
    engine = CausalReasoningEngine()

    # 模拟事件序列
    print("\n[场景] 用户启动AGI系统")
    events = [
        Event(id="E1", type="user_click", timestamp=time.time(),
              properties={"button": "start"}),
        Event(id="E2", type="system_init", timestamp=time.time()+0.5,
              properties={"component": "M1M4"}),
        Event(id="E3", type="memory_created", timestamp=time.time()+1.0,
              properties={"component": "M4", "count": 4}),
        Event(id="E4", type="insight_generated", timestamp=time.time()+2.0,
              properties={"iteration": 108}),
    ]

    # 添加事件
    for event in events:
        engine.graph.add_event(event)

    # 推断因果
    causal_graph = engine.infer_causality(events)

    # 预测干预
    intervention = {"system_init": {"fast_mode": True}}
    prediction = engine.predict_intervention(intervention)

    # 解释推理
    explanation = engine.explain_reasoning("为什么生成了洞察？")
    print(f"\n{explanation}")

    # 统计
    stats = engine.get_statistics()
    print(f"\n[统计] {stats}")
