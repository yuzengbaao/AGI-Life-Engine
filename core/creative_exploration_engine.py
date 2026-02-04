#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创造性探索引擎（Creative Exploration Engine）
==========================================

功能：实现创造性问题解决，打破常规思维
基于：组合创造力理论 + 类比推理 + 受控随机性

核心能力：
1. 类比推理（跨域知识迁移）
2. 概念组合（新颖概念生成）
3. 受控随机探索（惊奇值驱动）
4. 发散思维（打破思维定势）

版本: 1.0.0
"""

import time
import random
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ExplorationMode(Enum):
    """探索模式"""
    ANALOGICAL = "analogical"     # 类比推理
    COMBINATORIAL = "combinatorial" # 概念组合
    STOCHASTIC = "stochastic"      # 随机探索
    DIVERGENT = "divergent"        # 发散思维


@dataclass
class Concept:
    """概念"""
    id: str
    name: str
    domain: str  # 所属领域
    attributes: Dict[str, Any]  # 属性
    associations: List[str]  # 关联概念ID
    creation_time: float
    novelty_score: float = 0.5  # 新颖度 [0, 1]

    def __repr__(self):
        return f"Concept({self.name}, domain={self.domain})"


@dataclass
class ExplorationResult:
    """探索结果"""
    result_id: str
    mode: ExplorationMode
    input_query: str
    output_idea: str
    novelty_score: float
    feasibility_score: float
    value_score: float
    confidence: float
    timestamp: float
    reasoning_trace: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"Exploration({self.mode.value}, novelty={self.novelty_score:.2f})"


class CreativeExplorationEngine:
    """
    创造性探索引擎

    核心功能：
    1. 类比推理：从已知领域映射到新领域
    2. 概念组合：组合现有概念生成新概念
    3. 随机探索：受惊奇值驱动的探索
    4. 发散思维：生成多样化解决方案
    """

    def __init__(self, temperature: float = 0.7, enable_adaptive_temperature: bool = True):
        """
        初始化探索引擎

        Args:
            temperature: 创造温度 [0, 1]（已废弃，保留向后兼容）
                        0 = 保守（利用最佳）
                        1 = 激进（高度探索）
            enable_adaptive_temperature: 是否启用自适应温度（Task 11）
        """
        self.enable_adaptive_temperature = enable_adaptive_temperature

        # 初始化自适应温度控制器（Task 11）
        if enable_adaptive_temperature:
            from core.adaptive_temperature import get_temperature_controller
            self.temperature_controller = get_temperature_controller()
            self.temperature = self.temperature_controller.base_temperature
            logger.info("[创造性探索] 自适应温度控制器已启用")
        else:
            self.temperature_controller = None
            self.temperature = temperature

        # 概念知识库
        self.concepts: Dict[str, Concept] = {}
        self.concept_index: Dict[str, List[str]] = defaultdict(list)  # domain -> concept_ids

        # 探索历史
        self.exploration_history: List[ExplorationResult] = []

        # 跨域映射库（类比推理用）
        self.cross_domain_mappings: Dict[Tuple[str, str], List[str]] = {}

        # 统计信息
        self.stats = {
            'total_explorations': 0,
            'novel_ideas_generated': 0,
            'analogical_reasoning_used': 0,
            'combinatorial_creativity_used': 0,
            'stochastic_exploration_used': 0
        }

        # 初始化基础概念
        self._initialize_base_concepts()

    def _initialize_base_concepts(self):
        """初始化基础概念库"""
        base_concepts = [
            # 计算机科学领域
            Concept("c_recursion", "recursion", "computer_science",
                   {"complexity": "O(n)", "property": "self-reference"},
                   [], time.time()),
            Concept("c_neural_network", "neural_network", "computer_science",
                   {"type": "learning", "property": "parallel_processing"},
                   [], time.time()),
            Concept("c_algorithm", "algorithm", "computer_science",
                   {"property": "step_by_step", "complexity": "variable"},
                   [], time.time()),

            # 生物学领域
            Concept("b_evolution", "evolution", "biology",
                   {"mechanism": "selection", "property": "adaptation"},
                   [], time.time()),
            Concept("b_dna", "DNA", "biology",
                   {"structure": "double_helix", "property": "information_storage"},
                   [], time.time()),
            Concept("b_brain", "brain", "biology",
                   {"type": "neural", "property": "plasticity"},
                   [], time.time()),

            # 物理学领域
            Concept("p_entropy", "entropy", "physics",
                   {"property": "disorder", "tendency": "increase"},
                   [], time.time()),
            Concept("p_energy", "energy", "physics",
                   {"property": "conserved", "forms": "multiple"},
                   [], time.time()),
            Concept("p_quantum", "quantum", "physics",
                   {"property": "superposition", "scale": "microscopic"},
                   [], time.time()),

            # 认知科学领域
            Concept("m_memory", "memory", "cognitive_science",
                   {"type": "storage", "property": "retrieval"},
                   [], time.time()),
            Concept("m_attention", "attention", "cognitive_science",
                   {"property": "selective", "resource": "limited"},
                   [], time.time()),
            Concept("m_consciousness", "consciousness", "cognitive_science",
                   {"property": "subjective", "level": "varying"},
                   [], time.time()),
        ]

        for concept in base_concepts:
            self.concepts[concept.id] = concept
            self.concept_index[concept.domain].append(concept.id)

    def explore(self, query: str, context: Optional[Dict] = None,
                mode: Optional[ExplorationMode] = None) -> ExplorationResult:
        """
        执行创造性探索

        Args:
            query: 探索查询
            context: 上下文信息
            mode: 探索模式（None表示自动选择）

        Returns:
            探索结果
        """
        self.stats['total_explorations'] += 1

        # Task 11: 获取动态温度
        current_temperature = self._get_temperature(context)
        logger.debug(f"[创造性探索] 当前温度: {current_temperature:.2f}")

        # 计算惊奇值（基于不确定性）
        surprise = self._calculate_surprise(context)

        # 自动选择模式（如果未指定）
        if mode is None:
            mode = self._select_exploration_mode(surprise)

        # 根据模式执行探索
        if mode == ExplorationMode.ANALOGICAL:
            result = self._analogical_exploration(query, context)
        elif mode == ExplorationMode.COMBINATORIAL:
            result = self._combinatorial_exploration(query, context)
        elif mode == ExplorationMode.STOCHASTIC:
            result = self._stochastic_exploration(query, context)
        else:  # DIVERGENT
            result = self._divergent_exploration(query, context)

        self.exploration_history.append(result)

        # Task 11: 记录温度使用结果（用于自适应调整）
        if self.enable_adaptive_temperature and self.temperature_controller is not None:
            success = result.value_score > 0.6 and result.feasibility_score > 0.4
            self.temperature_controller.record_outcome(
                temperature=current_temperature,
                context=context or {},
                success=success,
                creativity_score=result.novelty_score,
                feasibility_score=result.feasibility_score
            )

        # 更新统计
        if result.novelty_score > 0.7:
            self.stats['novel_ideas_generated'] += 1

        if mode == ExplorationMode.ANALOGICAL:
            self.stats['analogical_reasoning_used'] += 1
        elif mode == ExplorationMode.COMBINATORIAL:
            self.stats['combinatorial_creativity_used'] += 1
        elif mode == ExplorationMode.STOCHASTIC:
            self.stats['stochastic_exploration_used'] += 1

        return result

    def _get_temperature(self, context: Optional[Dict] = None) -> float:
        """
        获取当前温度（Task 11：支持自适应温度）

        Args:
            context: 上下文信息

        Returns:
            温度值
        """
        if self.enable_adaptive_temperature and self.temperature_controller is not None:
            # 使用自适应温度控制器
            temperature = self.temperature_controller.get_temperature(context or {})
            return temperature
        else:
            # 使用固定温度（向后兼容）
            return self.temperature

    def _calculate_surprise(self, context: Optional[Dict]) -> float:
        """计算惊奇值"""
        if not context:
            return 0.5

        # 惊奇值 = 熵 * 好奇心
        entropy = context.get('entropy', 0.5)
        curiosity = context.get('curiosity', 0.5)

        return entropy * curiosity

    def _select_exploration_mode(self, surprise: float) -> ExplorationMode:
        """根据惊奇值选择探索模式"""
        # 低惊奇 -> 利用已知（类比推理）
        if surprise < 0.3:
            return ExplorationMode.ANALOGICAL

        # 中惊奇 -> 组合创新（概念组合）
        elif surprise < 0.7:
            return ExplorationMode.COMBINATORIAL

        # 高惊奇 -> 随机探索（打破常规）
        else:
            return ExplorationMode.STOCHASTIC

    def _analogical_exploration(self, query: str, context: Optional[Dict]) -> ExplorationResult:
        """类比推理探索"""
        print(f"  [CreativeEngine] Using analogical reasoning for: {query}")

        # 识别查询领域
        source_domain = self._identify_domain(query)

        # 选择目标领域（不同领域）
        target_domains = [d for d in self.concept_index.keys() if d != source_domain]

        if not target_domains:
            target_domain = source_domain
        else:
            target_domain = random.choice(target_domains)

        # 获取源领域和目标领域的概念
        source_concepts = self.concept_index.get(source_domain, [])
        target_concepts = self.concept_index.get(target_domain, [])

        if not source_concepts or not target_concepts:
            return self._fallback_result(query, ExplorationMode.ANALOGICAL)

        # 选择源概念和目标概念
        source_concept_id = random.choice(source_concepts)
        target_concept_id = random.choice(target_concepts)

        source_concept = self.concepts[source_concept_id]
        target_concept = self.concepts[target_concept_id]

        # 生成类比映射
        idea = self._generate_analogy(source_concept, target_concept, query)

        # 计算评分
        novelty = self._calculate_analogy_novelty(source_domain, target_domain)
        feasibility = min(0.8, novelty + 0.3)  # 类比通常较可行
        value = novelty * feasibility

        return ExplorationResult(
            result_id=f"expl_{int(time.time() * 1000)}",
            mode=ExplorationMode.ANALOGICAL,
            input_query=query,
            output_idea=idea,
            novelty_score=novelty,
            feasibility_score=feasibility,
            value_score=value,
            confidence=min(0.9, feasibility + 0.2),
            timestamp=time.time(),
            reasoning_trace=[
                f"Identified source domain: {source_domain}",
                f"Selected target domain: {target_domain}",
                f"Source concept: {source_concept.name}",
                f"Target concept: {target_concept.name}",
                f"Generated analogy mapping"
            ]
        )

    def _combinatorial_exploration(self, query: str, context: Optional[Dict]) -> ExplorationResult:
        """概念组合探索"""
        print(f"  [CreativeEngine] Using combinatorial creativity for: {query}")

        # Task 11: 获取动态温度
        temperature = self._get_temperature(context)

        # 随机选择2-3个概念
        all_concept_ids = list(self.concepts.keys())
        if len(all_concept_ids) < 2:
            return self._fallback_result(query, ExplorationMode.COMBINATORIAL)

        # 基于温度选择概念数量
        num_concepts = int(2 + temperature * 2)  # 2-4个概念
        selected_concept_ids = random.sample(all_concept_ids, min(num_concepts, len(all_concept_ids)))

        selected_concepts = [self.concepts[cid] for cid in selected_concept_ids]

        # 生成组合概念
        combined_idea = self._generate_combination(selected_concepts, query)

        # 计算评分
        domains = [c.domain for c in selected_concepts]
        novelty = len(set(domains)) / len(selected_concepts)  # 跨域组合更新颖
        feasibility = max(0.2, 1.0 - novelty * 0.5)  # 更新颖的可行性较低
        value = novelty * feasibility * (1 + temperature * 0.5)

        return ExplorationResult(
            result_id=f"expl_{int(time.time() * 1000)}",
            mode=ExplorationMode.COMBINATORIAL,
            input_query=query,
            output_idea=combined_idea,
            novelty_score=novelty,
            feasibility_score=feasibility,
            value_score=value,
            confidence=feasibility,
            timestamp=time.time(),
            reasoning_trace=[
                f"Selected {len(selected_concepts)} concepts: {[c.name for c in selected_concepts]}",
                f"Domains: {domains}",
                f"Temperature: {temperature:.2f}",
                f"Generated combination"
            ]
        )

    def _stochastic_exploration(self, query: str, context: Optional[Dict]) -> ExplorationResult:
        """随机探索（打破常规）"""
        print(f"  [CreativeEngine] Using stochastic exploration for: {query}")

        # Task 11: 获取动态温度
        temperature = self._get_temperature(context)

        # 高度随机性
        random_seed = random.randint(0, 0xFFFFFF)

        # 生成随机概念
        random_concept = f"RandomConcept_{random_seed:06x}"

        # 随机选择领域
        all_domains = list(self.concept_index.keys())
        random_domain = random.choice(all_domains) if all_domains else "unknown"

        # 生成随机属性
        random_attributes = {
            "entropy": random.random(),
            "complexity": random.choice(["low", "medium", "high"]),
            "novelty": random.random()
        }

        # 生成创新想法
        novel_idea = f"What if we apply {random_concept} from {random_domain} with properties {random_attributes} to solve: {query}?"

        # 计算评分
        novelty = 0.9 + random.random() * 0.1  # 非常新颖
        feasibility = random.random() * 0.5  # 但可行性不确定
        value = novelty * feasibility * temperature  # 价值取决于温度

        return ExplorationResult(
            result_id=f"expl_{int(time.time() * 1000)}",
            mode=ExplorationMode.STOCHASTIC,
            input_query=query,
            output_idea=novel_idea,
            novelty_score=novelty,
            feasibility_score=feasibility,
            value_score=value,
            confidence=0.3,  # 低置信度
            timestamp=time.time(),
            reasoning_trace=[
                f"Random seed: {random_seed}",
                f"Random domain: {random_domain}",
                f"Generated novel concept: {random_concept}",
                f"Temperature applied: {temperature:.2f}"
            ]
        )

    def _divergent_exploration(self, query: str, context: Optional[Dict]) -> ExplorationResult:
        """发散思维（生成多个解决方案）"""
        print(f"  [CreativeEngine] Using divergent thinking for: {query}")

        # 生成3-5个不同的想法
        ideas = []

        # 想法1: 类比
        idea1 = self._analogical_exploration(query, context)
        ideas.append(idea1)

        # 想法2: 组合
        idea2 = self._combinatorial_exploration(query, context)
        ideas.append(idea2)

        # 想法3: 反向思维
        reverse_idea = f"Inverse approach: What if we do the opposite of typical solutions for {query}?"
        idea3 = ExplorationResult(
            result_id=f"expl_{int(time.time() * 1000)}_3",
            mode=ExplorationMode.DIVERGENT,
            input_query=query,
            output_idea=reverse_idea,
            novelty_score=0.7,
            feasibility_score=0.5,
            value_score=0.35,
            confidence=0.5,
            timestamp=time.time(),
            reasoning_trace=["Applied reverse thinking"]
        )
        ideas.append(idea3)

        # 选择最佳想法
        best_idea = max(ideas, key=lambda x: x.value_score)

        # 添加到推理轨迹
        best_idea.reasoning_trace.append(f"Generated {len(ideas)} divergent ideas")
        best_idea.reasoning_trace.append(f"Selected best idea with value={best_idea.value_score:.2f}")

        return best_idea

    def _identify_domain(self, query: str) -> str:
        """识别查询所属领域"""
        # 简化版本：基于关键词
        domain_keywords = {
            "computer_science": ["algorithm", "code", "program", "compute", "data"],
            "biology": ["life", "evolve", "organism", "gene", "cell"],
            "physics": ["energy", "force", "motion", "quantum", "entropy"],
            "cognitive_science": ["think", "mind", "memory", "learn", "perceive"]
        }

        query_lower = query.lower()

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain

        # 默认：随机选择
        return random.choice(list(self.concept_index.keys())) if self.concept_index else "unknown"

    def _generate_analogy(self, source: Concept, target: Concept, query: str) -> str:
        """生成类比"""
        return f"Analogy: {query} is like {source.name} in {source.domain}, " \
               f"but applied to {target.name} in {target.domain}. " \
               f"Consider: {target.domain} mechanisms for {source.attributes}"

    def _generate_combination(self, concepts: List[Concept], query: str) -> str:
        """生成概念组合"""
        names = [c.name for c in concepts]
        domains = [c.domain for c in concepts]

        combined_name = "+".join(names[:3])

        return f"Novel combination: {combined_name} " \
               f"combines insights from {', '.join(domains)} " \
               f"to address: {query}"

    def _calculate_analogy_novelty(self, source_domain: str, target_domain: str) -> float:
        """计算类比新颖度"""
        # 跨域类比更新颖
        if source_domain != target_domain:
            return 0.7 + random.random() * 0.2
        else:
            return 0.4 + random.random() * 0.2

    def _fallback_result(self, query: str, mode: ExplorationMode) -> ExplorationResult:
        """降级结果"""
        return ExplorationResult(
            result_id=f"expl_{int(time.time() * 1000)}",
            mode=mode,
            input_query=query,
            output_idea=f"Exploration suggestion for: {query}",
            novelty_score=0.3,
            feasibility_score=0.5,
            value_score=0.15,
            confidence=0.3,
            timestamp=time.time(),
            reasoning_trace=["Used fallback exploration"]
        )

    def get_top_explorations(self, n: int = 5) -> List[ExplorationResult]:
        """获取最佳探索结果"""
        sorted_results = sorted(self.exploration_history, key=lambda x: x.value_score, reverse=True)
        return sorted_results[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.exploration_history:
            return self.stats

        avg_novelty = np.mean([r.novelty_score for r in self.exploration_history])
        avg_feasibility = np.mean([r.feasibility_score for r in self.exploration_history])
        avg_value = np.mean([r.value_score for r in self.exploration_history])

        return {
            **self.stats,
            'avg_novelty': avg_novelty,
            'avg_feasibility': avg_feasibility,
            'avg_value': avg_value,
            'total_explorations': len(self.exploration_history),
            'novelty_ratio': self.stats['novel_ideas_generated'] / max(self.stats['total_explorations'], 1)
        }


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("创造性探索引擎测试")
    print("=" * 60)

    # 创建探索引擎
    engine = CreativeExplorationEngine(temperature=0.7)

    # 测试1: 类比推理
    print("\n[测试1] 类比推理探索")
    print("-" * 60)

    result1 = engine.explore("如何优化学习算法", mode=ExplorationMode.ANALOGICAL)
    print(f"  Idea: {result1.output_idea}")
    print(f"  Novelty: {result1.novelty_score:.2f}, Value: {result1.value_score:.2f}")

    # 测试2: 概念组合
    print("\n[测试2] 概念组合探索")
    print("-" * 60)

    result2 = engine.explore("设计新型智能系统", mode=ExplorationMode.COMBINATORIAL)
    print(f"  Idea: {result2.output_idea}")
    print(f"  Novelty: {result2.novelty_score:.2f}, Value: {result2.value_score:.2f}")

    # 测试3: 随机探索
    print("\n[测试3] 随机探索")
    print("-" * 60)

    result3 = engine.explore("突破性能瓶颈", mode=ExplorationMode.STOCHASTIC)
    print(f"  Idea: {result3.output_idea}")
    print(f"  Novelty: {result3.novelty_score:.2f}, Value: {result3.value_score:.2f}")

    # 测试4: 自动模式选择
    print("\n[测试4] 自动模式选择")
    print("-" * 60)

    contexts = [
        {"entropy": 0.2, "curiosity": 0.3},  # 低惊奇 -> 类比
        {"entropy": 0.5, "curiosity": 0.5},  # 中惊奇 -> 组合
        {"entropy": 0.9, "curiosity": 0.8}   # 高惊奇 -> 随机
    ]

    for i, ctx in enumerate(contexts, 1):
        result = engine.explore(f"测试查询{i}", context=ctx)
        print(f"  Context (entropy={ctx['entropy']:.1f}) -> {result.mode.value}")

    # 测试5: 统计
    print("\n[测试5] 统计摘要")
    print("-" * 60)

    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
