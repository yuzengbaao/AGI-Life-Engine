#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨域知识迁移系统 (Cross-Domain Knowledge Transfer System)
==========================================================

功能: 实现跨域知识映射、元学习迁移和少样本学习能力
版本: 1.0.0 (2026-01-19)

核心组件:
1. CrossDomainMapper - 跨域知识映射器
2. MetaLearningTransfer - 元学习迁移引擎
3. FewShotLearner - 少样本学习器
4. SkillExtractor - 技能提取器

目标: 提升学习智能 67.5% → 80% (+12.5%)
"""

import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class DomainKnowledge:
    """领域知识表示"""
    domain: str
    concepts: Set[str]  # 领域概念集合
    relations: Dict[Tuple[str, str], str]  # 概念关系
    patterns: List[Dict[str, Any]]  # 抽象模式
    skills: List[Dict[str, Any]]  # 技能模式
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'domain': self.domain,
            'concepts': list(self.concepts),
            'relations': {f"{k[0]}->{k[1]}": v for k, v in self.relations.items()},
            'patterns': self.patterns,
            'skills': self.skills,
            'metadata': self.metadata
        }


@dataclass
class TransferResult:
    """迁移结果"""
    source_domain: str
    target_domain: str
    success: bool
    transferred_knowledge: Optional[DomainKnowledge]
    transfer_score: float  # 迁移置信度
    adaptation_effort: float  # 适配成本 (0-1)
    improvements: Dict[str, float]  # 性能提升
    errors: List[str] = field(default_factory=list)


@dataclass
class MetaKnowledge:
    """元知识（跨任务可迁移的知识）"""
    abstract_patterns: List[Dict[str, Any]]  # 抽象模式
    learning_strategies: List[str]  # 学习策略
    problem_solving_templates: List[Dict[str, Any]]  # 问题解决模板
    transferability_score: float  # 可迁移性评分


@dataclass
class FewShotExample:
    """少样本学习示例"""
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    domain: str
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 1. 跨域知识映射器 ====================

class CrossDomainMapper:
    """
    跨域知识映射器

    功能:
    1. 提取领域知识的抽象结构
    2. 映射到目标领域
    3. 评估映射质量
    """

    def __init__(self, similarity_threshold: float = 0.6):
        """
        初始化跨域映射器

        Args:
            similarity_threshold: 概念相似度阈值
        """
        self.similarity_threshold = similarity_threshold
        self.mapping_cache = {}  # 映射缓存
        self.domain_embeddings = {}  # 领域嵌入

    def extract_abstract_structure(self, knowledge: DomainKnowledge) -> Dict[str, Any]:
        """
        提取抽象结构（领域无关的模式）

        策略:
        1. 识别高频模式
        2. 提取关系骨架
        3. 抽象技能模板

        Args:
            knowledge: 源领域知识

        Returns:
            抽象结构表示
        """
        abstract_structure = {
            'patterns': [],
            'relations': [],
            'skills': [],
            'statistics': {}
        }

        # 1. 提取高频模式
        pattern_counts = defaultdict(int)
        for pattern in knowledge.patterns:
            # 使用模式的结构签名作为键
            signature = self._get_pattern_signature(pattern)
            pattern_counts[signature] += 1

        # 选择高频模式
        frequent_patterns = [
            sig for sig, count in pattern_counts.items()
            if count >= 2  # 至少出现2次
        ]

        abstract_structure['patterns'] = frequent_patterns
        abstract_structure['statistics']['pattern_count'] = len(frequent_patterns)

        # 2. 提取关系骨架（关键关系类型）
        relation_counts = Counter()
        for (_, _), rel_type in knowledge.relations.items():
            relation_counts[rel_type] += 1

        # 选择关键关系（高频关系）
        key_relations = [
            rel_type for rel_type, count in relation_counts.most_common(5)
        ]

        abstract_structure['relations'] = key_relations
        abstract_structure['statistics']['relation_count'] = len(key_relations)

        # 3. 抽象技能模板
        for skill in knowledge.skills:
            abstract_skill = self._abstract_skill(skill)
            abstract_structure['skills'].append(abstract_skill)

        abstract_structure['statistics']['skill_count'] = len(abstract_structure['skills'])

        logger.info(f"[CrossDomainMapper] 提取抽象结构: "
                   f"{len(frequent_patterns)} 模式, {len(key_relations)} 关系, "
                   f"{len(abstract_structure['skills'])} 技能")

        return abstract_structure

    def map_to_target_domain(self,
                            abstract_structure: Dict[str, Any],
                            target_knowledge: DomainKnowledge) -> DomainKnowledge:
        """
        将抽象结构映射到目标领域

        策略:
        1. 对齐概念
        2. 匹配关系
        3. 适配技能

        Args:
            abstract_structure: 抽象结构
            target_knowledge: 目标领域知识

        Returns:
            映射后的领域知识
        """
        mapped_knowledge = DomainKnowledge(
            domain=target_knowledge.domain,
            concepts=set(),
            relations={},
            patterns=[],
            skills=[]
        )

        # 1. 映射概念（基于语义相似度）
        concept_mapping = self._align_concepts(
            abstract_structure.get('patterns', []),
            target_knowledge
        )
        mapped_knowledge.concepts = set(concept_mapping.values())

        # 2. 映射关系
        for relation in abstract_structure.get('relations', []):
            # 在目标域中寻找类似关系
            mapped_relations = self._find_similar_relations(
                relation, target_knowledge
            )
            mapped_knowledge.relations.update(mapped_relations)

        # 3. 适配技能
        for skill_template in abstract_structure.get('skills', []):
            adapted_skill = self._adapt_skill_to_domain(
                skill_template, target_knowledge
            )
            if adapted_skill:
                mapped_knowledge.skills.append(adapted_skill)

        # 4. 映射模式
        for pattern in abstract_structure.get('patterns', []):
            mapped_pattern = self._adapt_pattern_to_domain(
                pattern, target_knowledge
            )
            if mapped_pattern:
                mapped_knowledge.patterns.append(mapped_pattern)

        logger.info(f"[CrossDomainMapper] 映射到目标域: "
                   f"{len(mapped_knowledge.concepts)} 概念, "
                   f"{len(mapped_knowledge.relations)} 关系, "
                   f"{len(mapped_knowledge.skills)} 技能")

        return mapped_knowledge

    def evaluate_mapping_quality(self,
                                mapped_knowledge: DomainKnowledge,
                                target_knowledge: DomainKnowledge) -> float:
        """
        评估映射质量

        指标:
        1. 概念覆盖率
        2. 关系一致性
        3. 技能适配度

        Args:
            mapped_knowledge: 映射的知识
            target_knowledge: 目标领域知识

        Returns:
            映射质量评分 (0-1)
        """
        # 1. 概念覆盖率
        if len(target_knowledge.concepts) > 0:
            concept_coverage = len(mapped_knowledge.concepts & target_knowledge.concepts) / len(target_knowledge.concepts)
        else:
            concept_coverage = 0.0

        # 2. 关系一致性
        if len(target_knowledge.relations) > 0:
            relation_overlap = set(mapped_knowledge.relations.values()) & set(target_knowledge.relations.values())
            relation_consistency = len(relation_overlap) / len(target_knowledge.relations)
        else:
            relation_consistency = 0.0

        # 3. 技能适配度（技能与领域的匹配度）
        skill_adaptation = 0.0
        if mapped_knowledge.skills:
            # 评估技能是否适合目标域
            adapted_count = sum(
                1 for skill in mapped_knowledge.skills
                if self._is_skill_compatible(skill, target_knowledge)
            )
            skill_adaptation = adapted_count / len(mapped_knowledge.skills)

        # 加权组合
        quality_score = (
            0.4 * concept_coverage +
            0.3 * relation_consistency +
            0.3 * skill_adaptation
        )

        logger.info(f"[CrossDomainMapper] 映射质量: {quality_score:.3f} "
                   f"(概念={concept_coverage:.2f}, 关系={relation_consistency:.2f}, 技能={skill_adaptation:.2f})")

        return quality_score

    # ==================== 辅助方法 ====================

    def _get_pattern_signature(self, pattern: Dict[str, Any]) -> str:
        """获取模式的结构签名"""
        # 简化：使用键的集合作为签名
        return json.dumps(sorted(pattern.keys()), sort_keys=True)

    def _abstract_skill(self, skill: Dict[str, Any]) -> Dict[str, Any]:
        """抽象技能（去除领域特定细节）"""
        return {
            'type': skill.get('type', 'unknown'),
            'operations': skill.get('operations', []),
            'parameters': skill.get('parameters', {}),
            'abstract_signature': self._get_pattern_signature(skill)
        }

    def _align_concepts(self, patterns: List[str], target_knowledge: DomainKnowledge) -> Dict[str, str]:
        """对齐概念（简化版：使用字符串相似度）"""
        mapping = {}
        for pattern in patterns:
            # 在目标域中寻找最相似的概念
            best_match = None
            best_score = 0.0

            for concept in target_knowledge.concepts:
                # 简单的字符串相似度
                similarity = self._string_similarity(pattern, concept)
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = concept

            if best_match:
                mapping[pattern] = best_match

        return mapping

    def _find_similar_relations(self, relation_type: str, target_knowledge: DomainKnowledge) -> Dict[Tuple[str, str], str]:
        """在目标域中寻找相似关系"""
        similar_relations = {}

        for (source, target), rel in target_knowledge.relations.items():
            if self._string_similarity(relation_type, rel) >= self.similarity_threshold:
                similar_relations[(source, target)] = rel

        return similar_relations

    def _adapt_skill_to_domain(self, skill_template: Dict[str, Any], target_knowledge: DomainKnowledge) -> Optional[Dict[str, Any]]:
        """适配技能到目标域"""
        # 简化：检查技能是否与目标域兼容
        if self._is_skill_compatible(skill_template, target_knowledge):
            return {
                **skill_template,
                'domain': target_knowledge.domain,
                'adapted': True
            }
        return None

    def _adapt_pattern_to_domain(self, pattern: str, target_knowledge: DomainKnowledge) -> Optional[str]:
        """适配模式到目标域"""
        # 简化：直接使用模式（实际应用中需要更复杂的适配）
        if any(self._string_similarity(pattern, str(concept)) >= self.similarity_threshold
               for concept in target_knowledge.concepts):
            return pattern
        return None

    def _is_skill_compatible(self, skill: Dict[str, Any], domain: DomainKnowledge) -> bool:
        """检查技能是否与领域兼容"""
        # 简化：检查技能所需的概念是否存在于领域
        required_concepts = skill.get('required_concepts', [])
        return all(concept in domain.concepts for concept in required_concepts)

    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度（简化版Jaccard）"""
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


# ==================== 2. 元学习迁移引擎 ====================

class MetaLearningTransfer:
    """
    元学习迁移引擎

    功能:
    1. 从源任务提取元知识
    2. 适配元知识到目标任务
    3. 评估迁移效果
    """

    def __init__(self):
        """初始化元学习引擎"""
        self.meta_knowledge_cache = {}  # 元知识缓存
        self.transfer_history = []  # 迁移历史

    def extract_meta_knowledge(self,
                              source_tasks: List[Dict[str, Any]],
                              domain: str) -> MetaKnowledge:
        """
        从源任务提取元知识

        策略:
        1. 识别跨任务的共同模式
        2. 提取学习策略
        3. 抽象问题解决模板

        Args:
            source_tasks: 源任务列表
            domain: 领域名称

        Returns:
            元知识
        """
        abstract_patterns = []
        learning_strategies = []
        problem_solving_templates = []

        # 1. 提取跨任务共同模式
        pattern_counts = defaultdict(int)
        for task in source_tasks:
            patterns = task.get('patterns', [])
            for pattern in patterns:
                signature = json.dumps(pattern, sort_keys=True)
                pattern_counts[signature] += 1

        # 选择跨任务模式（至少在2个任务中出现）
        for signature, count in pattern_counts.items():
            if count >= 2:
                pattern = json.loads(signature)
                abstract_patterns.append(pattern)

        # 2. 提取学习策略
        strategy_counts = defaultdict(int)
        for task in source_tasks:
            strategies = task.get('learning_strategies', ['default'])
            for strategy in strategies:
                strategy_counts[strategy] += 1

        # 选择高频策略
        for strategy, count in strategy_counts.items():
            if count >= len(source_tasks) * 0.5:  # 至少在50%的任务中出现
                learning_strategies.append(strategy)

        # 3. 抽象问题解决模板
        for task in source_tasks:
            template = {
                'task_type': task.get('type', 'unknown'),
                'steps': task.get('solution_steps', []),
                'success_rate': task.get('success_rate', 0.5)
            }
            problem_solving_templates.append(template)

        # 计算可迁移性评分
        transferability_score = self._compute_transferability(
            abstract_patterns, learning_strategies, problem_solving_templates
        )

        meta_knowledge = MetaKnowledge(
            abstract_patterns=abstract_patterns,
            learning_strategies=learning_strategies,
            problem_solving_templates=problem_solving_templates,
            transferability_score=transferability_score
        )

        self.meta_knowledge_cache[domain] = meta_knowledge

        logger.info(f"[MetaLearningTransfer] 提取元知识: "
                   f"{len(abstract_patterns)} 模式, {len(learning_strategies)} 策略, "
                   f"{len(problem_solving_templates)} 模板, "
                   f"可迁移性={transferability_score:.2f}")

        return meta_knowledge

    def adapt_to_target(self,
                       meta_knowledge: MetaKnowledge,
                       target_task: Dict[str, Any],
                       target_domain: str) -> TransferResult:
        """
        适配元知识到目标任务

        策略:
        1. 选择最相关的模式和模板
        2. 根据目标任务调整
        3. 评估适配效果

        Args:
            meta_knowledge: 元知识
            target_task: 目标任务
            target_domain: 目标领域

        Returns:
            迁移结果
        """
        try:
            # 1. 选择相关模式
            relevant_patterns = self._select_relevant_patterns(
                meta_knowledge.abstract_patterns,
                target_task
            )

            # 2. 选择相关模板
            relevant_templates = self._select_relevant_templates(
                meta_knowledge.problem_solving_templates,
                target_task
            )

            # 3. 适配学习策略
            adapted_strategies = self._adapt_strategies(
                meta_knowledge.learning_strategies,
                target_task
            )

            # 构建迁移的知识
            transferred_knowledge = DomainKnowledge(
                domain=target_domain,
                concepts=set(),  # 从目标任务提取
                relations={},
                patterns=relevant_patterns,
                skills=[{'strategies': adapted_strategies}]
            )

            # 计算迁移评分
            transfer_score = self._compute_transfer_score(
                meta_knowledge, target_task
            )

            # 计算适配成本
            adaptation_effort = self._compute_adaptation_effort(
                meta_knowledge, target_task
            )

            # 估算性能提升
            improvements = self._estimate_improvements(
                transfer_score, adaptation_effort
            )

            result = TransferResult(
                source_domain=meta_knowledge.__class__.__name__,
                target_domain=target_domain,
                success=True,
                transferred_knowledge=transferred_knowledge,
                transfer_score=transfer_score,
                adaptation_effort=adaptation_effort,
                improvements=improvements
            )

            self.transfer_history.append(result)

            logger.info(f"[MetaLearningTransfer] 迁移成功: "
                       f"评分={transfer_score:.2f}, "
                       f"适配成本={adaptation_effort:.2f}, "
                       f"预期提升={improvements}")

            return result

        except Exception as e:
            logger.error(f"[MetaLearningTransfer] 迁移失败: {e}")
            return TransferResult(
                source_domain=meta_knowledge.__class__.__name__,
                target_domain=target_domain,
                success=False,
                transferred_knowledge=None,
                transfer_score=0.0,
                adaptation_effort=1.0,
                improvements={},
                errors=[str(e)]
            )

    def _compute_transferability(self,
                                 patterns: List,
                                 strategies: List,
                                 templates: List) -> float:
        """计算可迁移性评分"""
        # 基于模式、策略、模板的数量和质量
        pattern_score = min(len(patterns) / 10, 1.0)  # 最多10个模式
        strategy_score = min(len(strategies) / 5, 1.0)  # 最多5个策略
        template_score = min(len(templates) / 10, 1.0)  # 最多10个模板

        return (pattern_score + strategy_score + template_score) / 3

    def _select_relevant_patterns(self,
                                  patterns: List[Dict[str, Any]],
                                  target_task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择与目标任务相关的模式"""
        # 简化：返回所有模式（实际应用中需要相似度匹配）
        return patterns[:5]  # 限制数量

    def _select_relevant_templates(self,
                                   templates: List[Dict[str, Any]],
                                   target_task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择与目标任务相关的模板"""
        target_type = target_task.get('type', 'unknown')
        # 选择同类型的高成功率模板
        relevant = [
            t for t in templates
            if t.get('task_type') == target_type and t.get('success_rate', 0) > 0.7
        ]
        return relevant[:3]  # 限制数量

    def _adapt_strategies(self,
                         strategies: List[str],
                         target_task: Dict[str, Any]) -> List[str]:
        """适配学习策略"""
        # 简化：直接返回策略
        return strategies

    def _compute_transfer_score(self,
                                meta_knowledge: MetaKnowledge,
                                target_task: Dict[str, Any]) -> float:
        """计算迁移评分"""
        # 基于元知识的可迁移性
        base_score = meta_knowledge.transferability_score

        # 根据目标任务调整
        task_complexity = target_task.get('complexity', 0.5)
        adjusted_score = base_score * (1 + task_complexity)

        return min(adjusted_score, 1.0)

    def _compute_adaptation_effort(self,
                                   meta_knowledge: MetaKnowledge,
                                   target_task: Dict[str, Any]) -> float:
        """计算适配成本（0-1，越低越好）"""
        # 简化：基于元知识数量
        knowledge_size = (
            len(meta_knowledge.abstract_patterns) +
            len(meta_knowledge.learning_strategies) +
            len(meta_knowledge.problem_solving_templates)
        )

        # 知识越多，适配成本越高
        effort = min(knowledge_size / 50, 1.0)
        return effort

    def _estimate_improvements(self,
                              transfer_score: float,
                              adaptation_effort: float) -> Dict[str, float]:
        """估算性能提升"""
        # 净收益 = 迁移评分 - 适配成本
        net_benefit = transfer_score - adaptation_effort * 0.3

        return {
            'learning_speed': net_benefit * 0.5,  # 学习速度提升
            'accuracy': net_benefit * 0.3,  # 准确率提升
            'efficiency': net_benefit * 0.2  # 效率提升
        }


# ==================== 3. 少样本学习器 ====================

class FewShotLearner:
    """
    少样本学习器

    功能:
    1. 从少量样本快速学习
    2. 元初始化
    3. 快速适应
    """

    def __init__(self, num_shots: int = 5):
        """
        初始化少样本学习器

        Args:
            num_shots: 使用的样本数量（默认5个）
        """
        self.num_shots = num_shots
        self.learned_models = {}  # 已学习的模型
        self.meta_initialization = None  # 元初始化参数

    def meta_initialize(self, meta_knowledge: Optional[MetaKnowledge] = None):
        """
        元初始化（学习如何学习）

        Args:
            meta_knowledge: 可选的元知识
        """
        # 简化：使用默认初始化
        self.meta_initialization = {
            'learning_rate': 0.01,
            'adaptation_steps': 10,
            'initialization_strategy': 'meta_learned'
        }

        if meta_knowledge:
            # 使用元知识调整初始化
            self.meta_initialization['transferability'] = meta_knowledge.transferability_score
            self.meta_initialization['patterns'] = len(meta_knowledge.abstract_patterns)

        logger.info(f"[FewShotLearner] 元初始化完成: {self.meta_initialization}")

    def learn_from_few_shots(self,
                            examples: List[FewShotExample],
                            task_type: str) -> Dict[str, Any]:
        """
        从少量样本学习

        Args:
            examples: 训练示例（少量）
            task_type: 任务类型

        Returns:
            学习到的模型
        """
        if len(examples) > self.num_shots:
            logger.warning(f"[FewShotLearner] 样本数({len(examples)})超过设定值({self.num_shots})，"
                          f"将只使用前{self.num_shots}个")
            examples = examples[:self.num_shots]

        logger.info(f"[FewShotLearner] 从{len(examples)}个样本学习 (任务类型: {task_type})")

        # 1. 元初始化
        if not self.meta_initialization:
            self.meta_initialize()

        # 2. 从示例中提取模式
        patterns = self._extract_patterns_from_examples(examples)

        # 3. 构建快速适应模型
        model = self._build_adaptive_model(patterns, task_type)

        # 4. 保存模型
        self.learned_models[task_type] = model

        logger.info(f"[FewShotLearner] 学习完成: {len(patterns)} 个模式")

        return model

    def adapt_to_new_task(self,
                         examples: List[FewShotExample],
                         task_type: str,
                         base_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        适应新任务（使用已学习的模型作为起点）

        Args:
            examples: 新任务的示例
            task_type: 任务类型
            base_model: 基础模型（可选）

        Returns:
            适应后的模型
        """
        # 使用已有模型或创建新模型
        if base_model is None and task_type in self.learned_models:
            base_model = self.learned_models[task_type]

        if base_model:
            logger.info(f"[FewShotLearner] 基于已有模型适应新任务")
            # 基于已有模型快速适应
            adapted_model = self._rapid_adaptation(base_model, examples)
        else:
            logger.info(f"[FewShotLearner] 从零学习新任务")
            # 从零开始学习
            adapted_model = self.learn_from_few_shots(examples, task_type)

        return adapted_model

    def predict(self,
               model: Dict[str, Any],
               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用学习到的模型进行预测

        Args:
            model: 学习到的模型
            input_data: 输入数据

        Returns:
            预测结果
        """
        # 简化：基于模式匹配
        patterns = model.get('patterns', [])

        # 寻找最匹配的模式
        best_match = None
        best_score = 0.0

        for pattern in patterns:
            score = self._match_pattern(pattern, input_data)
            if score > best_score:
                best_score = score
                best_match = pattern

        if best_match:
            return {
                'prediction': best_match.get('output', {}),
                'confidence': best_score,
                'matched_pattern': best_match.get('signature', 'unknown')
            }
        else:
            return {
                'prediction': {},
                'confidence': 0.0,
                'error': 'No matching pattern found'
            }

    def _extract_patterns_from_examples(self, examples: List[FewShotExample]) -> List[Dict[str, Any]]:
        """从示例中提取模式"""
        patterns = []

        for example in examples:
            pattern = {
                'input_signature': self._get_signature(example.input_data),
                'output': example.output_data,
                'domain': example.domain,
                'task_type': example.task_type,
                'signature': f"{example.domain}_{example.task_type}_{len(patterns)}"
            }
            patterns.append(pattern)

        return patterns

    def _build_adaptive_model(self, patterns: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """构建快速适应模型"""
        return {
            'task_type': task_type,
            'patterns': patterns,
            'learning_strategy': self.meta_initialization.get('initialization_strategy', 'default'),
            'adaptation_rate': self.meta_initialization.get('learning_rate', 0.01)
        }

    def _rapid_adaptation(self, base_model: Dict[str, Any], examples: List[FewShotExample]) -> Dict[str, Any]:
        """快速适应（基于已有模型）"""
        # 合并基础模型的模式和新示例的模式
        existing_patterns = base_model.get('patterns', [])
        new_patterns = self._extract_patterns_from_examples(examples)

        # 合并并去重
        all_patterns = existing_patterns + new_patterns
        unique_patterns = []
        seen_signatures = set()

        for pattern in all_patterns:
            sig = pattern.get('signature')
            if sig not in seen_signatures:
                unique_patterns.append(pattern)
                seen_signatures.add(sig)

        # 创建适应后的模型
        adapted_model = {
            **base_model,
            'patterns': unique_patterns,
            'adapted': True,
            'adaptation_count': len(new_patterns)
        }

        return adapted_model

    def _match_pattern(self, pattern: Dict[str, Any], input_data: Dict[str, Any]) -> float:
        """匹配模式（简化版）"""
        # 简化：使用键的匹配度
        pattern_keys = set(pattern.get('input_signature', {}).keys())
        input_keys = set(input_data.keys())

        if not pattern_keys:
            return 0.0

        intersection = len(pattern_keys & input_keys)
        return intersection / len(pattern_keys)

    def _get_signature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取数据签名（简化的结构表示）"""
        return {key: type(value).__name__ for key, value in data.items()}


# ==================== 4. 技能提取器 ====================

class SkillExtractor:
    """
    技能提取器

    功能:
    1. 从经验中提取可复用的技能
    2. 抽象技能模式
    3. 技能分类与索引
    """

    def __init__(self):
        """初始化技能提取器"""
        self.skill_library = {}  # 技能库
        self.skill_categories = defaultdict(list)  # 技能分类

    def extract_skills(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从经验中提取技能

        Args:
            experiences: 经验列表

        Returns:
            提取的技能列表
        """
        extracted_skills = []

        for exp in experiences:
            # 1. 识别技能类型
            skill_type = self._identify_skill_type(exp)

            # 2. 提取技能参数
            skill_params = self._extract_skill_parameters(exp)

            # 3. 评估技能质量
            skill_quality = self._evaluate_skill_quality(exp)

            # 4. 构建技能
            skill = {
                'type': skill_type,
                'parameters': skill_params,
                'quality': skill_quality,
                'source': exp.get('source', 'unknown'),
                'success_rate': exp.get('success_rate', 0.5),
                'usage_count': 0
            }

            # 5. 分类技能
            self._categorize_skill(skill)

            extracted_skills.append(skill)

        logger.info(f"[SkillExtractor] 提取{len(extracted_skills)}个技能")

        return extracted_skills

    def _identify_skill_type(self, experience: Dict[str, Any]) -> str:
        """识别技能类型"""
        # 简化：基于经验的操作类型
        operations = experience.get('operations', [])
        if not operations:
            return 'generic'

        # 使用第一个操作作为类型
        return operations[0] if isinstance(operations[0], str) else 'complex'

    def _extract_skill_parameters(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """提取技能参数"""
        # 简化：提取关键参数
        return {
            'complexity': experience.get('complexity', 0.5),
            'duration': experience.get('duration', 0),
            'resources': experience.get('resources', {}),
            'preconditions': experience.get('preconditions', [])
        }

    def _evaluate_skill_quality(self, experience: Dict[str, Any]) -> float:
        """评估技能质量"""
        # 基于成功率和效率
        success_rate = experience.get('success_rate', 0.5)
        efficiency = experience.get('efficiency', 0.5)

        return (success_rate + efficiency) / 2

    def _categorize_skill(self, skill: Dict[str, Any]):
        """分类技能"""
        skill_type = skill['type']
        self.skill_categories[skill_type].append(skill)

        # 添加到技能库
        skill_id = f"{skill_type}_{len(self.skill_library)}"
        self.skill_library[skill_id] = skill

    def find_similar_skills(self, target_skill: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        """查找相似技能"""
        similarities = []

        for skill_id, skill in self.skill_library.items():
            similarity = self._compute_skill_similarity(target_skill, skill)
            similarities.append((skill_id, skill, similarity))

        # 排序并返回top_k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return [(sid, skill) for sid, skill, _ in similarities[:top_k]]

    def _compute_skill_similarity(self, skill1: Dict[str, Any], skill2: Dict[str, Any]) -> float:
        """计算技能相似度"""
        # 简化：基于类型和参数
        if skill1['type'] != skill2['type']:
            return 0.0

        # 参数相似度
        params1 = skill1.get('parameters', {})
        params2 = skill2.get('parameters', {})

        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0

        similarity_sum = 0.0
        for key in common_keys:
            val1 = params1[key]
            val2 = params2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值相似度
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2), 1.0)
                similarity_sum += 1 - (diff / max_val)
            elif val1 == val2:
                similarity_sum += 1.0

        return similarity_sum / len(common_keys)


# ==================== 5. 跨域迁移系统集成 ====================

class CrossDomainTransferSystem:
    """
    跨域迁移系统集成

    整合所有组件，提供统一的跨域迁移接口
    
    🆕 [2026-01-24] 拓扑连接增强:
    - 新增 RecursiveSelfMemory 连接：记录跨域学习经验，增强元学习能力
    """

    def __init__(self, recursive_self_memory=None):
        """
        初始化跨域迁移系统
        
        Args:
            recursive_self_memory: 🆕 递归自引用记忆（用于记录跨域学习经验）
        """
        self.mapper = CrossDomainMapper()
        self.meta_learning = MetaLearningTransfer()
        self.few_shot_learner = FewShotLearner()
        self.skill_extractor = SkillExtractor()

        self.transfer_history = []  # 迁移历史
        self.performance_metrics = defaultdict(list)  # 性能指标
        
        # 🆕 [2026-01-24] 拓扑连接: RecursiveSelfMemory
        self.recursive_self_memory = recursive_self_memory

    def transfer_knowledge(self,
                          source_domain: str,
                          target_domain: str,
                          source_knowledge: DomainKnowledge,
                          target_knowledge: DomainKnowledge) -> TransferResult:
        """
        执行完整的跨域知识迁移流程

        流程:
        1. 提取抽象结构
        2. 映射到目标域
        3. 评估映射质量
        4. 返回迁移结果

        Args:
            source_domain: 源领域名称
            target_domain: 目标领域名称
            source_knowledge: 源领域知识
            target_knowledge: 目标领域知识

        Returns:
            迁移结果
        """
        logger.info(f"[CrossDomainTransfer] 开始迁移: {source_domain} → {target_domain}")

        try:
            # 1. 提取抽象结构
            abstract_structure = self.mapper.extract_abstract_structure(source_knowledge)

            # 2. 映射到目标域
            mapped_knowledge = self.mapper.map_to_target_domain(
                abstract_structure, target_knowledge
            )

            # 3. 评估映射质量
            quality_score = self.mapper.evaluate_mapping_quality(
                mapped_knowledge, target_knowledge
            )

            # 4. 构建迁移结果
            result = TransferResult(
                source_domain=source_domain,
                target_domain=target_domain,
                success=quality_score >= 0.5,  # 质量阈值0.5
                transferred_knowledge=mapped_knowledge,
                transfer_score=quality_score,
                adaptation_effort=1.0 - quality_score,  # 质量越高，成本越低
                improvements={
                    'knowledge_transfer': quality_score,
                    'expected_performance_gain': quality_score * 0.3
                }
            )

            self.transfer_history.append(result)
            
            # 🆕 [2026-01-24] 拓扑连接: 记录成功的迁移经验到递归自引用记忆
            if self.recursive_self_memory and result.success:
                try:
                    self.recursive_self_memory.store_experience(
                        experience={
                            'type': 'cross_domain_transfer',
                            'source_domain': source_domain,
                            'target_domain': target_domain,
                            'transfer_score': quality_score,
                            'patterns_transferred': len(abstract_structure.get('patterns', [])),
                            'skills_transferred': len(abstract_structure.get('skills', []))
                        },
                        why_remembered=f"成功的跨域迁移: {source_domain}→{target_domain}",
                        importance='high' if quality_score >= 0.7 else 'medium'
                    )
                    logger.debug(f"[CrossDomainTransfer] 迁移经验已记录到RecursiveSelfMemory")
                except Exception as mem_err:
                    logger.debug(f"[CrossDomainTransfer] 记忆记录失败: {mem_err}")

            logger.info(f"[CrossDomainTransfer] 迁移完成: 成功={result.success}, "
                       f"评分={quality_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"[CrossDomainTransfer] 迁移失败: {e}")
            return TransferResult(
                source_domain=source_domain,
                target_domain=target_domain,
                success=False,
                transferred_knowledge=None,
                transfer_score=0.0,
                adaptation_effort=1.0,
                improvements={},
                errors=[str(e)]
            )

    def meta_learning_transfer(self,
                               source_tasks: List[Dict[str, Any]],
                               target_task: Dict[str, Any],
                               target_domain: str) -> TransferResult:
        """
        使用元学习进行迁移

        Args:
            source_tasks: 源任务列表
            target_task: 目标任务
            target_domain: 目标领域

        Returns:
            迁移结果
        """
        logger.info(f"[CrossDomainTransfer] 元学习迁移: {len(source_tasks)}个任务 → {target_domain}")

        # 1. 提取元知识
        meta_knowledge = self.meta_learning.extract_meta_knowledge(
            source_tasks, target_domain
        )

        # 2. 适配到目标任务
        result = self.meta_learning.adapt_to_target(
            meta_knowledge, target_task, target_domain
        )

        return result

    def few_shot_learning(self,
                         examples: List[FewShotExample],
                         task_type: str,
                         adapt: bool = False) -> Dict[str, Any]:
        """
        执行少样本学习

        Args:
            examples: 训练示例
            task_type: 任务类型
            adapt: 是否基于已有模型适应

        Returns:
            学习到的模型
        """
        logger.info(f"[CrossDomainTransfer] 少样本学习: {len(examples)}个示例, 任务={task_type}")

        if adapt:
            # 基于已有模型适应
            model = self.few_shot_learner.adapt_to_new_task(examples, task_type)
        else:
            # 从零学习
            model = self.few_shot_learner.learn_from_few_shots(examples, task_type)

        return model

    def extract_and_index_skills(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取并索引技能

        Args:
            experiences: 经验列表

        Returns:
            提取的技能列表
        """
        logger.info(f"[CrossDomainTransfer] 提取技能: {len(experiences)}个经验")

        skills = self.skill_extractor.extract_skills(experiences)
        return skills

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'total_transfers': len(self.transfer_history),
            'successful_transfers': sum(1 for t in self.transfer_history if t.success),
            'average_transfer_score': np.mean([t.transfer_score for t in self.transfer_history]) if self.transfer_history else 0.0,
            'skill_library_size': len(self.skill_extractor.skill_library),
            'skill_categories': {
                category: len(skills)
                for category, skills in self.skill_extractor.skill_categories.items()
            },
            'few_shot_models': len(self.few_shot_learner.learned_models)
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("跨域知识迁移系统测试")
    print("=" * 70)

    # 创建系统
    system = CrossDomainTransferSystem()

    # ==================== 测试1: 跨域知识映射 ====================
    print("\n测试1: 跨域知识映射")
    print("-" * 70)

    # 创建源领域知识（数学域）
    math_knowledge = DomainKnowledge(
        domain='mathematics',
        concepts={'equation', 'variable', 'solution', 'function'},
        relations={
            ('equation', 'variable'): 'contains',
            ('equation', 'solution'): 'has',
            ('function', 'variable'): 'depends_on'
        },
        patterns=[
            {'type': 'linear', 'form': 'ax + b = 0'},
            {'type': 'quadratic', 'form': 'ax^2 + bx + c = 0'},
            {'type': 'equation', 'operation': 'solve_for_x'}
        ],
        skills=[
            {'type': 'solve_equation', 'method': 'algebraic', 'operations': ['isolate', 'substitute']},
            {'type': 'graph_function', 'method': 'plotting', 'operations': ['calculate_points', 'draw']}
        ]
    )

    # 创建目标领域知识（物理域）
    physics_knowledge = DomainKnowledge(
        domain='physics',
        concepts={'force', 'mass', 'acceleration', 'equation'},
        relations={
            ('force', 'mass'): 'proportional',
            ('force', 'acceleration'): 'related',
            ('equation', 'force'): 'describes'
        },
        patterns=[
            {'type': 'newton', 'form': 'F = ma'},
            {'type': 'kinematic', 'form': 'v = v0 + at'}
        ],
        skills=[
            {'type': 'apply_formula', 'method': 'substitution', 'operations': ['identify_vars', 'calculate']}
        ]
    )

    # 执行跨域迁移
    result = system.transfer_knowledge(
        source_domain='mathematics',
        target_domain='physics',
        source_knowledge=math_knowledge,
        target_knowledge=physics_knowledge
    )

    print(f"迁移结果: {'✅ 成功' if result.success else '❌ 失败'}")
    print(f"迁移评分: {result.transfer_score:.3f}")
    print(f"适配成本: {result.adaptation_effort:.3f}")
    print(f"预期提升: {result.improvements}")
    if result.transferred_knowledge:
        print(f"迁移的概念: {len(result.transferred_knowledge.concepts)} 个")
        print(f"迁移的技能: {len(result.transferred_knowledge.skills)} 个")

    # ==================== 测试2: 元学习迁移 ====================
    print("\n测试2: 元学习迁移")
    print("-" * 70)

    # 创建源任务
    source_tasks = [
        {
            'type': 'optimization',
            'patterns': [{'method': 'gradient_descent', 'learning_rate': 0.01}],
            'learning_strategies': ['iterative', 'gradient_based'],
            'solution_steps': ['initialize', 'compute_gradient', 'update', 'repeat'],
            'success_rate': 0.85
        },
        {
            'type': 'optimization',
            'patterns': [{'method': 'adam', 'learning_rate': 0.001}],
            'learning_strategies': ['iterative', 'momentum_based'],
            'solution_steps': ['initialize', 'compute_gradient', 'update_momentum', 'update', 'repeat'],
            'success_rate': 0.90
        }
    ]

    # 创建目标任务
    target_task = {
        'type': 'optimization',
        'complexity': 0.7,
        'description': 'Hyperparameter tuning'
    }

    # 执行元学习迁移
    meta_result = system.meta_learning_transfer(
        source_tasks=source_tasks,
        target_task=target_task,
        target_domain='machine_learning'
    )

    print(f"元学习迁移: {'✅ 成功' if meta_result.success else '❌ 失败'}")
    print(f"迁移评分: {meta_result.transfer_score:.3f}")
    print(f"适配成本: {meta_result.adaptation_effort:.3f}")
    print(f"性能提升: {meta_result.improvements}")

    # ==================== 测试3: 少样本学习 ====================
    print("\n测试3: 少样本学习")
    print("-" * 70)

    # 创建训练示例
    examples = [
        FewShotExample(
            input_data={'x': 1, 'y': 2},
            output_data={'sum': 3, 'product': 2},
            domain='arithmetic',
            task_type='basic_ops'
        ),
        FewShotExample(
            input_data={'x': 5, 'y': 3},
            output_data={'sum': 8, 'product': 15},
            domain='arithmetic',
            task_type='basic_ops'
        ),
        FewShotExample(
            input_data={'x': 10, 'y': 7},
            output_data={'sum': 17, 'product': 70},
            domain='arithmetic',
            task_type='basic_ops'
        )
    ]

    # 执行少样本学习
    model = system.few_shot_learning(
        examples=examples,
        task_type='basic_ops'
    )

    print(f"学习到的模型: {model['task_type']}")
    print(f"模式数量: {len(model['patterns'])}")
    print(f"学习策略: {model['learning_strategy']}")

    # 测试预测
    test_input = {'x': 4, 'y': 6}
    prediction = system.few_shot_learner.predict(model, test_input)

    print(f"\n预测测试:")
    print(f"  输入: {test_input}")
    print(f"  预测: {prediction['prediction']}")
    print(f"  置信度: {prediction['confidence']:.3f}")

    # ==================== 测试4: 技能提取 ====================
    print("\n测试4: 技能提取")
    print("-" * 70)

    # 创建经验数据
    experiences = [
        {
            'operations': ['plan', 'execute', 'evaluate'],
            'success_rate': 0.8,
            'efficiency': 0.7,
            'complexity': 0.6,
            'source': 'task_1'
        },
        {
            'operations': ['analyze', 'design', 'implement'],
            'success_rate': 0.9,
            'efficiency': 0.85,
            'complexity': 0.8,
            'source': 'task_2'
        }
    ]

    # 提取技能
    skills = system.extract_and_index_skills(experiences)

    print(f"提取的技能: {len(skills)} 个")
    for skill in skills:
        print(f"  - 类型: {skill['type']}, 质量: {skill['quality']:.2f}, "
              f"成功率: {skill['success_rate']:.2f}")

    # ==================== 系统统计 ====================
    print("\n" + "=" * 70)
    print("系统统计")
    print("=" * 70)

    stats = system.get_statistics()
    print(f"总迁移数: {stats['total_transfers']}")
    print(f"成功迁移: {stats['successful_transfers']}")
    print(f"平均评分: {stats['average_transfer_score']:.3f}")
    print(f"技能库大小: {stats['skill_library_size']}")
    print(f"技能分类: {stats['skill_categories']}")
    print(f"少样本模型数: {stats['few_shot_models']}")

    print("\n" + "=" * 70)
    print("✅ 所有测试完成")
    print("=" * 70)
