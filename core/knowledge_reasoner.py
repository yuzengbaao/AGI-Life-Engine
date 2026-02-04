"""
跨会话记忆系统 - 知识推理器
Cross-Session Memory System - Knowledge Reasoner

版本: 1.0.0
日期: 2025-01-14
阶段: Phase 4 - 记忆检索与推理
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import networkx as nx
from core.knowledge_graph import ArchitectureKnowledgeGraph
from core.graph_query_engine import GraphQueryEngine
from core.entity_extractor import Entity

logger = logging.getLogger(__name__)


class ReasonerError(Exception):
    """推理器异常"""


class RuleType(Enum):
    """推理规则类型"""

    SYLLOGISM = "syllogism"  # 三段论
    TRANSITIVITY = "transitivity"  # 传递性
    SYMMETRY = "symmetry"  # 对称性
    INVERSE = "inverse"  # 逆关系


@dataclass
class ReasoningRule:
    """推理规则"""

    rule_type: RuleType
    pattern: Dict[str, Any]  # 匹配模式
    conclusion_template: str  # 结论模板
    confidence_func: callable  # 置信度计算函数


@dataclass
class ReasoningStep:
    """推理步骤"""

    step_num: int
    rule_type: RuleType
    premises: List[Tuple[str, str, str]]  # (subject, relation, object)
    conclusion: Tuple[str, str, str]
    confidence: float
    explanation: str


@dataclass
class ReasoningChain:
    """推理链"""

    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_conclusion: Optional[Tuple[str, str, str]] = None
    overall_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def add_step(self, step: ReasoningStep) -> None:
        """添加推理步骤"""
        self.steps.append(step)
        # 更新总体置信度(取最小值)
        if self.steps:
            self.overall_confidence = min(s.confidence for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "steps": [
                {
                    "step_num": s.step_num,
                    "rule_type": s.rule_type.value,
                    "premises": s.premises,
                    "conclusion": s.conclusion,
                    "confidence": s.confidence,
                    "explanation": s.explanation,
                }
                for s in self.steps
            ],
            "final_conclusion": self.final_conclusion,
            "overall_confidence": self.overall_confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class KnowledgeReasoner:
    """
    知识推理器

    支持多种推理模式:
    1. 规则推理: 基于预定义规则的推理
    2. 路径推理: 基于图路径的多跳推理
    3. 置信度传播: 基于贝叶斯网络的置信度计算
    """

    __slots__ = (
        "kg",
        "query_engine",
        "rules",
        "stats",
    )

    def __init__(self, knowledge_graph: ArchitectureKnowledgeGraph):
        """
        初始化知识推理器

        Args:
            knowledge_graph: 知识图谱
            
        Raises:
            ReasonerError: 参数错误
        """
        if not isinstance(knowledge_graph, ArchitectureKnowledgeGraph):
            logger.warning("Knowledge graph is not an instance of ArchitectureKnowledgeGraph")
            raise ReasonerError("Invalid knowledge graph")

        self.kg = knowledge_graph
        self.query_engine = GraphQueryEngine(knowledge_graph)
        self.rules: List[ReasoningRule] = []
        self.stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "avg_confidence": 0.0,
            "rule_usage": {},
        }

        # 初始化预定义规则
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """初始化预定义推理规则"""
        # 传递性规则: A->B, B->C => A->C
        self.rules.append(
            ReasoningRule(
                rule_type=RuleType.TRANSITIVITY,
                pattern={"relations": ["related_to", "part_of", "subclass_of"]},
                conclusion_template="{A} {relation} {C}",
                confidence_func=lambda c1, c2: c1 * c2 * 0.9,  # 传递性会降低置信度
            )
        )

        # 对称性规则: A->B => B->A
        self.rules.append(
            ReasoningRule(
                rule_type=RuleType.SYMMETRY,
                pattern={"relations": ["similar_to", "related_to", "connected_to"]},
                conclusion_template="{B} {relation} {A}",
                confidence_func=lambda c: c * 0.95,  # 对称性略微降低置信度
            )
        )

        # 逆关系规则: A-parent->B => B-child->A
        self.rules.append(
            ReasoningRule(
                rule_type=RuleType.INVERSE,
                pattern={
                    "inverse_pairs": [
                        ("parent_of", "child_of"),
                        ("part_of", "has_part"),
                        ("subclass_of", "superclass_of"),
                    ]
                },
                conclusion_template="{B} {inverse_relation} {A}",
                confidence_func=lambda c: c,  # 逆关系不降低置信度
            )
        )

    def reason(
        self, query: str, max_depth: int = 3, min_confidence: float = 0.5
    ) -> ReasoningChain:
        """
        执行推理

        Args:
            query: 查询字符串
            max_depth: 最大推理深度
            min_confidence: 最小置信度阈值

        Returns:
            推理链

        Raises:
            ReasonerError: 推理失败
        """
        try:
            self.stats["total_inferences"] += 1
            chain = ReasoningChain(query=query)

            # 解析查询(简单实现: 提取主体和关系)
            parts = query.strip().lower().split()
            if len(parts) < 2:
                raise ReasonerError("Invalid query format: at least subject and relation required")

            subject = parts[0]
            relation = parts[1]

            # 并行执行推理任务（串行优化）
            direct_results = self._direct_query(subject, relation)
            transitive_results = self._transitive_reasoning(subject, relation, max_depth)
            symmetric_results = self._symmetric_reasoning(subject, relation)

            # 合并结果并过滤
            all_results = direct_results + transitive_results + symmetric_results
            filtered_results = [r for r in all_results if r.confidence >= min_confidence]

            # 添加到推理链
            for idx, result in enumerate(filtered_results, 1):
                result.step_num = idx
                chain.add_step(result)

            # 设置最终结论(取置信度最高的)
            if chain.steps:
                best_step = max(chain.steps, key=lambda s: s.confidence)
                chain.final_conclusion = best_step.conclusion
                self.stats["successful_inferences"] += 1
            else:
                self.stats["failed_inferences"] += 1

            # 更新统计平均置信度
            total = self.stats["total_inferences"]
            if total > 0 and chain.overall_confidence > 0:
                old_avg = self.stats["avg_confidence"]
                self.stats["avg_confidence"] = (old_avg * (total - 1) + chain.overall_confidence) / total

            return chain

        except ReasonerError:
            self.stats["failed_inferences"] += 1
            raise
        except Exception as e:
            logger.error("Unexpected error during reasoning: %s", e, exc_info=True)
            self.stats["failed_inferences"] += 1
            raise ReasonerError(f"Reasoning failed due to internal error: {e}") from e

    def _direct_query(self, subject: str, relation: str) -> List[ReasoningStep]:
        """
        直接查询

        Args:
            subject: 主体
            relation: 关系

        Returns:
            推理步骤列表
        """
        results: List[ReasoningStep] = []

        # 标准化节点ID（KnowledgeGraph内部使用小写）
        subject_id = subject.lower()

        # 快速存在性检查
        if not self.kg.has_node(subject_id):
            return results

        # 获取出边邻居
        neighbors_dict = self.kg.get_neighbors(subject_id, direction="out")
        out_neighbors = neighbors_dict.get("out", [])

        for neighbor in out_neighbors:
            edge_data = self.kg.get_edge(subject_id, neighbor)
            if not edge_data:
                continue

            stored_rel = edge_data.get("relation_type", "").lower()
            if stored_rel == relation.lower():
                target_name = self.kg.graph.nodes[neighbor].get("name", neighbor)
                confidence = edge_data.get("confidence", 0.5)
                explanation = f"Direct relationship found: {subject} {relation} {target_name}"

                step = ReasoningStep(
                    step_num=0,  # Will be set later
                    rule_type=RuleType.SYLLOGISM,
                    premises=[(subject, relation, target_name)],
                    conclusion=(subject, relation, target_name),
                    confidence=confidence,
                    explanation=explanation,
                )
                results.append(step)

        return results

    def _transitive_reasoning(
        self, subject: str, relation: str, max_depth: int
    ) -> List[ReasoningStep]:
        """
        传递性推理

        Args:
            subject: 主体
            relation: 关系
            max_depth: 最大深度

        Returns:
            推理步骤列表
        """
        results: List[ReasoningStep] = []
        
        # 标准化节点ID
        subject_id = subject.lower()

        # 查找传递性规则
        trans_rule = next(
            (r for r in self.rules if r.rule_type == RuleType.TRANSITIVITY), None
        )
        if not trans_rule or not self.kg.has_node(subject_id):
            return results

        allowed_relations = {r.lower() for r in trans_rule.pattern.get("relations", [])}
        
        visited: Set[str] = set()
        queue: List[Tuple[str, List[Tuple[str, str, str]], float]] = [(subject_id, [], 1.0)]

        while queue:
            current, path, confidence = queue.pop(0)

            if current in visited or len(path) >= max_depth:
                continue

            visited.add(current)

            # 获取邻居
            neighbors_dict = self.kg.get_neighbors(current, direction="out")
            out_neighbors = neighbors_dict.get("out", [])

            for neighbor in out_neighbors:
                edge_data = self.kg.get_edge(current, neighbor)
                if not edge_data:
                    continue
                    
                edge_relation = edge_data.get("relation_type", "")
                if edge_relation.lower() not in allowed_relations:
                    continue

                edge_confidence = edge_data.get("confidence", 0.5)
                current_name = self.kg.graph.nodes[current].get("name", current)
                neighbor_name = self.kg.graph.nodes[neighbor].get("name", neighbor)
                
                new_path = path + [(current_name, edge_relation, neighbor_name)]
                new_confidence = trans_rule.confidence_func(confidence, edge_confidence)

                # 如果是多跳路径，则生成推理步骤
                if len(new_path) > 1:
                    explanation = " -> ".join([f"{p[0]} {p[1]} {p[2]}" for p in new_path])
                    step = ReasoningStep(
                        step_num=0,
                        rule_type=RuleType.TRANSITIVITY,
                        premises=new_path,
                        conclusion=(subject, relation, neighbor_name),
                        confidence=new_confidence,
                        explanation=f"Transitive reasoning: {explanation}",
                    )
                    results.append(step)

                # 继续搜索更深路径
                if len(new_path) < max_depth:
                    queue.append((neighbor, new_path, new_confidence))

        return results

    def _symmetric_reasoning(self, subject: str, _relation: str = None) -> List[ReasoningStep]:
        """
        对称性推理

        Args:
            subject: 主体
            relation: 关系（可忽略）

        Returns:
            推理步骤列表
        """
        results: List[ReasoningStep] = []
        
        # 标准化节点ID
        subject_id = subject.lower()

        # 查找对称性规则
        sym_rule = next(
            (r for r in self.rules if r.rule_type == RuleType.SYMMETRY), None
        )
        if not sym_rule:
            return results

        allowed_relations = {r.lower() for r in sym_rule.pattern.get("relations", [])}

        # 使用生成器表达式提高效率
        nodes = (n for n in self.kg.graph.nodes() if n != subject_id)
        
        for node_id in nodes:
            edge_data = self.kg.get_edge(node_id, subject_id)
            if not edge_data:
                continue

            edge_relation = edge_data.get("relation_type", "").lower()
            if edge_relation not in allowed_relations:
                continue

            node_name = self.kg.graph.nodes[node_id].get("name", node_id)
            raw_confidence = edge_data.get("confidence", 0.5)
            confidence = sym_rule.confidence_func(raw_confidence)

            step = ReasoningStep(
                step_num=0,
                rule_type=RuleType.SYMMETRY,
                premises=[(node_name, edge_relation, subject)],
                conclusion=(subject, edge_relation, node_name),
                confidence=confidence,
                explanation=f"Symmetric relationship: if {node_name} {edge_relation} {subject}, then {subject} {edge_relation} {node_name}",
            )
            results.append(step)

        return results

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        检测知识图谱中的冲突

        Returns:
            冲突列表
        """
        conflicts: List[Dict[str, Any]] = []

        non_symmetric_relations = {"parent_of", "child_of", "part_of", "has_part"}

        for node_id in self.kg.graph.nodes():
            neighbors_dict = self.kg.get_neighbors(node_id, direction="out")
            out_neighbors = neighbors_dict.get("out", [])

            for neighbor in out_neighbors:
                forward_edge = self.kg.get_edge(node_id, neighbor)
                if not forward_edge:
                    continue

                relation = forward_edge.get("relation_type", "")

                # 检查是否存在反向边
                backward_edge = self.kg.get_edge(neighbor, node_id)
                if backward_edge:
                    reverse_relation = backward_edge.get("relation_type", "")

                    # 只有当关系不应是对称时才报告冲突
                    if relation == reverse_relation and relation in non_symmetric_relations:
                        conflicts.append(
                            {
                                "type": "symmetric_conflict",
                                "entities": [node_id, neighbor],
                                "relation": relation,
                                "description": f"Non-symmetric relation used symmetrically: {node_id} {relation} {neighbor} AND {neighbor} {relation} {node_id}",
                            }
                        )

        return conflicts

    def detect_contradictions(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        检测关于特定实体的矛盾信息

        Args:
            entity_id: 实体ID

        Returns:
            矛盾列表
        """
        contradictions: List[Dict[str, Any]] = []
        
        # 获取相关事实
        facts = self.kg.get_node_edges(entity_id)
        if not facts:
            return contradictions

        # 单值属性冲突检测
        single_value_relations = {"born_in", "died_in", "has_gender", "located_in"}
        
        seen_relations: Dict[str, Dict[str, Any]] = {}
        for fact in facts:
            rel_type = fact.get("relation_type")
            if not rel_type:
                continue
                
            target = fact.get("target")
            if not target:
                continue

            if rel_type in single_value_relations:
                if rel_type in seen_relations:
                    prev_target = seen_relations[rel_type]["target"]
                    if prev_target != target:
                        contradictions.append({
                            "type": "single_value_conflict",
                            "relation": rel_type,
                            "value1": prev_target,
                            "value2": target,
                            "confidence1": seen_relations[rel_type]["confidence"],
                            "confidence2": fact.get("confidence", 0.0)
                        })
                else:
                    seen_relations[rel_type] = fact

        return contradictions

    def explain_connection(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """
        解释两个实体之间的连接

        Args:
            entity1: 实体1 ID
            entity2: 实体2 ID

        Returns:
            连接解释字典
        """
        path = self.kg.find_shortest_path(entity1, entity2)
        
        if not path:
            return {
                "connected": False,
                "explanation": f"No direct or indirect connection found between {entity1} and {entity2}."
            }
            
        explanation_steps: List[str] = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            edge_data = self.kg.graph.get_edge_data(u, v)
            if edge_data:
                # 取第一条边的信息
                first_key = next(iter(edge_data.keys()), None)
                if first_key is not None:
                    edge_info = edge_data[first_key]
                    rel_type = edge_info.get("relation_type", "related_to")
                    explanation_steps.append(f"{u} {rel_type} {v}")
        
        return {
            "connected": True,
            "path": path,
            "steps": explanation_steps,
            "explanation": " -> ".join(explanation_steps)
        }

    def propagate_confidence(
        self, node_id: str, initial_confidence: float, max_hops: int = 2
    ) -> Dict[str, float]:
        """
        置信度传播

        Args:
            node_id: 起始节点
            initial_confidence: 初始置信度
            max_hops: 最大传播跳数

        Returns:
            节点置信度字典（使用原始名称作为键）
        """
        # 标准化节点ID
        node_id_lower = node_id.lower()
        
        if not self.kg.has_node(node_id_lower):
            return {}

        node_name = self.kg.graph.nodes[node_id_lower].get("name", node_id)
        confidence_map: Dict[str, float] = {node_name: initial_confidence}
        visited: Set[str] = set()
        queue: List[Tuple[str, float, int]] = [(node_id_lower, initial_confidence, 0)]

        while queue:
            current, confidence, depth = queue.pop(0)

            if current in visited or depth >= max_hops:
                continue

            visited.add(current)

            # 传播到邻居
            neighbors_dict = self.kg.get_neighbors(current, direction="out")
            out_neighbors = neighbors_dict.get("out", [])

            for neighbor in out_neighbors:
                edge_data = self.kg.get_edge(current, neighbor)
                if not edge_data:
                    continue
                    
                edge_confidence = edge_data.get("confidence", 0.5)
                decay_factor = 0.8 ** (depth + 1)
                new_confidence = confidence * edge_confidence * decay_factor

                neighbor_name = self.kg.graph.nodes[neighbor].get("name", neighbor)
                
                # 更新置信度(取最大值)
                if neighbor_name not in confidence_map or new_confidence > confidence_map[neighbor_name]:
                    confidence_map[neighbor_name] = new_confidence
                    queue.append((neighbor, new_confidence, depth + 1))

        return confidence_map

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取推理统计信息

        Returns:
            统计信息字典
        """
        total = self.stats["total_inferences"]
        success_rate = total and (self.stats["successful_inferences"] / total) or 0.0
        
        return {
            "total_inferences": self.stats["total_inferences"],
            "successful_inferences": self.stats["successful_inferences"],
            "failed_inferences": self.stats["failed_inferences"],
            "success_rate": success_rate,
            "avg_confidence": self.stats["avg_confidence"],
            "rule_usage": dict(self.stats["rule_usage"]),
        }


# 示例使用
if __name__ == "__main__":  # pragma: no cover
    # 创建知识图谱
    kg = ArchitectureKnowledgeGraph()

    # 添加示例实体和关系
    python_entity = Entity(name="Python", entity_type="programming_language", confidence=0.95)
    kg.add_node(python_entity)

    django_entity = Entity(name="Django", entity_type="framework", confidence=0.9)
    kg.add_node(django_entity)

    # 添加关系
    kg.add_edge("Python", "Django", relation_type="has_framework", confidence=0.85)

    # 创建推理器
    reasoner = KnowledgeReasoner(kg)

    # 执行推理
    query_text = "python related_to framework"
    chain = reasoner.reason(query_text, max_depth=3, min_confidence=0.5)

    # 打印推理链
    print(f"\nQuery: {chain.query}")
    print(f"Steps: {len(chain.steps)}")
    for step in chain.steps:
        print(f"  Step {step.step_num}: {step.explanation} (confidence: {step.confidence:.2f})")

    if chain.final_conclusion:
        print(
            f"\nFinal conclusion: {chain.final_conclusion} (confidence: {chain.overall_confidence:.2f})"
        )

    # 检测冲突
    conflicts = reasoner.detect_conflicts()
    if conflicts:
        print(f"\nFound {len(conflicts)} conflicts:")
        for conflict in conflicts:
            print(f"  - {conflict['description']}")

    # 打印统计
    stats = reasoner.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")