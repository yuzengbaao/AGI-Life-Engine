"""
测试知识推理器 - Knowledge Reasoner Tests

版本: 1.0.0
日期: 2025-01-14
阶段: Phase 4 - 记忆检索与推理
"""

import pytest
from knowledge_reasoner import (
    KnowledgeReasoner,
    ReasonerError,
    RuleType,
    ReasoningRule,
    ReasoningStep,
    ReasoningChain,
)
from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType


class TestInitialization:
    """测试初始化"""

    def test_valid_initialization(self):
        """测试有效初始化"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        assert reasoner.kg == kg
        assert reasoner.query_engine is not None
        assert len(reasoner.rules) > 0  # 应该有预定义规则
        assert reasoner.stats["total_inferences"] == 0

    def test_invalid_knowledge_graph(self):
        """测试无效知识图谱"""
        with pytest.raises(ReasonerError):
            KnowledgeReasoner("not a graph")


class TestRuleInitialization:
    """测试规则初始化"""

    def test_transitivity_rule(self):
        """测试传递性规则"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        trans_rule = next(
            (r for r in reasoner.rules if r.rule_type == RuleType.TRANSITIVITY), None
        )

        assert trans_rule is not None
        assert "relations" in trans_rule.pattern
        assert len(trans_rule.pattern["relations"]) > 0

    def test_symmetry_rule(self):
        """测试对称性规则"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        sym_rule = next((r for r in reasoner.rules if r.rule_type == RuleType.SYMMETRY), None)

        assert sym_rule is not None
        assert "relations" in sym_rule.pattern

    def test_inverse_rule(self):
        """测试逆关系规则"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        inv_rule = next((r for r in reasoner.rules if r.rule_type == RuleType.INVERSE), None)

        assert inv_rule is not None
        assert "inverse_pairs" in inv_rule.pattern


class TestDirectQuery:
    """测试直接查询"""

    @pytest.fixture
    def sample_graph(self):
        """创建示例图谱"""
        kg = KnowledgeGraph()

        # 添加实体
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)
        flask = Entity(name="Flask", entity_type=EntityType.PRODUCT, confidence=0.9)

        kg.add_node(python)
        kg.add_node(django)
        kg.add_node(flask)

        # 添加关系
        kg.add_edge(Relationship(source="Python", target="Django", relation_type=RelationshipType.HAS_A, confidence=0.85))
        kg.add_edge(Relationship(source="Python", target="Flask", relation_type=RelationshipType.HAS_A, confidence=0.8))

        return kg

    def test_direct_relationship_found(self, sample_graph):
        """测试找到直接关系"""
        reasoner = KnowledgeReasoner(sample_graph)

        results = reasoner._direct_query("Python", "has_a")

        assert len(results) > 0
        assert all(isinstance(r, ReasoningStep) for r in results)
        assert all(r.conclusion[0] == "Python" for r in results)

    def test_no_direct_relationship(self, sample_graph):
        """测试未找到直接关系"""
        reasoner = KnowledgeReasoner(sample_graph)

        results = reasoner._direct_query("Django", "has_framework")

        assert len(results) == 0

    def test_nonexistent_subject(self, sample_graph):
        """测试不存在的主体"""
        reasoner = KnowledgeReasoner(sample_graph)

        results = reasoner._direct_query("Java", "has_framework")

        assert len(results) == 0


class TestTransitiveReasoning:
    """测试传递性推理"""

    @pytest.fixture
    def transitive_graph(self):
        """创建传递性图谱"""
        kg = KnowledgeGraph()

        # A -> B -> C 传递性路径
        a = Entity(name="A", entity_type=EntityType.CONCEPT, confidence=0.9)
        b = Entity(name="B", entity_type=EntityType.CONCEPT, confidence=0.9)
        c = Entity(name="C", entity_type=EntityType.CONCEPT, confidence=0.9)

        kg.add_node(a)
        kg.add_node(b)
        kg.add_node(c)

        kg.add_edge(Relationship(source="A", target="B", relation_type=RelationshipType.RELATED_TO, confidence=0.8))
        kg.add_edge(Relationship(source="B", target="C", relation_type=RelationshipType.RELATED_TO, confidence=0.7))

        return kg

    def test_two_hop_transitivity(self, transitive_graph):
        """测试两跳传递性"""
        reasoner = KnowledgeReasoner(transitive_graph)

        results = reasoner._transitive_reasoning("A", "related_to", max_depth=3)

        # 应该找到 A -> B -> C 的传递路径
        assert len(results) > 0
        # 验证有到C的推理
        assert any(r.conclusion[2] == "C" for r in results)

    def test_max_depth_limit(self, transitive_graph):
        """测试最大深度限制"""
        reasoner = KnowledgeReasoner(transitive_graph)

        # 深度为1时应该只找到直接邻居
        results = reasoner._transitive_reasoning("A", "related_to", max_depth=1)

        # 结果中不应该有到C的推理(需要2跳)
        assert not any(r.conclusion[2] == "C" for r in results)


class TestSymmetricReasoning:
    """测试对称性推理"""

    @pytest.fixture
    def symmetric_graph(self):
        """创建对称性图谱"""
        kg = KnowledgeGraph()

        x = Entity(name="X", entity_type=EntityType.CONCEPT, confidence=0.9)
        y = Entity(name="Y", entity_type=EntityType.CONCEPT, confidence=0.9)

        kg.add_node(x)
        kg.add_node(y)

        # X -> Y (对称关系)
        kg.add_edge(Relationship(source="X", target="Y", relation_type=RelationshipType.SIMILAR_TO, confidence=0.8))

        return kg

    def test_symmetric_relationship(self, symmetric_graph):
        """测试对称关系推理"""
        reasoner = KnowledgeReasoner(symmetric_graph)

        results = reasoner._symmetric_reasoning("Y", "similar_to")

        # 应该推理出 Y similar_to X
        assert len(results) > 0
        assert any(r.conclusion == ("Y", "similar_to", "X") for r in results)


class TestReasoningChain:
    """测试推理链"""

    def test_empty_chain(self):
        """测试空推理链"""
        chain = ReasoningChain(query="test query")

        assert chain.query == "test query"
        assert len(chain.steps) == 0
        assert chain.final_conclusion is None
        assert chain.overall_confidence == 0.0

    def test_add_step(self):
        """测试添加步骤"""
        chain = ReasoningChain(query="test")

        step = ReasoningStep(
            step_num=1,
            rule_type=RuleType.TRANSITIVITY,
            premises=[("A", "related_to", "B")],
            conclusion=("A", "related_to", "B"),
            confidence=0.8,
            explanation="Test step",
        )

        chain.add_step(step)

        assert len(chain.steps) == 1
        assert chain.overall_confidence == 0.8

    def test_confidence_update(self):
        """测试置信度更新"""
        chain = ReasoningChain(query="test")

        step1 = ReasoningStep(
            step_num=1,
            rule_type=RuleType.TRANSITIVITY,
            premises=[],
            conclusion=("A", "r", "B"),
            confidence=0.9,
            explanation="",
        )

        step2 = ReasoningStep(
            step_num=2,
            rule_type=RuleType.TRANSITIVITY,
            premises=[],
            conclusion=("B", "r", "C"),
            confidence=0.7,
            explanation="",
        )

        chain.add_step(step1)
        chain.add_step(step2)

        # 总体置信度应该是最小值
        assert chain.overall_confidence == 0.7

    def test_to_dict(self):
        """测试转换为字典"""
        chain = ReasoningChain(query="test query")

        step = ReasoningStep(
            step_num=1,
            rule_type=RuleType.SYMMETRY,
            premises=[("X", "similar_to", "Y")],
            conclusion=("Y", "similar_to", "X"),
            confidence=0.8,
            explanation="Symmetric relation",
        )

        chain.add_step(step)
        chain.final_conclusion = ("Y", "similar_to", "X")

        result = chain.to_dict()

        assert result["query"] == "test query"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["rule_type"] == "symmetry"
        assert result["final_conclusion"] == ("Y", "similar_to", "X")
        assert result["overall_confidence"] == 0.8


class TestFullReasoning:
    """测试完整推理"""

    @pytest.fixture
    def complex_graph(self):
        """创建复杂图谱"""
        kg = KnowledgeGraph()

        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)
        web = Entity(name="Web", entity_type=EntityType.CONCEPT, confidence=0.85)

        kg.add_node(python)
        kg.add_node(django)
        kg.add_node(web)

        kg.add_edge(Relationship(source="Python", target="Django", relation_type=RelationshipType.HAS_A, confidence=0.85))
        kg.add_edge(Relationship(source="Django", target="Web", relation_type=RelationshipType.USES, confidence=0.8))

        return kg

    def test_successful_reasoning(self, complex_graph):
        """测试成功推理"""
        reasoner = KnowledgeReasoner(complex_graph)

        chain = reasoner.reason("python has_framework django", max_depth=2, min_confidence=0.5)

        assert isinstance(chain, ReasoningChain)
        assert chain.query == "python has_framework django"
        assert reasoner.stats["total_inferences"] == 1
        assert reasoner.stats["successful_inferences"] >= 0

    def test_invalid_query_format(self):
        """测试无效查询格式"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        with pytest.raises(ReasonerError):
            reasoner.reason("invalid", max_depth=2)

    def test_empty_graph_reasoning(self):
        """测试空图谱推理"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        # 应该不抛出异常,但结果为空
        chain = reasoner.reason("python has_framework django", max_depth=2)

        assert len(chain.steps) == 0
        assert chain.final_conclusion is None


class TestConflictDetection:
    """测试冲突检测"""

    @pytest.fixture
    def conflict_graph(self):
        """创建有冲突的图谱"""
        kg = KnowledgeGraph()

        a = Entity(name="A", entity_type=EntityType.PERSON, confidence=0.9)
        b = Entity(name="B", entity_type=EntityType.PERSON, confidence=0.9)

        kg.add_node(a)
        kg.add_node(b)

        # 创建矛盾关系: A parent_of B AND B parent_of A
        kg.add_edge(Relationship(source="A", target="B", relation_type=RelationshipType.HAS_A, confidence=0.8))
        kg.add_edge(Relationship(source="B", target="A", relation_type=RelationshipType.HAS_A, confidence=0.8))

        return kg

    def test_detect_symmetric_conflict(self, conflict_graph):
        """测试检测对称冲突"""
        reasoner = KnowledgeReasoner(conflict_graph)

        conflicts = reasoner.detect_conflicts()

        assert len(conflicts) > 0
        assert any(c["type"] == "symmetric_conflict" for c in conflicts)

    def test_no_conflicts(self):
        """测试无冲突情况"""
        kg = KnowledgeGraph()

        a = Entity(name="A", entity_type=EntityType.CONCEPT, confidence=0.9)
        b = Entity(name="B", entity_type=EntityType.CONCEPT, confidence=0.9)

        kg.add_node(a)
        kg.add_node(b)

        # 正常的对称关系
        kg.add_edge(Relationship(source="A", target="B", relation_type=RelationshipType.SIMILAR_TO, confidence=0.8))
        kg.add_edge(Relationship(source="B", target="A", relation_type=RelationshipType.SIMILAR_TO, confidence=0.8))

        reasoner = KnowledgeReasoner(kg)
        conflicts = reasoner.detect_conflicts()

        # 对称关系不应该被检测为冲突
        assert len(conflicts) == 0


class TestConfidencePropagation:
    """测试置信度传播"""

    @pytest.fixture
    def propagation_graph(self):
        """创建传播图谱"""
        kg = KnowledgeGraph()

        # A -> B -> C 链
        a = Entity(name="A", entity_type=EntityType.CONCEPT, confidence=0.9)
        b = Entity(name="B", entity_type=EntityType.CONCEPT, confidence=0.9)
        c = Entity(name="C", entity_type=EntityType.CONCEPT, confidence=0.9)

        kg.add_node(a)
        kg.add_node(b)
        kg.add_node(c)

        kg.add_edge(Relationship(source="A", target="B", relation_type=RelationshipType.RELATED_TO, confidence=0.8))
        kg.add_edge(Relationship(source="B", target="C", relation_type=RelationshipType.RELATED_TO, confidence=0.7))

        return kg

    def test_confidence_propagation(self, propagation_graph):
        """测试置信度传播"""
        reasoner = KnowledgeReasoner(propagation_graph)

        confidence_map = reasoner.propagate_confidence("A", initial_confidence=1.0, max_hops=2)

        assert "A" in confidence_map
        assert confidence_map["A"] == 1.0
        assert "B" in confidence_map
        assert confidence_map["B"] < 1.0  # 应该衰减
        assert "C" in confidence_map
        assert confidence_map["C"] < confidence_map["B"]  # 进一步衰减

    def test_nonexistent_node_propagation(self):
        """测试不存在节点的传播"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        confidence_map = reasoner.propagate_confidence("NonExistent", initial_confidence=1.0)

        assert len(confidence_map) == 0

    def test_max_hops_limit(self, propagation_graph):
        """测试最大跳数限制"""
        reasoner = KnowledgeReasoner(propagation_graph)

        # 只传播1跳
        confidence_map = reasoner.propagate_confidence("A", initial_confidence=1.0, max_hops=1)

        assert "A" in confidence_map
        assert "B" in confidence_map
        # C在2跳外,不应该包含
        assert "C" not in confidence_map


class TestStatistics:
    """测试统计功能"""

    def test_statistics_tracking(self):
        """测试统计追踪"""
        kg = KnowledgeGraph()

        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)

        kg.add_node(python)
        kg.add_node(django)

        kg.add_edge(Relationship(source="Python", target="Django", relation_type=RelationshipType.HAS_A, confidence=0.85))

        reasoner = KnowledgeReasoner(kg)

        # 执行推理
        reasoner.reason("python has_framework django", max_depth=2)

        stats = reasoner.get_statistics()

        assert stats["total_inferences"] == 1
        assert "successful_inferences" in stats
        assert "failed_inferences" in stats
        assert "success_rate" in stats
        assert "avg_confidence" in stats

    def test_success_rate_calculation(self):
        """测试成功率计算"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)

        # 初始成功率应该是0
        stats = reasoner.get_statistics()
        assert stats["success_rate"] == 0.0

        # 执行失败的推理
        try:
            reasoner.reason("invalid query", max_depth=2)
        except ReasonerError:
            pass

        stats = reasoner.get_statistics()
        assert stats["total_inferences"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
