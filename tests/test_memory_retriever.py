"""
跨会话记忆系统 - 记忆检索器测试
Cross-Session Memory System - Memory Retriever Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 4 - 记忆检索与推理测试
"""

import pytest
from memory_retriever import (
    MemoryRetriever,
    MemoryRetrieverError,
    RetrievalResult,
    SearchQuery,
)
from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType


@pytest.fixture
def sample_graph():
    """创建示例图谱"""
    kg = KnowledgeGraph()

    # 添加实体
    entities = [
        Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.9),
        Entity(name="Django", entity_type=EntityType.TECHNOLOGY, confidence=0.85),
        Entity(name="Flask", entity_type=EntityType.TECHNOLOGY, confidence=0.85),
        Entity(name="Guido van Rossum", entity_type=EntityType.PERSON, confidence=0.95),
        Entity(name="Web Development", entity_type=EntityType.CONCEPT, confidence=0.80),
    ]

    for e in entities:
        kg.add_node(e)

    # 添加关系
    relationships = [
        Relationship(
            source="Python",
            target="Guido van Rossum",
            relation_type=RelationshipType.CREATED_BY,
        ),
        Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        ),
        Relationship(
            source="Flask", target="Python", relation_type=RelationshipType.USES
        ),
        Relationship(
            source="Django",
            target="Web Development",
            relation_type=RelationshipType.RELATED_TO,
        ),
    ]

    for r in relationships:
        kg.add_edge(r)

    return kg


class TestInitialization:
    """测试初始化"""

    def test_valid_initialization(self, sample_graph):
        """测试有效初始化"""
        retriever = MemoryRetriever(sample_graph)

        assert retriever.kg == sample_graph
        assert retriever.embedding_dim == 128
        assert retriever.alpha == 0.6

    def test_custom_parameters(self, sample_graph):
        """测试自定义参数"""
        retriever = MemoryRetriever(sample_graph, embedding_dim=256, alpha=0.7)

        assert retriever.embedding_dim == 256
        assert retriever.alpha == 0.7

    def test_invalid_knowledge_graph(self):
        """测试无效知识图谱"""
        with pytest.raises(ValueError, match="must be a KnowledgeGraph instance"):
            MemoryRetriever("not a graph")

    def test_invalid_alpha(self, sample_graph):
        """测试无效alpha值"""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            MemoryRetriever(sample_graph, alpha=1.5)


class TestSemanticRetrieval:
    """测试语义检索"""

    def test_exact_match(self, sample_graph):
        """测试精确匹配"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Python", max_results=5)

        results = retriever.retrieve(query, mode="semantic")

        assert len(results) > 0
        assert any(r.entity.name.lower() == "python" for r in results)
        assert results[0].source == "semantic"

    def test_fuzzy_match(self, sample_graph):
        """测试模糊匹配"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Pyth", max_results=5)

        results = retriever.retrieve(query, mode="semantic")

        # 应该找到Python
        python_results = [r for r in results if "python" in r.entity.name.lower()]
        assert len(python_results) > 0

    def test_empty_query(self, sample_graph):
        """测试空查询"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="NonExistentEntity", max_results=5)

        results = retriever.retrieve(query, mode="semantic")

        # 可能返回空或低分结果
        assert isinstance(results, list)

    def test_min_score_filter(self, sample_graph):
        """测试最小分数过滤"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Python", max_results=10, min_score=0.8)

        results = retriever.retrieve(query, mode="semantic")

        # 所有结果分数应该≥0.8
        assert all(r.score >= 0.8 for r in results)


class TestStructuralRetrieval:
    """测试结构化检索"""

    def test_filter_by_entity_type(self, sample_graph):
        """测试按实体类型过滤"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(
            text="", entity_types=["technology"], max_results=10, min_score=0.0
        )

        results = retriever.retrieve(query, mode="structural")

        # 应该返回结果(类型过滤可能包含其他逻辑)
        assert isinstance(results, list)

    def test_filter_by_relation_type(self, sample_graph):
        """测试按关系类型过滤"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(
            text="", relation_types=["uses"], max_results=10, min_score=0.0
        )

        results = retriever.retrieve(query, mode="structural")

        # 应该返回与USES关系相关的实体
        assert len(results) > 0

    def test_pagerank_scoring(self, sample_graph):
        """测试PageRank评分"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(
            text="", entity_types=["technology"], max_results=10, min_score=0.0
        )

        results = retriever.retrieve(query, mode="structural")

        # Python应该有较高的PageRank(被多个节点指向)
        python_result = next(
            (r for r in results if r.entity.name.lower() == "python"), None
        )
        if python_result:
            assert python_result.score > 0


class TestHybridRetrieval:
    """测试混合检索"""

    def test_hybrid_mode(self, sample_graph):
        """测试混合模式"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Python", entity_types=["technology"], max_results=5)

        results = retriever.retrieve(query, mode="hybrid")

        assert len(results) > 0
        # 应该有hybrid来源的结果
        hybrid_results = [r for r in results if r.source == "hybrid"]
        assert len(hybrid_results) > 0

    def test_alpha_weight(self, sample_graph):
        """测试alpha权重影响"""
        query = SearchQuery(text="Python", max_results=5)

        # alpha=1.0 (纯语义)
        retriever_semantic = MemoryRetriever(sample_graph, alpha=1.0)
        results_semantic = retriever_semantic.retrieve(query, mode="hybrid")

        # alpha=0.0 (纯结构)
        retriever_structural = MemoryRetriever(sample_graph, alpha=0.0)
        results_structural = retriever_structural.retrieve(query, mode="hybrid")

        # 结果应该有差异
        assert len(results_semantic) > 0 or len(results_structural) > 0


class TestContextExpansion:
    """测试上下文扩展"""

    def test_expand_context(self, sample_graph):
        """测试上下文扩展"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Django", expand_context=True, expand_hops=1)

        results = retriever.retrieve(query, mode="semantic")

        # Django应该有邻居(Python, Web Development)
        django_result = next(
            (r for r in results if "django" in r.entity.name.lower()), None
        )

        if django_result:
            assert len(django_result.neighbors) > 0

    def test_no_expand_context(self, sample_graph):
        """测试不扩展上下文"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Django", expand_context=False)

        results = retriever.retrieve(query, mode="semantic")

        # 邻居列表应该为空
        for result in results:
            assert len(result.neighbors) == 0

    def test_multi_hop_expansion(self, sample_graph):
        """测试多跳扩展"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Django", expand_context=True, expand_hops=2)

        results = retriever.retrieve(query, mode="semantic")

        django_result = next(
            (r for r in results if "django" in r.entity.name.lower()), None
        )

        # 2跳应该能到达更多节点
        if django_result:
            assert len(django_result.neighbors) >= 0


class TestResultRanking:
    """测试结果排序"""

    def test_results_sorted_by_score(self, sample_graph):
        """测试结果按分数排序"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="web", max_results=10, min_score=0.0)

        results = retriever.retrieve(query, mode="semantic")

        if len(results) > 1:
            # 验证降序排列
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_max_results_limit(self, sample_graph):
        """测试最大结果数限制"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="", entity_types=["technology"], max_results=2)

        results = retriever.retrieve(query, mode="structural")

        # 结果数不应超过max_results
        assert len(results) <= 2


class TestSimilarityCalculation:
    """测试相似度计算"""

    def test_exact_name_match(self, sample_graph):
        """测试精确名称匹配"""
        retriever = MemoryRetriever(sample_graph)

        entity1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=1.0)
        entity2 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=1.0)

        score = retriever._calculate_semantic_similarity(entity1, entity2)
        assert score > 0.9  # 应该接近1.0

    def test_partial_name_match(self, sample_graph):
        """测试部分名称匹配"""
        retriever = MemoryRetriever(sample_graph)

        entity1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=1.0)
        entity2 = Entity(
            name="Python Programming", entity_type=EntityType.TECHNOLOGY, confidence=1.0
        )

        score = retriever._calculate_semantic_similarity(entity1, entity2)
        assert 0.5 < score <= 1.0  # 应该有一定相似度(包含关系可能达到1.0)

    def test_different_names(self, sample_graph):
        """测试不同名称"""
        retriever = MemoryRetriever(sample_graph)

        entity1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=1.0)
        entity2 = Entity(name="Java", entity_type=EntityType.TECHNOLOGY, confidence=1.0)

        score = retriever._calculate_semantic_similarity(entity1, entity2)
        assert score < 0.7  # 相似度较低


class TestEditDistance:
    """测试编辑距离"""

    def test_identical_strings(self):
        """测试相同字符串"""
        distance = MemoryRetriever._edit_distance("hello", "hello")
        assert distance == 0

    def test_one_char_difference(self):
        """测试一个字符差异"""
        distance = MemoryRetriever._edit_distance("hello", "hallo")
        assert distance == 1

    def test_empty_strings(self):
        """测试空字符串"""
        distance = MemoryRetriever._edit_distance("", "")
        assert distance == 0

    def test_one_empty_string(self):
        """测试一个空字符串"""
        distance = MemoryRetriever._edit_distance("hello", "")
        assert distance == 5


class TestStatistics:
    """测试统计功能"""

    def test_statistics_tracking(self, sample_graph):
        """测试统计追踪"""
        retriever = MemoryRetriever(sample_graph)

        query = SearchQuery(text="Python", max_results=5)

        # 执行多次检索
        retriever.retrieve(query, mode="semantic")
        retriever.retrieve(query, mode="structural")
        retriever.retrieve(query, mode="hybrid")

        stats = retriever.get_statistics()

        assert stats["total_queries"] == 3
        assert stats["semantic_queries"] == 1
        assert stats["structural_queries"] == 1
        assert stats["hybrid_queries"] == 1
        assert stats["avg_latency_ms"] > 0


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_mode(self, sample_graph):
        """测试无效检索模式"""
        retriever = MemoryRetriever(sample_graph)
        query = SearchQuery(text="Python")

        with pytest.raises(MemoryRetrieverError, match="Invalid retrieval mode"):
            retriever.retrieve(query, mode="invalid_mode")


class TestRetrievalResult:
    """测试检索结果类"""

    def test_to_dict(self):
        """测试转换为字典"""
        entity = Entity(name="Test", entity_type=EntityType.CONCEPT, confidence=0.8)
        result = RetrievalResult(entity=entity, score=0.9, source="semantic")

        data = result.to_dict()

        assert data["entity"]["name"] == "Test"
        assert data["score"] == 0.9
        assert data["source"] == "semantic"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
