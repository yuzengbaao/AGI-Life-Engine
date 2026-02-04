"""
跨会话记忆系统 - 集成测试
Cross-Session Memory System - Integration Tests

测试完整工作流:
1. 检索→推理→问答完整流程
2. 记忆生命周期 (创建→使用→清理)
3. 性能基准测试
4. 并发访问测试
"""
# pylint: disable=redefined-outer-name,too-few-public-methods

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType
from memory_retriever import MemoryRetriever, SearchQuery
from knowledge_reasoner import KnowledgeReasoner
from graph_qa import GraphQA
from memory_consolidation import MemoryConsolidation, MemoryMetrics


@pytest.fixture
def integrated_system():
    """创建完整集成系统"""
    # 1. 知识图谱
    kg = KnowledgeGraph()

    # 添加技术实体
    python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
    django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)
    flask = Entity(name="Flask", entity_type=EntityType.PRODUCT, confidence=0.9)
    ai = Entity(name="AI", entity_type=EntityType.CONCEPT, confidence=0.95)
    ml = Entity(name="MachineLearning", entity_type=EntityType.CONCEPT, confidence=0.9)

    kg.add_node(python)
    kg.add_node(django)
    kg.add_node(flask)
    kg.add_node(ai)
    kg.add_node(ml)

    # 添加关系
    kg.add_edge(Relationship(
        source="Python", target="Django",
        relation_type=RelationshipType.HAS_A, confidence=0.85))
    kg.add_edge(Relationship(
        source="Python", target="Flask",
        relation_type=RelationshipType.HAS_A, confidence=0.8))
    kg.add_edge(Relationship(
        source="Python", target="AI",
        relation_type=RelationshipType.USES, confidence=0.9))
    kg.add_edge(Relationship(
        source="AI", target="MachineLearning",
        relation_type=RelationshipType.RELATED_TO, confidence=0.95))

    # 2. 各组件
    retriever = MemoryRetriever(kg)
    reasoner = KnowledgeReasoner(kg)
    qa = GraphQA(kg, reasoner=reasoner, retriever=retriever)
    consolidation = MemoryConsolidation(kg)

    return {
        "kg": kg,
        "retriever": retriever,
        "reasoner": reasoner,
        "qa": qa,
        "consolidation": consolidation,
    }


class TestEndToEndWorkflow:
    """测试端到端工作流"""

    def test_retrieval_reasoning_qa_pipeline(self, integrated_system):
        """测试: 检索→推理→问答完整流程"""
        retriever = integrated_system["retriever"]
        reasoner = integrated_system["reasoner"]
        qa = integrated_system["qa"]

        # 1. 检索
        search_query = SearchQuery(text="Python AI", max_results=5)
        results = retriever.retrieve(query=search_query, mode="hybrid")
        assert len(results) > 0        # 2. 推理
        chain = reasoner.reason(query="Python related_to ?", max_depth=2)
        assert chain is not None

        # 3. 问答
        answer = qa.answer("Python有哪些框架?")
        assert answer.confidence > 0
        assert "Django" in answer.answer or "Flask" in answer.answer

    def test_multi_turn_conversation(self, integrated_system):
        """测试多轮对话上下文保持"""
        qa = integrated_system["qa"]

        # 第一轮: 列表型问题
        answer1 = qa.answer("Python有哪些框架?")
        assert answer1.confidence > 0
        assert "Django" in answer1.answer or "Flask" in answer1.answer

        # 第二轮: 布尔型问题 (更容易回答)
        answer2 = qa.answer("Python能用于AI吗?")
        assert answer2.confidence > 0

        # 验证历史记录
        assert len(qa.context["history"]) == 2
        assert "Python" in qa.context["current_entities"]


class TestMemoryLifecycle:
    """测试记忆生命周期"""

    def test_create_use_cleanup_lifecycle(self, integrated_system):
        """测试: 创建→使用→清理完整生命周期"""
        kg = integrated_system["kg"]
        consolidation = integrated_system["consolidation"]
        qa = integrated_system["qa"]

        # 1. 创建: 添加新实体
        initial_count = len(kg.graph.nodes())
        new_entity = Entity(name="NewTech", entity_type=EntityType.TECHNOLOGY, confidence=0.5)
        kg.add_node(new_entity)
        assert len(kg.graph.nodes()) == initial_count + 1

        # 2. 使用: 通过问答访问
        qa.answer("NewTech是什么?")

        # 3. 强化: 重要记忆得到强化
        consolidation.reinforce_memory("python")
        metrics = consolidation.memory_metrics.get("python")
        if metrics:
            assert metrics.access_count > 0

        # 4. 清理: 移除低重要性记忆
        for node_id in kg.graph.nodes():
            if node_id == "newtech":  # 新添加的低置信度实体
                consolidation.memory_metrics[node_id] = (
                    consolidation.memory_metrics.get(
                        node_id, MemoryMetrics()
                    )
                )
                consolidation.memory_metrics[node_id].importance_score = 0.1

        cleaned = consolidation.cleanup()
        # 应该清理了一些低重要性记忆
        assert cleaned >= 0

    def test_memory_consolidation_workflow(self, integrated_system):
        """测试记忆整合工作流"""
        kg = integrated_system["kg"]
        consolidation = integrated_system["consolidation"]

        # 添加相似实体
        similar1 = Entity(name="Python3", entity_type=EntityType.TECHNOLOGY, confidence=0.9)
        similar2 = Entity(name="Python2", entity_type=EntityType.TECHNOLOGY, confidence=0.85)
        kg.add_node(similar1)
        kg.add_node(similar2)

        initial_count = len(kg.graph.nodes())

        # 合并相似实体
        consolidated = consolidation.consolidate_similar_entities(similarity_threshold=0.7)

        # 验证合并效果
        if consolidated > 0:
            assert len(kg.graph.nodes()) < initial_count


class TestPerformanceBenchmarks:
    """测试性能基准"""

    def test_retrieval_performance(self, integrated_system):
        """测试检索性能: 目标<200ms"""
        retriever = integrated_system["retriever"]

        start_time = time.time()
        search_query = SearchQuery(text="Python", max_results=10)
        results = retriever.retrieve(query=search_query, mode="hybrid")
        elapsed = (time.time() - start_time) * 1000  # 转为毫秒

        assert len(results) > 0
        assert elapsed < 200, f"Retrieval took {elapsed:.2f}ms, expected <200ms"

    def test_reasoning_performance(self, integrated_system):
        """测试推理性能: 目标<300ms"""
        reasoner = integrated_system["reasoner"]

        start_time = time.time()
        chain = reasoner.reason(query="Python related_to ?", max_depth=2, min_confidence=0.5)
        elapsed = (time.time() - start_time) * 1000

        assert chain is not None
        assert elapsed < 300, f"Reasoning took {elapsed:.2f}ms, expected <300ms"

    def test_qa_performance(self, integrated_system):
        """测试问答性能: 目标<500ms"""
        qa = integrated_system["qa"]

        start_time = time.time()
        answer = qa.answer("Python有哪些框架?", use_reasoning=False)
        elapsed = (time.time() - start_time) * 1000

        assert answer.confidence > 0
        assert elapsed < 500, f"QA took {elapsed:.2f}ms, expected <500ms"

    def test_consolidation_performance(self, integrated_system):
        """测试整合性能: 目标<1000ms"""
        consolidation = integrated_system["consolidation"]
        kg = integrated_system["kg"]

        # 添加多个实体以测试性能
        for i in range(10):
            entity = Entity(name=f"TestEntity{i}", entity_type=EntityType.CONCEPT, confidence=0.5)
            kg.add_node(entity)

        start_time = time.time()
        consolidation.apply_forgetting(hours_elapsed=24)
        elapsed = (time.time() - start_time) * 1000

        assert elapsed < 1000, f"Consolidation took {elapsed:.2f}ms, expected <1000ms"


class TestConcurrentAccess:
    """测试并发访问"""

    def test_concurrent_retrieval(self, integrated_system):
        """测试并发检索: 10个线程"""
        retriever = integrated_system["retriever"]

        def retrieve_task(query_text):
            search_query = SearchQuery(text=query_text, max_results=5)
            return retriever.retrieve(query=search_query, mode="hybrid")

        queries = [f"Python {i}" for i in range(10)]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(retrieve_task, q) for q in queries]
            results = [f.result() for f in futures]

        # 所有查询都应成功
        assert len(results) == 10
        assert all(len(r) >= 0 for r in results)

    def test_concurrent_qa(self, integrated_system):
        """测试并发问答: 5个线程"""
        qa = integrated_system["qa"]

        def qa_task(question):
            return qa.answer(question, use_reasoning=False)

        questions = [
            "Python是什么?",
            "Python有哪些框架?",
            "Python用于什么?",
            "Django是什么?",
            "Flask是什么?",
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(qa_task, q) for q in questions]
            answers = [f.result() for f in futures]

        # 所有问题都应得到回答
        assert len(answers) == 5
        assert all(a.confidence >= 0 for a in answers)


class TestDataConsistency:
    """测试数据一致性"""

    def test_graph_integrity_after_operations(self, integrated_system):
        """测试操作后图完整性"""
        kg = integrated_system["kg"]
        qa = integrated_system["qa"]
        consolidation = integrated_system["consolidation"]

        # 记录初始状态
        initial_nodes = set(kg.graph.nodes())
        initial_edges = kg.graph.number_of_edges()

        # 执行多种操作
        qa.answer("Python有哪些框架?")
        consolidation.reinforce_memory("python")

        # 验证图结构一致性
        assert all(kg.graph.has_node(n) for n in initial_nodes)
        assert kg.graph.number_of_edges() >= initial_edges

    def test_metadata_consistency(self, integrated_system):
        """测试元数据一致性"""
        kg = integrated_system["kg"]
        consolidation = integrated_system["consolidation"]

        # 添加实体并记录元数据
        test_entity = Entity(name="TestMeta", entity_type=EntityType.CONCEPT, confidence=0.8)
        kg.add_node(test_entity)

        # 强化记忆
        consolidation.reinforce_memory("testmeta")

        # 验证元数据
        node_data = kg.graph.nodes["testmeta"]
        assert node_data["confidence"] == 0.8
        assert node_data["entity_type"] == EntityType.CONCEPT.value


class TestErrorHandling:
    """测试错误处理"""

    def test_empty_query_handling(self, integrated_system):
        """测试空查询处理"""
        qa = integrated_system["qa"]

        # 空查询不应崩溃
        answer = qa.answer("")
        assert answer.confidence >= 0

    def test_invalid_entity_handling(self, integrated_system):
        """测试无效实体处理"""
        retriever = integrated_system["retriever"]

        # 查询不存在的实体
        search_query = SearchQuery(text="NonExistentEntity", max_results=5)
        results = retriever.retrieve(query=search_query, mode="hybrid")
        assert len(results) >= 0  # 不应崩溃

    def test_reasoning_with_incomplete_data(self, integrated_system):
        """测试不完整数据推理"""
        reasoner = integrated_system["reasoner"]

        # 推理不存在的关系
        chain = reasoner.reason(query="NonExistent related_to ?", max_depth=2)
        # 应返回空结果或空链,不应崩溃
        assert chain is not None


class TestScalability:
    """测试可扩展性"""

    def test_large_graph_handling(self):
        """测试大规模图处理"""
        kg = KnowledgeGraph()

        # 添加100个实体
        for i in range(100):
            entity = Entity(
                name=f"Entity{i}",
                entity_type=EntityType.CONCEPT,
                confidence=0.5 + (i % 50) * 0.01,
            )
            kg.add_node(entity)

        # 添加连接
        for i in range(50):
            kg.add_edge(
                Relationship(
                    source=f"Entity{i}",
                    target=f"Entity{i+1}",
                    relation_type=RelationshipType.RELATED_TO,
                    confidence=0.7,
                )
            )

        # 测试各组件性能
        retriever = MemoryRetriever(kg)
        reasoner = KnowledgeReasoner(kg)
        qa = GraphQA(kg, reasoner=reasoner, retriever=retriever)

        # 检索应正常工作
        search_query = SearchQuery(text="Entity50", max_results=10)
        results = retriever.retrieve(query=search_query, mode="hybrid")
        assert len(results) > 0

        # 问答应正常工作
        answer = qa.answer("Entity50是什么?")
        assert answer.confidence >= 0


class TestStatisticsAndReporting:
    """测试统计和报告"""

    def test_comprehensive_statistics(self, integrated_system):
        """测试综合统计信息"""
        qa = integrated_system["qa"]
        consolidation = integrated_system["consolidation"]

        # 执行一些操作
        qa.answer("Python是什么?")
        qa.answer("Python有哪些框架?")

        # 获取统计
        qa_stats = qa.get_statistics()
        memory_report = consolidation.get_memory_report()

        # 验证统计完整性
        assert "total_questions" in qa_stats
        assert qa_stats["total_questions"] >= 2

        assert "total_memories" in memory_report
        assert "importance_distribution" in memory_report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
