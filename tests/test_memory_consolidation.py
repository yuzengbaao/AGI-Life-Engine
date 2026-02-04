"""
跨会话记忆系统 - 记忆整合测试
Cross-Session Memory System - Memory Consolidation Tests

测试覆盖:
1. 重要性评分
2. 遗忘策略 (时间衰减、频率、LRU、混合)
3. 记忆合并
4. 自动清理
5. 记忆强化
6. 统计报告
"""

import pytest
from datetime import datetime, timedelta
from memory_consolidation import (
    MemoryConsolidation,
    ImportanceLevel,
    ForgettingStrategy,
    MemoryMetrics,
    MemoryConsolidationError,
)
from knowledge_graph import KnowledgeGraph
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType


class TestInitialization:
    """测试初始化"""

    def test_valid_initialization(self):
        """测试正确初始化"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)

        assert mc.kg == kg
        assert mc.forgetting_strategy == ForgettingStrategy.HYBRID
        assert mc.cleanup_threshold == 0.2
        assert mc.max_memory_size == 10000

    def test_invalid_knowledge_graph(self):
        """测试无效知识图谱"""
        with pytest.raises(MemoryConsolidationError):
            MemoryConsolidation("invalid")

    def test_custom_parameters(self):
        """测试自定义参数"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(
            kg,
            forgetting_strategy=ForgettingStrategy.TIME_DECAY,
            cleanup_threshold=0.3,
            max_memory_size=5000,
        )

        assert mc.forgetting_strategy == ForgettingStrategy.TIME_DECAY
        assert mc.cleanup_threshold == 0.3
        assert mc.max_memory_size == 5000


class TestImportanceCalculation:
    """测试重要性计算"""

    @pytest.fixture
    def kg_with_data(self):
        """创建包含数据的知识图谱"""
        kg = KnowledgeGraph()

        # 添加实体
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)
        flask = Entity(name="Flask", entity_type=EntityType.PRODUCT, confidence=0.8)

        kg.add_node(python)
        kg.add_node(django)
        kg.add_node(flask)

        # 添加关系 (增加连接度)
        kg.add_edge(
            Relationship(
                source="Python",
                target="Django",
                relation_type=RelationshipType.HAS_A,
                confidence=0.85,
            )
        )
        kg.add_edge(
            Relationship(
                source="Python",
                target="Flask",
                relation_type=RelationshipType.HAS_A,
                confidence=0.8,
            )
        )

        return kg

    def test_basic_importance(self, kg_with_data):
        """测试基本重要性计算"""
        mc = MemoryConsolidation(kg_with_data)
        score = mc.calculate_importance("python")
        assert 0.0 <= score <= 1.0

    def test_importance_factors(self, kg_with_data):
        """测试重要性因素权重"""
        mc = MemoryConsolidation(kg_with_data)
        
        # 自定义权重
        custom_factors = {
            "confidence": 0.5,
            "connectivity": 0.3,
            "access_freq": 0.1,
            "recency": 0.1,
        }
        
        score = mc.calculate_importance("python", factors=custom_factors)
        assert 0.0 <= score <= 1.0

    def test_nonexistent_node_importance(self):
        """测试不存在节点的重要性"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        score = mc.calculate_importance("nonexistent")
        assert score == 0.0


class TestForgettingStrategies:
    """测试遗忘策略"""

    @pytest.fixture
    def mc_with_metrics(self):
        """创建包含度量指标的整合器"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(python)

        mc = MemoryConsolidation(kg)
        
        # 添加度量指标
        metrics = MemoryMetrics()
        metrics.access_count = 5
        metrics.last_access = datetime.now() - timedelta(hours=48)
        metrics.creation_time = datetime.now() - timedelta(days=7)
        metrics.decay_rate = 0.1
        
        mc.memory_metrics["python"] = metrics
        return mc

    def test_time_decay_strategy(self, mc_with_metrics):
        """测试时间衰减策略"""
        mc_with_metrics.forgetting_strategy = ForgettingStrategy.TIME_DECAY
        to_forget = mc_with_metrics.apply_forgetting(hours_elapsed=48)
        
        # 验证返回列表
        assert isinstance(to_forget, list)

    def test_frequency_strategy(self, mc_with_metrics):
        """测试频率策略"""
        mc_with_metrics.forgetting_strategy = ForgettingStrategy.FREQUENCY
        to_forget = mc_with_metrics.apply_forgetting()
        
        assert isinstance(to_forget, list)

    def test_lru_strategy(self, mc_with_metrics):
        """测试LRU策略"""
        mc_with_metrics.forgetting_strategy = ForgettingStrategy.LRU
        to_forget = mc_with_metrics.apply_forgetting()
        
        assert isinstance(to_forget, list)

    def test_hybrid_strategy(self, mc_with_metrics):
        """测试混合策略"""
        mc_with_metrics.forgetting_strategy = ForgettingStrategy.HYBRID
        to_forget = mc_with_metrics.apply_forgetting(hours_elapsed=24)
        
        assert isinstance(to_forget, list)


class TestMemoryConsolidation:
    """测试记忆合并"""

    def test_consolidate_similar_entities(self):
        """测试合并相似实体"""
        kg = KnowledgeGraph()
        
        # 添加相似实体 (注意KG会转小写,所以用不同名称但相似)
        entity1 = Entity(name="Python3", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        entity2 = Entity(name="Python2", entity_type=EntityType.TECHNOLOGY, confidence=0.9)
        
        kg.add_node(entity1)
        kg.add_node(entity2)
        
        mc = MemoryConsolidation(kg)
        initial_count = len(kg.graph.nodes())
        
        # 降低阈值以匹配"python3"和"python2"
        consolidated = mc.consolidate_similar_entities(similarity_threshold=0.7)
        
        assert consolidated > 0
        assert len(kg.graph.nodes()) < initial_count

    def test_no_consolidation_needed(self):
        """测试无需合并的情况"""
        kg = KnowledgeGraph()
        
        # 添加不相似实体
        entity1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        entity2 = Entity(name="Java", entity_type=EntityType.TECHNOLOGY, confidence=0.9)
        
        kg.add_node(entity1)
        kg.add_node(entity2)
        
        mc = MemoryConsolidation(kg)
        initial_count = len(kg.graph.nodes())
        
        consolidated = mc.consolidate_similar_entities(similarity_threshold=0.8)
        
        assert consolidated == 0
        assert len(kg.graph.nodes()) == initial_count


class TestMemoryCleanup:
    """测试记忆清理"""

    def test_cleanup_low_importance(self):
        """测试清理低重要性记忆"""
        kg = KnowledgeGraph()
        
        # 添加多个实体
        for i in range(5):
            entity = Entity(
                name=f"Entity{i}",
                entity_type=EntityType.CONCEPT,
                confidence=0.1 + i * 0.1,
            )
            kg.add_node(entity)
        
        mc = MemoryConsolidation(kg, cleanup_threshold=0.5)
        
        # 设置低重要性分数
        for node_id in kg.graph.nodes():
            metrics = MemoryMetrics()
            metrics.importance_score = 0.1
            mc.memory_metrics[node_id] = metrics
        
        initial_count = len(kg.graph.nodes())
        cleaned = mc.cleanup()
        
        assert cleaned > 0
        assert len(kg.graph.nodes()) < initial_count

    def test_cleanup_exceeds_max_size(self):
        """测试超过最大容量时清理"""
        kg = KnowledgeGraph()
        
        # 添加超过最大容量的实体
        max_size = 10
        for i in range(max_size + 5):
            entity = Entity(
                name=f"Entity{i}",
                entity_type=EntityType.CONCEPT,
                confidence=0.5,
            )
            kg.add_node(entity)
        
        mc = MemoryConsolidation(kg, max_memory_size=max_size)
        
        cleaned = mc.cleanup(force=True)
        assert cleaned >= 5

    def test_no_cleanup_needed(self):
        """测试无需清理的情况"""
        kg = KnowledgeGraph()
        
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(entity)
        
        mc = MemoryConsolidation(kg)
        
        # 设置高重要性分数
        metrics = MemoryMetrics()
        metrics.importance_score = 0.9
        mc.memory_metrics["python"] = metrics
        
        cleaned = mc.cleanup()
        assert cleaned == 0


class TestMemoryReinforcement:
    """测试记忆强化"""

    def test_reinforce_memory(self):
        """测试强化记忆"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(python)
        
        mc = MemoryConsolidation(kg)
        
        # 初始度量(明确设置初始值)
        initial_metrics = MemoryMetrics()
        initial_metrics.access_count = 0
        initial_metrics.importance_score = 0.5
        mc.memory_metrics["python"] = initial_metrics
        
        # 强化
        mc.reinforce_memory("python", boost=0.2)
        
        updated_metrics = mc.memory_metrics["python"]
        assert updated_metrics.access_count == 1  # 从0增加到1
        assert updated_metrics.reinforcement_count == 1
        assert updated_metrics.importance_score == 0.7  # 0.5 + 0.2

    def test_reinforce_nonexistent(self):
        """测试强化不存在的记忆"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        
        # 不应抛出异常
        mc.reinforce_memory("nonexistent")


class TestImportanceLevel:
    """测试重要性级别"""

    def test_critical_level(self):
        """测试关键级别"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        level = mc.get_importance_level(0.9)
        assert level == ImportanceLevel.CRITICAL

    def test_high_level(self):
        """测试高级别"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        level = mc.get_importance_level(0.7)
        assert level == ImportanceLevel.HIGH

    def test_medium_level(self):
        """测试中级别"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        level = mc.get_importance_level(0.5)
        assert level == ImportanceLevel.MEDIUM

    def test_low_level(self):
        """测试低级别"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        level = mc.get_importance_level(0.3)
        assert level == ImportanceLevel.LOW

    def test_trivial_level(self):
        """测试琐碎级别"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        level = mc.get_importance_level(0.1)
        assert level == ImportanceLevel.TRIVIAL


class TestMemoryReport:
    """测试记忆报告"""

    def test_memory_report_structure(self):
        """测试报告结构"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(python)
        
        mc = MemoryConsolidation(kg)
        report = mc.get_memory_report()
        
        assert "total_memories" in report
        assert "importance_distribution" in report
        assert "avg_importance" in report
        assert "stats" in report

    def test_empty_graph_report(self):
        """测试空图报告"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        report = mc.get_memory_report()
        
        assert report["total_memories"] == 0
        assert report["avg_importance"] == 0.0


class TestStatistics:
    """测试统计信息"""

    def test_statistics_tracking(self):
        """测试统计跟踪"""
        kg = KnowledgeGraph()
        mc = MemoryConsolidation(kg)
        
        stats = mc.get_statistics()
        assert "total_memories" in stats
        assert "consolidated_memories" in stats
        assert "forgotten_memories" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
