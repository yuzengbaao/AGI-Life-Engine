"""
跨会话记忆系统 - Phase 2 集成测试
Cross-Session Memory System - Phase 2 Integration Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 2 - 智能检索集成测试

测试三大组件的端到端集成:
1. TemporalRetrieval (时序检索)
2. ContextFilter (上下文过滤)
3. RelevanceRanker (相关性排序)
"""

import pytest
import os
import logging
from datetime import datetime, timedelta
from temporal_retrieval import TemporalRetrieval, TemporalRetrievalError
from context_filter import ContextFilter, FilterCondition, CompositeFilter
from relevance_ranker import RelevanceRanker, RankingWeights
from conversation_manager import ConversationManager
from cross_session_migration import CrossSessionMigration

logging.disable(logging.CRITICAL)


@pytest.fixture
def test_db(tmp_path):
    """创建临时测试数据库"""
    db_path = tmp_path / "test_integration.db"
    migration = CrossSessionMigration(str(db_path))
    migration.migrate_up()

    # 使用ConversationManager添加消息
    manager = ConversationManager(str(db_path))
    now = datetime.now()

    import json

    # 构建测试消息
    messages = [
        # Python相关(最近,重要)
        {
            "content": "How to use Python decorators for caching?",
            "role": "user",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "importance": 0.9,
            "tags": ["python", "decorators", "caching"],
        },
        {
            "content": "Python decorators wrap functions to add functionality.",
            "role": "assistant",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "importance": 0.9,
            "tags": ["python", "decorators"],
        },
        # 机器学习相关(较新,重要)
        {
            "content": "Explain machine learning model evaluation metrics",
            "role": "user",
            "timestamp": (now - timedelta(hours=3)).isoformat(),
            "importance": 0.8,
            "tags": ["ml", "evaluation", "metrics"],
        },
        {
            "content": "Key metrics include accuracy, precision, recall, and F1-score.",
            "role": "assistant",
            "timestamp": (now - timedelta(hours=3)).isoformat(),
            "importance": 0.8,
            "tags": ["ml", "metrics"],
        },
        # JavaScript相关(中等时间)
        {
            "content": "What are JavaScript promises?",
            "role": "user",
            "timestamp": (now - timedelta(days=1)).isoformat(),
            "importance": 0.6,
            "tags": ["javascript", "async"],
        },
        {
            "content": "Promises handle asynchronous operations in JavaScript.",
            "role": "assistant",
            "timestamp": (now - timedelta(days=1)).isoformat(),
            "importance": 0.6,
            "tags": ["javascript", "promises"],
        },
        # Python旧内容(较旧,中等重要)
        {
            "content": "How to read files in Python?",
            "role": "user",
            "timestamp": (now - timedelta(days=7)).isoformat(),
            "importance": 0.5,
            "tags": ["python", "files"],
        },
        {
            "content": "Use open() function with context manager.",
            "role": "assistant",
            "timestamp": (now - timedelta(days=7)).isoformat(),
            "importance": 0.5,
            "tags": ["python", "io"],
        },
        # 数据库相关(很旧,低重要性)
        {
            "content": "SQL JOIN types explained",
            "role": "user",
            "timestamp": (now - timedelta(days=30)).isoformat(),
            "importance": 0.4,
            "tags": ["sql", "database"],
        },
        {
            "content": "INNER, LEFT, RIGHT, and FULL OUTER joins.",
            "role": "assistant",
            "timestamp": (now - timedelta(days=30)).isoformat(),
            "importance": 0.4,
            "tags": ["sql", "joins"],
        },
    ]

    # 使用ConversationManager的record_message添加数据
    session_id = "test_session_1"
    for msg in messages:
        metadata_dict = {
            "importance": msg["importance"],
            "tags": msg["tags"],
            "timestamp": msg["timestamp"],  # 保存原始时间戳
        }
        # 记录消息
        manager.record_message(
            session_id=session_id,
            user_id="user1",
            role=msg["role"],
            content=msg["content"],
            metadata=metadata_dict,
        )

    yield str(db_path)

    # 清理
    if os.path.exists(str(db_path)):
        os.remove(str(db_path))


@pytest.fixture
def temporal_retrieval(test_db):
    """创建时序检索实例"""
    manager = ConversationManager(test_db)
    return TemporalRetrieval(manager)


@pytest.fixture
def context_filter(test_db):
    """创建上下文过滤实例"""
    manager = ConversationManager(test_db)
    return ContextFilter(manager)


@pytest.fixture
def relevance_ranker():
    """创建相关性排序实例"""
    return RelevanceRanker()


class TestBasicIntegration:
    """测试基本集成功能"""

    def test_temporal_to_filter_integration(self, temporal_retrieval, context_filter):
        """测试时序检索+上下文过滤集成"""
        # Step 1: 获取最近记忆
        recent = temporal_retrieval.get_recent_memories("user1", limit=5)
        assert len(recent) > 0

        # Step 2: 从最近记忆中过滤Python相关
        python_msgs = context_filter.filter_by_tags(recent, ["python"], match_any=True)
        assert len(python_msgs) > 0

        # 验证结果包含Python内容
        for msg in python_msgs:
            content = msg.get("content", "").lower()
            assert "python" in content or "decorator" in content

    def test_temporal_to_ranker_integration(self, temporal_retrieval, relevance_ranker):
        """测试时序检索+相关性排序集成"""
        # Step 1: 获取最近记忆
        recent = temporal_retrieval.get_recent_memories("user1", limit=10)

        # Step 2: 对这些记忆进行相关性排序
        query = "Python programming best practices"
        ranked = relevance_ranker.rank_memories(recent, query, top_k=3)

        assert len(ranked) > 0
        # 第一个结果应该是Python相关且分数最高
        top_memory, top_score = ranked[0]
        assert "python" in top_memory.get("content", "").lower()
        assert top_score > 0

    def test_filter_to_ranker_integration(self, context_filter, relevance_ranker):
        """测试上下文过滤+相关性排序集成"""
        # Step 1: 过滤用户消息
        user_msgs = context_filter.filter_by_role("user1", "user", limit=None)

        # Step 2: 排序这些用户消息
        query = "machine learning"
        ranked = relevance_ranker.rank_memories(user_msgs, query, top_k=5)

        assert len(ranked) > 0
        # ML相关消息应该排在前面
        top_memory, _ = ranked[0]
        assert "machine learning" in top_memory.get("content", "").lower()


class TestFullPipeline:
    """测试完整检索流程"""

    def test_temporal_filter_rank_pipeline(
        self, temporal_retrieval, context_filter, relevance_ranker
    ):
        """测试完整的 时序→过滤→排序 流程"""
        # Step 1: 获取时间加权的记忆
        time_weighted = temporal_retrieval.get_time_weighted_memories(
            "user1", limit=20
        )
        assert len(time_weighted) > 0

        # Step 2: 过滤出Python相关的
        python_msgs = context_filter.filter_by_tags(
            time_weighted, ["python"], match_any=True
        )
        assert len(python_msgs) > 0

        # Step 3: 按相关性排序
        query = "Python caching techniques"
        ranked = relevance_ranker.rank_memories(python_msgs, query, top_k=3)

        assert len(ranked) > 0
        # 验证排序结果
        for memory, score in ranked:
            assert "python" in memory.get("content", "").lower()
            assert score > 0

        # 分数应该递减
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_complex_query_scenario(
        self, temporal_retrieval, context_filter, relevance_ranker
    ):
        """测试复杂查询场景"""
        # 场景: 查找最近一周内关于Python的用户提问
        now = datetime.now()
        start_time = now - timedelta(days=7)

        # Step 1: 时间窗口过滤
        recent_week = temporal_retrieval.get_memories_by_time_range(
            "user1", start_time.isoformat(), now.isoformat()
        )

        # Step 2: 过滤用户角色+Python标签
        user_python = context_filter.filter_by_time_and_user(
            start_time, now, "user1", role="user"
        )
        python_only = [
            msg
            for msg in user_python
            if "python" in msg.get("content", "").lower()
        ]

        # Step 3: 按相关性排序
        query = "Python advanced features"
        ranked = relevance_ranker.rank_memories(python_only, query)

        # 验证结果
        assert len(ranked) > 0
        for memory, _ in ranked:
            assert memory.get("role") == "user"
            assert "python" in memory.get("content", "").lower()

    def test_multi_tag_composite_filter_pipeline(
        self, context_filter, relevance_ranker
    ):
        """测试多标签复合过滤+排序"""
        # 创建复合过滤条件: (python OR ml) AND NOT javascript
        cond_python = FilterCondition("tags", "python", operator="contains")
        cond_ml = FilterCondition("tags", "ml", operator="contains")
        cond_js = FilterCondition("tags", "javascript", operator="contains")

        # (python OR ml)
        comp_or = CompositeFilter([cond_python, cond_ml], logic="OR")
        # NOT javascript
        cond_not_js = cond_js.negate()

        # 组合
        comp_final = CompositeFilter([comp_or, cond_not_js], logic="AND")

        # 应用过滤
        all_msgs = context_filter.manager.get_history("user1", limit=None)
        filtered = context_filter.filter_by_composite(all_msgs, comp_final)

        # 排序
        query = "programming best practices"
        ranked = relevance_ranker.rank_memories(filtered, query)

        # 验证: 应该有Python或ML,但没有JavaScript
        for memory, _ in ranked:
            content = memory.get("content", "").lower()
            assert "python" in content or "machine learning" in content
            assert "javascript" not in content


class TestEdgeCases:
    """测试边界条件"""

    def test_empty_results_handling(
        self, temporal_retrieval, context_filter, relevance_ranker
    ):
        """测试空结果处理"""
        # 过滤不存在的标签
        empty = context_filter.filter_by_tags([], ["nonexistent"], match_any=True)
        assert empty == []

        # 对空列表排序
        ranked = relevance_ranker.rank_memories([], "test query")
        assert ranked == []

    def test_single_result_pipeline(
        self, temporal_retrieval, context_filter, relevance_ranker
    ):
        """测试单结果流程"""
        # 获取最新的1条记忆
        recent = temporal_retrieval.get_recent_memories("user1", limit=1)
        assert len(recent) == 1

        # 排序
        ranked = relevance_ranker.rank_memories(recent, "test", top_k=1)
        assert len(ranked) == 1

    def test_large_batch_processing(self, temporal_retrieval, relevance_ranker):
        """测试大批量处理"""
        # 获取所有记忆
        all_memories = temporal_retrieval.get_recent_memories("user1", limit=100)

        # 批量排序
        ranked = relevance_ranker.rank_memories(
            all_memories, "programming tutorial"
        )

        assert len(ranked) == len(all_memories)
        # 验证排序正确性
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)


class TestWeightConfiguration:
    """测试权重配置对结果的影响"""

    def test_different_weight_configs(self, temporal_retrieval, relevance_ranker):
        """测试不同权重配置的影响"""
        memories = temporal_retrieval.get_recent_memories("user1", limit=10)
        query = "Python programming"

        # 配置1: 重相似度
        weights1 = RankingWeights(
            similarity_weight=0.8, temporal_weight=0.1, importance_weight=0.1
        )
        ranker1 = RelevanceRanker(weights=weights1)
        ranked1 = ranker1.rank_memories(memories, query, top_k=3)

        # 配置2: 重时间
        weights2 = RankingWeights(
            similarity_weight=0.2, temporal_weight=0.7, importance_weight=0.1
        )
        ranker2 = RelevanceRanker(weights=weights2)
        ranked2 = ranker2.rank_memories(memories, query, top_k=3)

        # 配置3: 重重要性
        weights3 = RankingWeights(
            similarity_weight=0.2, temporal_weight=0.1, importance_weight=0.7
        )
        ranker3 = RelevanceRanker(weights=weights3)
        ranked3 = ranker3.rank_memories(memories, query, top_k=3)

        # 不同配置应该产生不同的排序
        top1 = ranked1[0][0].get("content")
        top2 = ranked2[0][0].get("content")
        top3 = ranked3[0][0].get("content")

        # 至少有一个配置的top结果不同
        assert top1 != top2 or top2 != top3 or top1 != top3

    def test_temporal_weight_impact(self, temporal_retrieval):
        """测试时间权重的影响"""
        # 比较不同衰减因子的效果
        memories1 = temporal_retrieval.get_time_weighted_memories(
            "user1", decay_factor=0.1, limit=10
        )
        memories2 = temporal_retrieval.get_time_weighted_memories(
            "user1", decay_factor=0.5, limit=10
        )

        # 更大的衰减因子应该让旧记忆的权重更低
        # (但结果数量应该相同)
        assert len(memories1) == len(memories2)


class TestRobustness:
    """测试健壮性"""

    def test_mixed_quality_data(self, context_filter, relevance_ranker):
        """测试混合质量数据"""
        # 获取所有消息(包含各种质量的数据)
        all_msgs = context_filter.manager.get_history("user1", limit=None)

        # 应该能处理缺失字段
        ranked = relevance_ranker.rank_memories(all_msgs, "test query")
        assert len(ranked) > 0

    def test_unicode_content(self, relevance_ranker):
        """测试Unicode内容"""
        memories = [
            {
                "content": "Python编程最佳实践",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"importance": 0.8},
            },
            {
                "content": "机器学习算法详解",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"importance": 0.7},
            },
        ]

        # 中文查询
        ranked = relevance_ranker.rank_memories(memories, "Python编程")
        assert len(ranked) == 2

    def test_special_characters(self, relevance_ranker):
        """测试特殊字符"""
        memories = [
            {
                "content": "Use @decorator for caching in Python!",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"importance": 0.8},
            },
            {
                "content": "JavaScript: promises & async/await",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"importance": 0.7},
            },
        ]

        ranked = relevance_ranker.rank_memories(memories, "@decorator caching")
        assert len(ranked) == 2


class TestPerformanceConsiderations:
    """测试性能考虑"""

    def test_top_k_efficiency(self, temporal_retrieval, relevance_ranker):
        """测试Top-K的效率优势"""
        import time

        memories = temporal_retrieval.get_recent_memories("user1", limit=100)

        # 测试Top-K
        start = time.time()
        ranked_k = relevance_ranker.rank_memories(memories, "test", top_k=5)
        time_k = time.time() - start

        # 测试全量排序
        start = time.time()
        ranked_all = relevance_ranker.rank_memories(memories, "test")
        time_all = time.time() - start

        # Top-K应该至少和全量排序一样快(可能更快)
        assert time_k <= time_all * 1.5  # 允许50%的误差范围

        # Top-K只返回K个结果
        assert len(ranked_k) == min(5, len(memories))
        assert len(ranked_all) == len(memories)

    def test_decay_factor_performance(self, temporal_retrieval):
        """测试衰减因子不影响性能"""
        import time

        # 测试不同衰减因子的性能
        factors = [0.1, 0.5, 1.0]
        times = []

        for factor in factors:
            start = time.time()
            temporal_retrieval.get_time_weighted_memories(
                "user1", decay_factor=factor, limit=50
            )
            times.append(time.time() - start)

        # 性能差异应该很小(<50%)
        assert max(times) < min(times) * 1.5


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
