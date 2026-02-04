"""
跨会话记忆系统 - 相关性排序器测试
Cross-Session Memory System - Relevance Ranker Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 2 - 智能检索
"""

import pytest
import math
import logging
from datetime import datetime, timedelta
from relevance_ranker import (
    RelevanceRanker,
    RankingWeights,
    RelevanceRankerError,
)

logging.disable(logging.CRITICAL)


class TestRankingWeights:
    """测试排序权重配置"""

    def test_default_weights(self):
        """测试默认权重"""
        weights = RankingWeights()
        assert weights.similarity_weight == 0.5
        assert weights.temporal_weight == 0.3
        assert weights.importance_weight == 0.2
        assert math.isclose(
            weights.similarity_weight
            + weights.temporal_weight
            + weights.importance_weight,
            1.0,
        )

    def test_custom_weights(self):
        """测试自定义权重"""
        weights = RankingWeights(
            similarity_weight=0.6, temporal_weight=0.2, importance_weight=0.2
        )
        assert weights.similarity_weight == 0.6
        assert weights.temporal_weight == 0.2
        assert weights.importance_weight == 0.2

    def test_invalid_weights_sum(self):
        """测试权重总和不为1.0"""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RankingWeights(
                similarity_weight=0.5, temporal_weight=0.3, importance_weight=0.3
            )

    def test_negative_weights(self):
        """测试负权重"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            RankingWeights(
                similarity_weight=-0.1, temporal_weight=0.5, importance_weight=0.6
            )

    def test_weights_exceed_one(self):
        """测试权重超过1.0"""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            RankingWeights(
                similarity_weight=1.5, temporal_weight=0.0, importance_weight=-0.5
            )


class TestInitialization:
    """测试初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        ranker = RelevanceRanker()
        assert ranker.weights.similarity_weight == 0.5
        assert ranker.weights.temporal_weight == 0.3
        assert ranker.weights.importance_weight == 0.2
        assert len(ranker.stop_words) > 0
        assert "the" in ranker.stop_words

    def test_custom_weights_initialization(self):
        """测试自定义权重初始化"""
        weights = RankingWeights(
            similarity_weight=0.7, temporal_weight=0.2, importance_weight=0.1
        )
        ranker = RelevanceRanker(weights=weights)
        assert ranker.weights.similarity_weight == 0.7

    def test_custom_stop_words(self):
        """测试自定义停用词"""
        custom_stops = {"hello", "world"}
        ranker = RelevanceRanker(stop_words=custom_stops)
        assert ranker.stop_words == custom_stops


class TestTokenization:
    """测试分词"""

    def test_basic_tokenization(self):
        """测试基本分词"""
        ranker = RelevanceRanker()
        text = "Python is a great programming language"
        tokens = ranker._tokenize(text)
        assert "python" in tokens
        assert "great" in tokens
        assert "programming" in tokens
        assert "language" in tokens
        # 停用词应被过滤
        assert "is" not in tokens
        assert "a" not in tokens

    def test_punctuation_removal(self):
        """测试标点符号移除"""
        ranker = RelevanceRanker()
        text = "Hello, World! How are you?"
        tokens = ranker._tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens

    def test_short_words_filter(self):
        """测试短词过滤"""
        ranker = RelevanceRanker()
        text = "I am a Python developer"
        tokens = ranker._tokenize(text)
        # 1-2字母的词应被过滤
        assert "i" not in tokens
        assert "am" not in tokens
        assert "python" in tokens
        assert "developer" in tokens

    def test_empty_text(self):
        """测试空文本"""
        ranker = RelevanceRanker()
        tokens = ranker._tokenize("")
        assert tokens == []


class TestTFCalculation:
    """测试TF计算"""

    def test_basic_tf(self):
        """测试基本TF计算"""
        ranker = RelevanceRanker()
        tokens = ["python", "programming", "python", "language"]
        tf = ranker._calculate_tf(tokens)
        assert tf["python"] == 0.5  # 2/4
        assert tf["programming"] == 0.25  # 1/4
        assert tf["language"] == 0.25

    def test_empty_tokens(self):
        """测试空tokens"""
        ranker = RelevanceRanker()
        tf = ranker._calculate_tf([])
        assert tf == {}

    def test_single_token(self):
        """测试单个token"""
        ranker = RelevanceRanker()
        tf = ranker._calculate_tf(["hello"])
        assert tf["hello"] == 1.0


class TestIDFCalculation:
    """测试IDF计算"""

    def test_basic_idf(self):
        """测试基本IDF计算"""
        ranker = RelevanceRanker()
        documents = [
            ["python", "programming"],
            ["python", "language"],
            ["java", "programming"],
        ]
        idf = ranker._calculate_idf(documents)

        # python出现在2个文档中
        assert idf["python"] == pytest.approx(math.log(4 / 3) + 1.0)
        # programming出现在2个文档中
        assert idf["programming"] == pytest.approx(math.log(4 / 3) + 1.0)
        # java和language各出现在1个文档中
        assert idf["java"] == pytest.approx(math.log(4 / 2) + 1.0)
        assert idf["language"] == pytest.approx(math.log(4 / 2) + 1.0)

    def test_empty_documents(self):
        """测试空文档列表"""
        ranker = RelevanceRanker()
        idf = ranker._calculate_idf([])
        assert idf == {}


class TestTFIDFCalculation:
    """测试TF-IDF计算"""

    def test_basic_tfidf(self):
        """测试基本TF-IDF计算"""
        ranker = RelevanceRanker()
        tf = {"python": 0.5, "programming": 0.25}
        idf = {"python": 2.0, "programming": 1.5}
        tfidf = ranker._calculate_tfidf(tf, idf)

        assert tfidf["python"] == 1.0  # 0.5 * 2.0
        assert tfidf["programming"] == 0.375  # 0.25 * 1.5

    def test_missing_idf(self):
        """测试IDF缺失时使用默认值1.0"""
        ranker = RelevanceRanker()
        tf = {"python": 0.5, "unknown": 0.25}
        idf = {"python": 2.0}
        tfidf = ranker._calculate_tfidf(tf, idf)

        assert tfidf["python"] == 1.0
        assert tfidf["unknown"] == 0.25  # 0.25 * 1.0(default)


class TestCosineSimilarity:
    """测试余弦相似度"""

    def test_identical_vectors(self):
        """测试相同向量"""
        ranker = RelevanceRanker()
        vec = {"python": 0.5, "programming": 0.3}
        similarity = ranker._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """测试正交向量"""
        ranker = RelevanceRanker()
        vec1 = {"python": 1.0}
        vec2 = {"java": 1.0}
        similarity = ranker._cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_partial_overlap(self):
        """测试部分重叠向量"""
        ranker = RelevanceRanker()
        vec1 = {"python": 0.5, "programming": 0.5}
        vec2 = {"python": 0.5, "language": 0.5}
        similarity = ranker._cosine_similarity(vec1, vec2)
        assert 0.0 < similarity < 1.0

    def test_empty_vectors(self):
        """测试空向量"""
        ranker = RelevanceRanker()
        assert ranker._cosine_similarity({}, {"python": 1.0}) == 0.0
        assert ranker._cosine_similarity({"python": 1.0}, {}) == 0.0


class TestTextSimilarity:
    """测试文本相似度计算"""

    def test_basic_similarity(self):
        """测试基本相似度计算"""
        ranker = RelevanceRanker()
        memories = [
            {"content": "Python programming tutorial"},
            {"content": "Java enterprise application"},
        ]
        query = "Python programming"
        results = ranker.calculate_text_similarity(query, memories)

        assert len(results) == 2
        assert results[0][1] > results[1][1]  # Python记忆应该相似度更高

    def test_identical_content(self):
        """测试相同内容"""
        ranker = RelevanceRanker()
        content = "Python programming tutorial"
        memories = [{"content": content}]
        results = ranker.calculate_text_similarity(content, memories)

        assert results[0][1] == pytest.approx(1.0)

    def test_empty_query(self):
        """测试空查询"""
        ranker = RelevanceRanker()
        memories = [{"content": "test"}]
        with pytest.raises(ValueError, match="query cannot be empty"):
            ranker.calculate_text_similarity("", memories)

    def test_no_common_words(self):
        """测试无共同词汇"""
        ranker = RelevanceRanker()
        memories = [{"content": "Python programming"}]
        query = "Java development"
        results = ranker.calculate_text_similarity(query, memories)

        assert results[0][1] == 0.0

    def test_query_with_only_stopwords(self):
        """测试仅含停用词的查询"""
        ranker = RelevanceRanker()
        memories = [{"content": "Python programming"}]
        query = "the a is"
        results = ranker.calculate_text_similarity(query, memories)

        # 所有记忆应返回0.0相似度
        for memory, score in results:
            assert score == 0.0

    def test_error_handling(self):
        """测试错误处理"""
        ranker = RelevanceRanker()
        # 模拟内部错误(传入None作为memories)
        with pytest.raises(RelevanceRankerError):
            ranker.calculate_text_similarity("test", None)


class TestTemporalScore:
    """测试时间分数计算"""

    def test_recent_timestamp(self):
        """测试最近的时间戳"""
        ranker = RelevanceRanker()
        now = datetime.now()
        recent = (now - timedelta(hours=1)).isoformat()
        score = ranker.calculate_temporal_score(recent)
        assert score > 0.95

    def test_old_timestamp(self):
        """测试较旧的时间戳"""
        ranker = RelevanceRanker()
        now = datetime.now()
        old = (now - timedelta(days=30)).isoformat()
        score = ranker.calculate_temporal_score(old)
        assert score < 0.5

    def test_custom_decay_factor(self):
        """测试自定义衰减因子"""
        ranker = RelevanceRanker()
        now = datetime.now()
        timestamp = (now - timedelta(days=10)).isoformat()

        score1 = ranker.calculate_temporal_score(timestamp, decay_factor=0.1)
        score2 = ranker.calculate_temporal_score(timestamp, decay_factor=0.5)
        assert score1 > score2  # 衰减因子越小,分数越高

    def test_invalid_timestamp(self):
        """测试无效时间戳"""
        ranker = RelevanceRanker()
        with pytest.raises(ValueError, match="Invalid timestamp"):
            ranker.calculate_temporal_score("invalid-timestamp")


class TestImportanceScore:
    """测试重要性分数计算"""

    def test_explicit_importance(self):
        """测试显式重要性"""
        ranker = RelevanceRanker()
        memory = {"metadata": {"importance": 0.9}}
        score = ranker.calculate_importance_score(memory)
        assert score == 0.9

    def test_default_importance(self):
        """测试默认重要性"""
        ranker = RelevanceRanker()
        memory = {"metadata": {}}
        score = ranker.calculate_importance_score(memory)
        assert score == 0.5

    def test_no_metadata(self):
        """测试无metadata"""
        ranker = RelevanceRanker()
        memory = {}
        score = ranker.calculate_importance_score(memory)
        assert score == 0.5

    def test_json_string_metadata(self):
        """测试JSON字符串metadata"""
        ranker = RelevanceRanker()
        import json

        memory = {"metadata": json.dumps({"importance": 0.8})}
        score = ranker.calculate_importance_score(memory)
        assert score == 0.8

    def test_double_encoded_metadata(self):
        """测试双重编码metadata"""
        ranker = RelevanceRanker()
        import json

        memory = {"metadata": json.dumps(json.dumps({"importance": 0.7}))}
        score = ranker.calculate_importance_score(memory)
        assert score == 0.7

    def test_invalid_importance_type(self):
        """测试无效importance类型"""
        ranker = RelevanceRanker()
        memory = {"metadata": {"importance": "high"}}
        score = ranker.calculate_importance_score(memory)
        assert score == 0.5  # 应使用默认值

    def test_out_of_range_importance(self):
        """测试超出范围的importance"""
        ranker = RelevanceRanker()
        memory1 = {"metadata": {"importance": 1.5}}
        memory2 = {"metadata": {"importance": -0.5}}

        score1 = ranker.calculate_importance_score(memory1)
        score2 = ranker.calculate_importance_score(memory2)

        assert score1 == 1.0  # 应被限制在1.0
        assert score2 == 0.0  # 应被限制在0.0


class TestCompositeScore:
    """测试综合评分"""

    def test_balanced_factors(self):
        """测试平衡因子"""
        ranker = RelevanceRanker()
        memory = {}
        score = ranker.calculate_composite_score(memory, 0.5, 0.5, 0.5)
        assert score == 0.5

    def test_high_similarity(self):
        """测试高相似度"""
        weights = RankingWeights(
            similarity_weight=1.0, temporal_weight=0.0, importance_weight=0.0
        )
        ranker = RelevanceRanker(weights=weights)
        memory = {}
        score = ranker.calculate_composite_score(memory, 1.0, 0.0, 0.0)
        assert score == 1.0

    def test_weighted_combination(self):
        """测试加权组合"""
        weights = RankingWeights(
            similarity_weight=0.5, temporal_weight=0.3, importance_weight=0.2
        )
        ranker = RelevanceRanker(weights=weights)
        memory = {}
        score = ranker.calculate_composite_score(memory, 1.0, 0.5, 0.8)
        expected = 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.8
        assert score == pytest.approx(expected)


class TestRankMemories:
    """测试记忆排序"""

    def test_basic_ranking(self):
        """测试基本排序"""
        ranker = RelevanceRanker()
        now = datetime.now()
        memories = [
            {
                "content": "Python programming",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "metadata": {"importance": 0.9},
            },
            {
                "content": "Java development",
                "timestamp": (now - timedelta(days=7)).isoformat(),
                "metadata": {"importance": 0.5},
            },
        ]
        query = "Python tutorial"
        ranked = ranker.rank_memories(memories, query)

        assert len(ranked) == 2
        # Python记忆应排在前面
        assert "Python" in ranked[0][0]["content"]
        # 分数应递减
        assert ranked[0][1] >= ranked[1][1]

    def test_top_k_limit(self):
        """测试Top-K限制"""
        ranker = RelevanceRanker()
        now = datetime.now()
        memories = [
            {
                "content": f"Content {i}",
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "metadata": {"importance": 0.5},
            }
            for i in range(10)
        ]
        ranked = ranker.rank_memories(memories, "test", top_k=3)
        assert len(ranked) == 3

    def test_empty_query_ranking(self):
        """测试空查询排序(仅基于时间和重要性)"""
        ranker = RelevanceRanker()
        now = datetime.now()
        memories = [
            {
                "content": "Old but important",
                "timestamp": (now - timedelta(days=30)).isoformat(),
                "metadata": {"importance": 0.9},
            },
            {
                "content": "Recent but less important",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "metadata": {"importance": 0.1},
            },
        ]
        ranked = ranker.rank_memories(memories, query="")

        # 排序应主要基于时间和重要性
        assert len(ranked) == 2

    def test_empty_memories(self):
        """测试空记忆列表"""
        ranker = RelevanceRanker()
        ranked = ranker.rank_memories([], "test")
        assert ranked == []

    def test_missing_timestamp(self):
        """测试缺失timestamp"""
        ranker = RelevanceRanker()
        memories = [{"content": "Test", "metadata": {"importance": 0.5}}]
        ranked = ranker.rank_memories(memories, "test")
        assert len(ranked) == 1


class TestIntegrationScenarios:
    """测试集成场景"""

    def test_realistic_scenario(self):
        """测试真实场景"""
        weights = RankingWeights(
            similarity_weight=0.4, temporal_weight=0.4, importance_weight=0.2
        )
        ranker = RelevanceRanker(weights=weights)

        now = datetime.now()
        memories = [
            {
                "content": "Python machine learning tutorial with scikit-learn and pandas",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "metadata": {"importance": 0.9},
            },
            {
                "content": "Introduction to Python programming basics",
                "timestamp": (now - timedelta(days=1)).isoformat(),
                "metadata": {"importance": 0.7},
            },
            {
                "content": "Java Spring Boot REST API development",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "metadata": {"importance": 0.8},
            },
            {
                "content": "Machine learning algorithms and neural networks",
                "timestamp": (now - timedelta(days=3)).isoformat(),
                "metadata": {"importance": 0.6},
            },
        ]

        query = "Python machine learning"
        ranked = ranker.rank_memories(memories, query, top_k=2)

        # 第一个应该是Python ML教程(高相似度+高重要性+较新)
        assert "machine learning" in ranked[0][0]["content"].lower()
        assert "python" in ranked[0][0]["content"].lower()

    def test_time_importance_tradeoff(self):
        """测试时间与重要性权衡"""
        # 设置更高的重要性权重
        weights = RankingWeights(
            similarity_weight=0.2, temporal_weight=0.2, importance_weight=0.6
        )
        ranker = RelevanceRanker(weights=weights)

        now = datetime.now()
        memories = [
            {
                "content": "Very important old memory",
                "timestamp": (now - timedelta(days=30)).isoformat(),
                "metadata": {"importance": 1.0},
            },
            {
                "content": "Less important recent memory",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "metadata": {"importance": 0.1},
            },
        ]

        ranked = ranker.rank_memories(memories, "memory")
        # 高重要性应胜过时间新鲜度
        assert ranked[0][0]["metadata"]["importance"] == 1.0


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
