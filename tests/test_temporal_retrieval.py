"""
跨会话记忆系统 - 时序检索管理器单元测试
Cross-Session Memory System - Temporal Retrieval Unit Tests

版本: 1.0.0
日期: 2025-11-14

测试覆盖:
1. 初始化与参数验证
2. 时间衰减计算
3. 最近记忆检索
4. 时间窗口过滤
5. 时间加权检索
6. 时间范围查询
7. 时间分布统计
"""

import pytest
import os
import time
from datetime import datetime, timedelta
from temporal_retrieval import TemporalRetrieval, TemporalRetrievalError
from conversation_manager import ConversationManager, ConversationManagerError
from cross_session_migration import CrossSessionMigration


@pytest.fixture
def test_db(tmp_path):
    """创建临时测试数据库"""
    db_path = tmp_path / "test_temporal.db"

    # 运行迁移
    migration = CrossSessionMigration(str(db_path))
    migration.migrate_up()

    yield str(db_path)

    # 清理
    if os.path.exists(str(db_path)):
        os.remove(str(db_path))


@pytest.fixture
def conv_mgr(test_db):
    """创建ConversationManager实例"""
    return ConversationManager(db_path=test_db)


@pytest.fixture
def retrieval(conv_mgr):
    """创建TemporalRetrieval实例"""
    return TemporalRetrieval(conv_mgr, decay_factor=0.1, recency_weight=0.7)


class TestInitialization:
    """测试初始化"""

    def test_init_with_valid_params(self, conv_mgr):
        """测试正常初始化"""
        retrieval = TemporalRetrieval(conv_mgr)
        assert retrieval.conversation_mgr == conv_mgr
        assert retrieval.decay_factor == 0.1
        assert retrieval.recency_weight == 0.7

    def test_init_with_custom_params(self, conv_mgr):
        """测试自定义参数"""
        retrieval = TemporalRetrieval(
            conv_mgr, decay_factor=0.2, recency_weight=0.8
        )
        assert retrieval.decay_factor == 0.2
        assert retrieval.recency_weight == 0.8

    def test_init_without_conv_mgr(self):
        """测试缺少conversation_mgr"""
        with pytest.raises(ValueError, match="conversation_mgr is required"):
            TemporalRetrieval(None)

    def test_init_with_invalid_decay_factor(self, conv_mgr):
        """测试无效的衰减因子"""
        with pytest.raises(ValueError, match="decay_factor must be between 0 and 1"):
            TemporalRetrieval(conv_mgr, decay_factor=1.5)

    def test_init_with_invalid_recency_weight(self, conv_mgr):
        """测试无效的新近度权重"""
        with pytest.raises(ValueError, match="recency_weight must be between 0 and 1"):
            TemporalRetrieval(conv_mgr, recency_weight=-0.1)


class TestTimeDecay:
    """测试时间衰减计算"""

    def test_decay_recent_memory(self, retrieval):
        """测试最近记忆的衰减权重"""
        now = datetime.now()
        weight = retrieval._calculate_time_decay(now)
        assert 0.99 <= weight <= 1.0  # 非常新，权重接近1

    def test_decay_old_memory(self, retrieval):
        """测试旧记忆的衰减权重"""
        old_time = datetime.now() - timedelta(days=30)
        weight = retrieval._calculate_time_decay(old_time)
        assert weight < 0.1  # 30天前，权重很低

    def test_decay_progression(self, retrieval):
        """测试衰减递进性"""
        now = datetime.now()
        weight_now = retrieval._calculate_time_decay(now)
        weight_1day = retrieval._calculate_time_decay(now - timedelta(days=1))
        weight_7days = retrieval._calculate_time_decay(now - timedelta(days=7))

        assert weight_now > weight_1day > weight_7days


class TestRecentMemories:
    """测试最近记忆检索"""

    def test_get_recent_memories_empty(self, retrieval):
        """测试空记忆库"""
        recent = retrieval.get_recent_memories("user_empty", hours=24)
        assert recent == []

    def test_get_recent_memories_within_window(self, retrieval, conv_mgr):
        """测试时间窗口内的记忆"""
        # 记录最近的消息
        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_001",
            role="user",
            content="Recent message",
        )

        recent = retrieval.get_recent_memories("user_001", hours=1)
        assert len(recent) == 1
        assert recent[0]["content"] == "Recent message"

    def test_get_recent_memories_sorted(self, retrieval, conv_mgr):
        """测试记忆按时间倒序"""
        # 记录多条消息
        for i in range(3):
            conv_mgr.record_message(
                session_id="sess_001",
                user_id="user_002",
                role="user",
                content=f"Message {i}",
            )
            time.sleep(0.01)

        recent = retrieval.get_recent_memories("user_002", hours=24)
        assert len(recent) == 3
        # 最新的在前
        assert recent[0]["content"] == "Message 2"
        assert recent[2]["content"] == "Message 0"

    def test_get_recent_memories_with_session_filter(self, retrieval, conv_mgr):
        """测试会话过滤"""
        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_003",
            role="user",
            content="Session 1 message",
        )
        conv_mgr.record_message(
            session_id="sess_002",
            user_id="user_003",
            role="user",
            content="Session 2 message",
        )

        recent = retrieval.get_recent_memories(
            "user_003", hours=24, session_id="sess_001"
        )
        assert len(recent) == 1
        assert recent[0]["content"] == "Session 1 message"

    def test_get_recent_memories_invalid_hours(self, retrieval):
        """测试无效的时间窗口"""
        with pytest.raises(ValueError, match="hours must be positive"):
            retrieval.get_recent_memories("user_test", hours=-1)


class TestTemporalDecayApplication:
    """测试时间衰减应用"""

    def test_apply_temporal_decay_empty(self, retrieval):
        """测试空列表"""
        weighted = retrieval.apply_temporal_decay([])
        assert weighted == []

    def test_apply_temporal_decay_single(self, retrieval, conv_mgr):
        """测试单个记忆"""
        conv_mgr.record_message(
            session_id="sess_001", user_id="user_test", role="user", content="Test"
        )
        memories = conv_mgr.get_history("sess_001")

        weighted = retrieval.apply_temporal_decay(memories)
        assert len(weighted) == 1
        memory, weight = weighted[0]
        assert memory["content"] == "Test"
        assert 0 < weight <= 1.0

    def test_apply_temporal_decay_multiple(self, retrieval, conv_mgr):
        """测试多个记忆"""
        for i in range(3):
            conv_mgr.record_message(
                session_id="sess_001",
                user_id="user_test",
                role="user",
                content=f"Message {i}",
            )

        memories = conv_mgr.get_history("sess_001")
        weighted = retrieval.apply_temporal_decay(memories)

        assert len(weighted) == 3
        # 所有权重应在0-1之间
        for _, weight in weighted:
            assert 0 < weight <= 1.0


class TestTimeWeightedMemories:
    """测试时间加权记忆检索"""

    def test_get_time_weighted_memories_basic(self, retrieval, conv_mgr):
        """测试基本时间加权检索"""
        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_tw",
            role="user",
            content="Test message",
        )

        weighted = retrieval.get_time_weighted_memories("user_tw", hours=24)
        assert len(weighted) == 1
        memory, weight = weighted[0]
        assert memory["content"] == "Test message"
        assert 0 < weight <= 1.0

    def test_get_time_weighted_memories_with_query(self, retrieval, conv_mgr):
        """测试带查询的时间加权检索"""
        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_tw",
            role="user",
            content="Talk about AGI",
        )
        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_tw",
            role="user",
            content="Talk about weather",
        )

        weighted = retrieval.get_time_weighted_memories("user_tw", query="AGI")
        assert len(weighted) == 1
        assert "AGI" in weighted[0][0]["content"]

    def test_get_time_weighted_memories_limit(self, retrieval, conv_mgr):
        """测试数量限制"""
        for i in range(10):
            conv_mgr.record_message(
                session_id="sess_001",
                user_id="user_tw",
                role="user",
                content=f"Message {i}",
            )

        weighted = retrieval.get_time_weighted_memories("user_tw", hours=24, limit=5)
        assert len(weighted) == 5


class TestTimeRangeQuery:
    """测试时间范围查询"""

    def test_get_memories_by_time_range_valid(self, retrieval, conv_mgr):
        """测试有效时间范围查询"""
        now = datetime.now()
        start_time = now - timedelta(hours=2)
        end_time = now + timedelta(hours=1)

        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_range",
            role="user",
            content="Within range",
        )

        memories = retrieval.get_memories_by_time_range(
            "user_range", start_time, end_time
        )
        assert len(memories) >= 1

    def test_get_memories_by_time_range_invalid(self, retrieval):
        """测试无效时间范围"""
        now = datetime.now()
        start_time = now
        end_time = now - timedelta(hours=1)

        with pytest.raises(ValueError, match="start_time must be before end_time"):
            retrieval.get_memories_by_time_range("user_test", start_time, end_time)


class TestTimeDistribution:
    """测试时间分布统计"""

    def test_get_time_distribution_empty(self, retrieval):
        """测试空记忆库"""
        distribution = retrieval.get_time_distribution("user_empty")
        assert distribution["today"] == 0
        assert distribution["yesterday"] == 0

    def test_parse_timestamp_invalid(self, retrieval):
        """测试无效时间戳解析"""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            retrieval._parse_timestamp("invalid-timestamp")

    def test_get_time_distribution_with_messages(self, retrieval, conv_mgr):
        """测试有记忆的分布"""
        # 记录今天的消息
        for i in range(3):
            conv_mgr.record_message(
                session_id="sess_001",
                user_id="user_dist",
                role="user",
                content=f"Today message {i}",
            )

        distribution = retrieval.get_time_distribution("user_dist")
        assert distribution["today"] == 3

    def test_get_time_distribution_with_session_filter(self, retrieval, conv_mgr):
        """测试会话过滤的时间分布"""
        conv_mgr.record_message(
            session_id="sess_001",
            user_id="user_dist",
            role="user",
            content="Session 1",
        )
        conv_mgr.record_message(
            session_id="sess_002",
            user_id="user_dist",
            role="user",
            content="Session 2",
        )

        distribution = retrieval.get_time_distribution("user_dist", session_id="sess_001")
        assert distribution["today"] == 1


class TestErrorHandling:
    """测试错误处理"""

    def test_get_all_user_messages_db_error(self, retrieval):
        """测试数据库连接错误"""
        # 使用不存在的数据库路径触发错误
        old_path = retrieval.conversation_mgr.db_path
        retrieval.conversation_mgr.db_path = "/nonexistent/path/to/db.sqlite"
        
        with pytest.raises(ConversationManagerError):
            retrieval._get_all_user_messages("user_test")
        
        # 恢复路径
        retrieval.conversation_mgr.db_path = old_path

    def test_get_recent_memories_db_error(self, retrieval):
        """测试get_recent_memories数据库错误"""
        old_path = retrieval.conversation_mgr.db_path
        retrieval.conversation_mgr.db_path = "/invalid/db/path.sqlite"
        
        with pytest.raises(TemporalRetrievalError, match="Failed to get recent memories"):
            retrieval.get_recent_memories("user_test", hours=24)
        
        retrieval.conversation_mgr.db_path = old_path

    def test_apply_temporal_decay_skip_invalid(self, retrieval):
        """测试apply_temporal_decay跳过无效时间戳"""
        # 传入无效时间戳的消息(应该被跳过)
        invalid_messages = [
            {"message_id": "msg1", "timestamp": "invalid-timestamp", "content": "test"}
        ]
        
        # 不应抛出异常,而是返回空列表
        result = retrieval.apply_temporal_decay(invalid_messages)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=temporal_retrieval", "--cov-report=term"])
