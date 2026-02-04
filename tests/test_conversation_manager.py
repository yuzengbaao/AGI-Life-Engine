"""
跨会话记忆系统 - 对话管理器单元测试
Cross-Session Memory System - Conversation Manager Unit Tests

版本: 1.0.0
日期: 2025-11-14

测试覆盖:
1. 基本消息记录
2. 对话历史检索
3. FTS5全文检索
4. 缓存机制
5. 异常处理
6. 边界条件
"""

import pytest
import os
import time
from datetime import datetime
from conversation_manager import (
    ConversationManager,
    ConversationError,
    MessageNotFoundError,
    ConversationManagerError,
)
from cross_session_migration import CrossSessionMigration


@pytest.fixture
def test_db(tmp_path):
    """创建临时测试数据库"""
    db_path = tmp_path / "test_conversations.db"

    # 运行迁移
    migration = CrossSessionMigration(str(db_path))
    migration.migrate_up()

    yield str(db_path)

    # 清理
    if os.path.exists(str(db_path)):
        os.remove(str(db_path))


@pytest.fixture
def manager(test_db):
    """创建ConversationManager实例"""
    return ConversationManager(db_path=test_db, cache_ttl=2)


class TestConversationManagerInit:
    """测试初始化"""

    def test_init_with_valid_db(self, test_db):
        """测试正常初始化"""
        manager = ConversationManager(db_path=test_db)
        assert manager.db_path == test_db
        assert manager.cache_ttl == 900

    def test_init_with_custom_ttl(self, test_db):
        """测试自定义TTL"""
        manager = ConversationManager(db_path=test_db, cache_ttl=1800)
        assert manager.cache_ttl == 1800

    def test_init_with_invalid_db(self):
        """测试无效数据库路径"""
        with pytest.raises(ConversationManagerError):
            ConversationManager(db_path="/invalid/path/db.db")


class TestRecordMessage:
    """测试消息记录"""

    def test_record_user_message(self, manager):
        """测试记录用户消息"""
        msg_id = manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Hello AGI!",
        )

        assert msg_id.startswith("msg_")
        assert len(msg_id) > 10

    def test_record_assistant_message(self, manager):
        """测试记录助手消息"""
        msg_id = manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="assistant",
            content="Hello! How can I help?",
        )

        assert msg_id.startswith("msg_")

    def test_record_system_message(self, manager):
        """测试记录系统消息"""
        msg_id = manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="system",
            content="Session started",
        )

        assert msg_id.startswith("msg_")

    def test_record_message_with_metadata(self, manager):
        """测试带元数据的消息"""
        metadata = {"source": "web", "ip": "127.0.0.1", "tokens": 150}

        msg_id = manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Test message",
            metadata=metadata,
        )

        # 验证消息已记录
        history = manager.get_history("sess_001")
        assert len(history) == 1
        assert history[0]["metadata"] == metadata

    def test_record_message_invalid_role(self, manager):
        """测试无效角色"""
        with pytest.raises(ValueError, match="Invalid role"):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="invalid_role",
                content="Test",
            )

    def test_record_message_empty_content(self, manager):
        """测试空内容"""
        with pytest.raises(ValueError, match="Invalid content"):
            manager.record_message(
                session_id="sess_001", user_id="user_123", role="user", content=""
            )

    def test_record_message_invalidates_cache(self, manager):
        """测试记录消息失效缓存"""
        # 先建立缓存
        manager.get_history("sess_001")

        # 记录新消息
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="New message",
        )

        # 验证缓存已失效
        assert "sess_001" not in manager.cache


class TestGetHistory:
    """测试对话历史检索"""

    def test_get_empty_history(self, manager):
        """测试空历史"""
        history = manager.get_history("sess_nonexistent")
        assert history == []

    def test_get_history_single_message(self, manager):
        """测试单条消息"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="First message",
        )

        history = manager.get_history("sess_001")
        assert len(history) == 1
        assert history[0]["content"] == "First message"

    def test_get_history_multiple_messages(self, manager):
        """测试多条消息"""
        for i in range(5):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        history = manager.get_history("sess_001")
        assert len(history) == 5
        assert history[0]["content"] == "Message 0"
        assert history[4]["content"] == "Message 4"

    def test_get_history_ordered_by_time(self, manager):
        """测试按时间排序"""
        messages = ["First", "Second", "Third"]
        for msg in messages:
            manager.record_message(
                session_id="sess_001", user_id="user_123", role="user", content=msg
            )
            time.sleep(0.01)  # 确保时间戳不同

        history = manager.get_history("sess_001")
        assert len(history) == 3
        assert history[0]["content"] == "First"
        assert history[1]["content"] == "Second"
        assert history[2]["content"] == "Third"

    def test_get_history_with_limit(self, manager):
        """测试限制返回数量"""
        for i in range(10):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user",
                content=f"Message {i}",
            )

        history = manager.get_history("sess_001", limit=5)
        assert len(history) == 5

    def test_get_history_with_offset(self, manager):
        """测试偏移量"""
        for i in range(10):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user",
                content=f"Message {i}",
            )

        history = manager.get_history("sess_001", limit=3, offset=5)
        assert len(history) == 3
        assert history[0]["content"] == "Message 5"

    def test_get_history_uses_cache(self, manager):
        """测试缓存机制"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Test message",
        )

        # 第一次查询，建立缓存
        history1 = manager.get_history("sess_001")
        assert "sess_001" in manager.cache

        # 第二次查询，使用缓存
        history2 = manager.get_history("sess_001")
        assert history1 == history2

    def test_get_history_cache_expiry(self, manager):
        """测试缓存过期"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Test message",
        )

        # 第一次查询
        manager.get_history("sess_001")
        assert "sess_001" in manager.cache

        # 等待缓存过期 (cache_ttl=2秒)
        time.sleep(2.5)

        # 查询时缓存应已过期
        manager.get_history("sess_001")
        # 新缓存应该建立
        assert "sess_001" in manager.cache


class TestSearchConversations:
    """测试全文检索"""

    def test_search_single_match(self, manager):
        """测试单个匹配"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Tell me about artificial intelligence",
        )

        results = manager.search_conversations("user_123", "artificial intelligence")
        assert len(results) == 1
        assert "artificial intelligence" in results[0]["content"]

    def test_search_multiple_matches(self, manager):
        """测试多个匹配"""
        keywords = ["machine learning", "deep learning", "neural networks"]
        for keyword in keywords:
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user",
                content=f"Explain {keyword}",
            )

        results = manager.search_conversations("user_123", "learning")
        assert len(results) == 2  # machine learning + deep learning

    def test_search_no_matches(self, manager):
        """测试无匹配"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Hello world",
        )

        results = manager.search_conversations("user_123", "nonexistent")
        assert len(results) == 0

    def test_search_within_session(self, manager):
        """测试会话内搜索"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Session 1 message about AGI",
        )
        manager.record_message(
            session_id="sess_002",
            user_id="user_123",
            role="user",
            content="Session 2 message about AGI",
        )

        results = manager.search_conversations(
            "user_123", "AGI", session_id="sess_001"
        )
        assert len(results) == 1
        assert results[0]["session_id"] == "sess_001"

    def test_search_across_sessions(self, manager):
        """测试跨会话搜索"""
        for i in range(3):
            manager.record_message(
                session_id=f"sess_00{i}",
                user_id="user_123",
                role="user",
                content=f"Session {i} discussing quantum computing",
            )

        results = manager.search_conversations("user_123", "quantum")
        assert len(results) == 3

    def test_search_with_limit(self, manager):
        """测试限制结果数量"""
        for i in range(10):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user",
                content=f"Message {i} about Python programming",
            )

        results = manager.search_conversations("user_123", "Python", limit=5)
        assert len(results) == 5

    def test_search_different_users(self, manager):
        """测试不同用户的消息"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="User 123 message about AI",
        )
        manager.record_message(
            session_id="sess_002",
            user_id="user_456",
            role="user",
            content="User 456 message about AI",
        )

        results = manager.search_conversations("user_123", "AI")
        assert len(results) == 1
        assert results[0]["user_id"] == "user_123"


class TestGetMessageCount:
    """测试消息计数"""

    def test_count_empty_session(self, manager):
        """测试空会话"""
        count = manager.get_message_count("sess_nonexistent")
        assert count == 0

    def test_count_single_message(self, manager):
        """测试单条消息"""
        manager.record_message(
            session_id="sess_001", user_id="user_123", role="user", content="Test"
        )
        count = manager.get_message_count("sess_001")
        assert count == 1

    def test_count_multiple_messages(self, manager):
        """测试多条消息"""
        for i in range(7):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user",
                content=f"Message {i}",
            )

        count = manager.get_message_count("sess_001")
        assert count == 7


class TestDeleteSessionMessages:
    """测试删除会话消息"""

    def test_delete_empty_session(self, manager):
        """测试删除空会话"""
        # 不应抛出异常
        manager.delete_session_messages("sess_nonexistent")

    def test_delete_single_message(self, manager):
        """测试删除单条消息"""
        manager.record_message(
            session_id="sess_001", user_id="user_123", role="user", content="Test"
        )

        manager.delete_session_messages("sess_001")

        history = manager.get_history("sess_001")
        assert len(history) == 0

    def test_delete_multiple_messages(self, manager):
        """测试删除多条消息"""
        for i in range(5):
            manager.record_message(
                session_id="sess_001",
                user_id="user_123",
                role="user",
                content=f"Message {i}",
            )

        manager.delete_session_messages("sess_001")

        count = manager.get_message_count("sess_001")
        assert count == 0

    def test_delete_invalidates_cache(self, manager):
        """测试删除失效缓存"""
        manager.record_message(
            session_id="sess_001", user_id="user_123", role="user", content="Test"
        )

        # 建立缓存
        manager.get_history("sess_001")
        assert "sess_001" in manager.cache

        # 删除消息
        manager.delete_session_messages("sess_001")
        assert "sess_001" not in manager.cache

    def test_delete_from_fts(self, manager):
        """测试从FTS表删除"""
        manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content="Python programming",
        )

        # 验证可搜索
        results = manager.search_conversations("user_123", "Python")
        assert len(results) == 1

        # 删除消息
        manager.delete_session_messages("sess_001")

        # 验证搜索不到
        results = manager.search_conversations("user_123", "Python")
        assert len(results) == 0


class TestEdgeCases:
    """测试边界条件"""

    def test_very_long_content(self, manager):
        """测试超长内容"""
        long_content = "A" * 10000

        msg_id = manager.record_message(
            session_id="sess_001", user_id="user_123", role="user", content=long_content
        )

        history = manager.get_history("sess_001")
        assert len(history[0]["content"]) == 10000

    def test_special_characters_in_content(self, manager):
        """测试特殊字符"""
        special_content = "Test 中文 émojis 🚀 \n\t quotes \"' etc."

        msg_id = manager.record_message(
            session_id="sess_001",
            user_id="user_123",
            role="user",
            content=special_content,
        )

        history = manager.get_history("sess_001")
        assert history[0]["content"] == special_content

    def test_concurrent_sessions(self, manager):
        """测试并发会话"""
        sessions = [f"sess_00{i}" for i in range(10)]

        for sess_id in sessions:
            manager.record_message(
                session_id=sess_id,
                user_id="user_123",
                role="user",
                content=f"Message in {sess_id}",
            )

        # 验证所有会话都有消息
        for sess_id in sessions:
            history = manager.get_history(sess_id)
            assert len(history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=conversation_manager", "--cov-report=term"])
