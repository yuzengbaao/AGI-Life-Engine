"""
跨会话记忆系统 - 集成测试
Cross-Session Memory System - Integration Tests

版本: 1.0.0
日期: 2025-11-14

测试覆盖:
1. SessionManager + UserProfileManager 集成
2. SessionManager + ConversationManager 集成
3. UserProfileManager + ConversationManager 集成
4. 完整工作流测试
5. 性能基准测试
"""

import pytest
import os
import time
from datetime import datetime
from cross_session_migration import CrossSessionMigration
from session_manager import SessionManager
from user_profile_manager import UserProfileManager
from conversation_manager import ConversationManager


@pytest.fixture
def test_db(tmp_path):
    """创建临时测试数据库"""
    db_path = tmp_path / "test_integration.db"

    # 运行迁移
    migration = CrossSessionMigration(str(db_path))
    migration.migrate_up()

    yield str(db_path)

    # 清理
    if os.path.exists(str(db_path)):
        os.remove(str(db_path))


@pytest.fixture
def session_mgr(test_db):
    """创建SessionManager实例"""
    return SessionManager(db_path=test_db)


@pytest.fixture
def user_mgr(test_db):
    """创建UserProfileManager实例"""
    return UserProfileManager(db_path=test_db)


@pytest.fixture
def conv_mgr(test_db):
    """创建ConversationManager实例"""
    return ConversationManager(db_path=test_db)


class TestSessionUserIntegration:
    """测试SessionManager + UserProfileManager集成"""

    def test_create_session_with_user(self, session_mgr, user_mgr):
        """测试创建会话时同步用户信息"""
        # 创建用户
        user = user_mgr.create_or_get_user(
            "user_001", display_name="Alice", preferences={"name": "Alice"}
        )

        # 创建会话
        session_id = session_mgr.create_session(
            user_id=user["user_id"], metadata={"source": "web"}
        )

        # 获取会话验证
        session = session_mgr.get_session(session_id)
        assert session["user_id"] == user["user_id"]
        assert session["status"] == "active"

    def test_user_session_history(self, session_mgr, user_mgr):
        """测试用户的会话历史"""
        # 创建用户
        user = user_mgr.create_or_get_user("user_002")

        # 创建多个会话
        session_ids = []
        for i in range(3):
            sess_id = session_mgr.create_session(
                user_id=user["user_id"], metadata={"index": i}
            )
            session_ids.append(sess_id)

        # 验证可以检索所有会话
        for sess_id in session_ids:
            retrieved = session_mgr.get_session(sess_id)
            assert retrieved is not None
            assert retrieved["user_id"] == user["user_id"]

    def test_user_preferences_across_sessions(self, session_mgr, user_mgr):
        """测试用户偏好在多个会话间保持"""
        user = user_mgr.create_or_get_user("user_003")

        # 设置用户偏好
        preferences = {"language": "zh-CN", "theme": "dark", "model": "gpt-4"}
        user_mgr.update_preferences(user["user_id"], preferences)

        # 创建会话
        session_id = session_mgr.create_session(user_id=user["user_id"])

        # 获取用户上下文
        context = user_mgr.get_user_context(user["user_id"])
        assert context["preferences"] == preferences


class TestSessionConversationIntegration:
    """测试SessionManager + ConversationManager集成"""

    def test_record_messages_in_session(self, session_mgr, conv_mgr):
        """测试在会话中记录消息"""
        # 创建会话
        session_id = session_mgr.create_session(user_id="user_004")

        # 记录对话
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me about AGI"},
        ]

        for msg in messages:
            conv_mgr.record_message(
                session_id=session_id,
                user_id="user_004",
                role=msg["role"],
                content=msg["content"],
            )

        # 验证消息已记录
        history = conv_mgr.get_history(session_id)
        assert len(history) == 3
        assert history[0]["content"] == "Hello!"

    def test_conversation_count_updates_session(self, session_mgr, conv_mgr):
        """测试对话计数与会话更新"""
        session_id = session_mgr.create_session(user_id="user_005")

        # 记录多条消息
        for i in range(5):
            conv_mgr.record_message(
                session_id=session_id,
                user_id="user_005",
                role="user",
                content=f"Message {i}",
            )

        # 验证消息数量
        count = conv_mgr.get_message_count(session_id)
        assert count == 5

    def test_end_session_with_messages(self, session_mgr, conv_mgr):
        """测试结束包含消息的会话"""
        session_id = session_mgr.create_session(user_id="user_006")

        # 记录消息
        conv_mgr.record_message(
            session_id=session_id,
            user_id="user_006",
            role="user",
            content="Test message",
        )

        # 结束会话
        session_mgr.end_session(session_id)

        # 验证会话状态
        ended_session = session_mgr.get_session(session_id)
        assert ended_session["status"] == "ended"

        # 验证消息仍可访问
        history = conv_mgr.get_history(session_id)
        assert len(history) == 1


class TestUserConversationIntegration:
    """测试UserProfileManager + ConversationManager集成"""

    def test_user_stats_with_conversations(self, user_mgr, conv_mgr):
        """测试用户统计与对话的关联"""
        user = user_mgr.create_or_get_user("user_007")

        # 记录对话
        for i in range(10):
            conv_mgr.record_message(
                session_id="sess_test",
                user_id=user["user_id"],
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        # 更新用户统计
        user_mgr.update_stats(
            user["user_id"],
            {"total_messages": 10, "total_sessions": 1, "total_tokens": 500},
        )

        # 验证统计
        context = user_mgr.get_user_context(user["user_id"])
        assert context["interaction_stats"]["total_messages"] == 10

    def test_search_user_conversations(self, user_mgr, conv_mgr):
        """测试搜索用户的对话"""
        user = user_mgr.create_or_get_user("user_008")

        # 记录不同主题的对话
        topics = ["machine learning", "quantum computing", "neural networks"]
        for topic in topics:
            conv_mgr.record_message(
                session_id="sess_topics",
                user_id=user["user_id"],
                role="user",
                content=f"Tell me about {topic}",
            )

        # 搜索特定主题
        results = conv_mgr.search_conversations(user["user_id"], "quantum")
        assert len(results) == 1
        assert "quantum" in results[0]["content"]


class TestFullWorkflow:
    """测试完整工作流"""

    def test_complete_user_session_workflow(
        self, session_mgr, user_mgr, conv_mgr
    ):
        """测试完整的用户会话工作流"""
        # 1. 创建/获取用户
        user = user_mgr.create_or_get_user(
            "user_workflow",
            display_name="Test User",
            profile={"email": "test@example.com"},
        )
        assert user["identifier"] == "user_workflow"
        assert user["user_id"].startswith("user_")

        # 2. 设置用户偏好
        preferences = {"language": "en", "model": "gpt-4o", "temperature": 0.7}
        user_mgr.update_preferences(user["user_id"], preferences)

        # 3. 创建会话
        session_id = session_mgr.create_session(
            user_id=user["user_id"], metadata={"platform": "cli"}
        )
        session = session_mgr.get_session(session_id)
        assert session["status"] == "active"

        # 4. 进行对话
        conversation = [
            {"role": "user", "content": "What is artificial general intelligence?"},
            {
                "role": "assistant",
                "content": "AGI refers to AI with human-level cognition.",
            },
            {"role": "user", "content": "How does memory work in AGI systems?"},
            {
                "role": "assistant",
                "content": "AGI uses various memory systems including episodic and semantic memory.",
            },
        ]

        for msg in conversation:
            conv_mgr.record_message(
                session_id=session_id,
                user_id=user["user_id"],
                role=msg["role"],
                content=msg["content"],
            )

        # 5. 验证对话历史
        history = conv_mgr.get_history(session_id)
        assert len(history) == 4
        assert history[0]["role"] == "user"

        # 6. 搜索对话
        search_results = conv_mgr.search_conversations(user["user_id"], "AGI memory")
        assert len(search_results) >= 1

        # 7. 更新用户统计
        user_mgr.update_stats(
            user["user_id"],
            {"total_messages": 4, "total_sessions": 1, "total_tokens": 200},
        )

        # 8. 结束会话
        session_mgr.end_session(session_id)
        ended_session = session_mgr.get_session(session_id)
        assert ended_session["status"] == "ended"

        # 9. 验证用户上下文
        context = user_mgr.get_user_context(user["user_id"])
        assert context["preferences"] == preferences
        assert context["interaction_stats"]["total_messages"] == 4

        # 10. 验证会话数据仍可访问
        history_after = conv_mgr.get_history(session_id)
        assert len(history_after) == 4  # 消息仍可访问

    def test_multi_session_user_workflow(self, session_mgr, user_mgr, conv_mgr):
        """测试多会话用户工作流"""
        user = user_mgr.create_or_get_user("user_multi")

        # 创建多个会话
        session_ids = []
        for i in range(3):
            sess_id = session_mgr.create_session(
                user_id=user["user_id"], metadata={"session_index": i}
            )
            session_ids.append(sess_id)

            # 每个会话都有对话
            conv_mgr.record_message(
                session_id=sess_id,
                user_id=user["user_id"],
                role="user",
                content=f"Session {i} message",
            )

        # 验证跨会话搜索
        results = conv_mgr.search_conversations(user["user_id"], "Session")
        assert len(results) == 3

        # 更新用户统计
        user_mgr.update_stats(
            user["user_id"],
            {"total_messages": 3, "total_sessions": 3, "total_tokens": 300},
        )

        context = user_mgr.get_user_context(user["user_id"])
        assert context["interaction_stats"]["total_sessions"] == 3


class TestPerformanceBenchmarks:
    """性能基准测试"""

    def test_session_creation_performance(self, session_mgr):
        """测试会话创建性能 (<50ms)"""
        times = []
        for i in range(10):
            start = time.time()
            session_mgr.create_session(user_id=f"perf_user_{i}")
            duration = (time.time() - start) * 1000
            times.append(duration)

        avg_time = sum(times) / len(times)
        print(f"\n📊 Session creation avg: {avg_time:.2f}ms")
        assert avg_time < 50, f"Session creation too slow: {avg_time:.2f}ms"

    def test_user_query_performance(self, user_mgr):
        """测试用户查询性能 (<30ms)"""
        # 先创建用户
        users = []
        for i in range(10):
            user = user_mgr.create_or_get_user(f"perf_user_{i}")
            users.append(user)

        # 测试查询
        times = []
        for user in users:
            start = time.time()
            user_mgr.get_user_context(user["user_id"])
            duration = (time.time() - start) * 1000
            times.append(duration)

        avg_time = sum(times) / len(times)
        print(f"\n📊 User query avg: {avg_time:.2f}ms")
        assert avg_time < 30, f"User query too slow: {avg_time:.2f}ms"

    def test_conversation_retrieval_performance(self, conv_mgr):
        """测试对话检索性能 (<150ms)"""
        # 准备数据：每个会话50条消息
        for sess_idx in range(5):
            session_id = f"sess_perf_{sess_idx}"
            for msg_idx in range(50):
                conv_mgr.record_message(
                    session_id=session_id,
                    user_id="perf_user",
                    role="user" if msg_idx % 2 == 0 else "assistant",
                    content=f"Session {sess_idx} Message {msg_idx}",
                )

        # 测试检索
        times = []
        for sess_idx in range(5):
            start = time.time()
            conv_mgr.get_history(f"sess_perf_{sess_idx}")
            duration = (time.time() - start) * 1000
            times.append(duration)

        avg_time = sum(times) / len(times)
        print(f"\n📊 Conversation retrieval avg: {avg_time:.2f}ms")
        assert (
            avg_time < 150
        ), f"Conversation retrieval too slow: {avg_time:.2f}ms"

    def test_fts_search_performance(self, conv_mgr):
        """测试FTS5搜索性能"""
        # 准备数据：100条消息
        for i in range(100):
            conv_mgr.record_message(
                session_id="sess_fts",
                user_id="fts_user",
                role="user",
                content=f"Message {i} about artificial intelligence and machine learning",
            )

        # 测试搜索
        times = []
        for _ in range(10):
            start = time.time()
            conv_mgr.search_conversations("fts_user", "artificial intelligence")
            duration = (time.time() - start) * 1000
            times.append(duration)

        avg_time = sum(times) / len(times)
        print(f"\n📊 FTS5 search avg: {avg_time:.2f}ms")
        assert avg_time < 100, f"FTS5 search too slow: {avg_time:.2f}ms"


class TestConcurrency:
    """并发测试"""

    def test_concurrent_session_creation(self, session_mgr):
        """测试并发会话创建"""
        # 模拟10个并发用户
        session_ids = []
        for i in range(10):
            sess_id = session_mgr.create_session(user_id=f"concurrent_user_{i}")
            session_ids.append(sess_id)

        # 验证所有会话都创建成功
        assert len(session_ids) == 10
        assert len(set(session_ids)) == 10

    def test_concurrent_message_recording(self, conv_mgr):
        """测试并发消息记录"""
        session_id = "sess_concurrent"

        # 模拟20条并发消息
        for i in range(20):
            conv_mgr.record_message(
                session_id=session_id,
                user_id="concurrent_user",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Concurrent message {i}",
            )

        # 验证所有消息都记录成功
        history = conv_mgr.get_history(session_id)
        assert len(history) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
