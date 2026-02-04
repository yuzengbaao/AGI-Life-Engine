"""
跨会话记忆系统 - 上下文过滤器单元测试
Cross-Session Memory System - Context Filter Unit Tests

版本: 1.0.0
日期: 2025-11-14

测试覆盖:
1. 初始化与参数验证
2. 用户级过滤
3. 会话级过滤
4. 角色过滤
5. 标签过滤 (OR/AND)
6. 复合条件过滤
7. 时间+用户组合过滤
8. 错误处理
"""

import pytest
import os
import json
from context_filter import (
    ContextFilter,
    ContextFilterError,
    FilterCondition,
    CompositeFilter,
    FilterOperator,
)
from conversation_manager import ConversationManager
from temporal_retrieval import TemporalRetrieval
from cross_session_migration import CrossSessionMigration


@pytest.fixture
def test_db(tmp_path):
    """创建临时测试数据库"""
    db_path = tmp_path / "test_context_filter.db"
    migration = CrossSessionMigration(str(db_path))
    migration.migrate_up()
    yield str(db_path)
    # 清理
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def conv_mgr(test_db):
    """创建对话管理器"""
    return ConversationManager(test_db)


@pytest.fixture
def temporal(conv_mgr):
    """创建时序检索器"""
    return TemporalRetrieval(conv_mgr)


@pytest.fixture
def context_filter(conv_mgr, temporal):
    """创建上下文过滤器"""
    return ContextFilter(conv_mgr, temporal)


class TestInitialization:
    """测试初始化"""

    def test_init_with_valid_params(self, conv_mgr, temporal):
        """测试有效参数初始化"""
        filter_mgr = ContextFilter(conv_mgr, temporal)
        assert filter_mgr.conversation_mgr == conv_mgr
        assert filter_mgr.temporal_retrieval == temporal

    def test_init_without_temporal(self, conv_mgr):
        """测试不带temporal初始化"""
        filter_mgr = ContextFilter(conv_mgr)
        assert filter_mgr.conversation_mgr == conv_mgr
        assert filter_mgr.temporal_retrieval is None

    def test_init_without_conv_mgr(self):
        """测试缺少conversation_mgr"""
        with pytest.raises(ValueError, match="conversation_mgr is required"):
            ContextFilter(None)


class TestUserFilter:
    """测试用户级过滤"""

    def test_filter_by_user_valid(self, context_filter, conv_mgr):
        """测试有效用户过滤"""
        # 记录消息
        conv_mgr.record_message("sess_001", "user_alice", "user", "Hello")
        conv_mgr.record_message("sess_001", "user_alice", "assistant", "Hi")
        conv_mgr.record_message("sess_002", "user_bob", "user", "Bye")

        memories = context_filter.filter_by_user("user_alice")
        assert len(memories) == 2
        assert all(m["user_id"] == "user_alice" for m in memories)

    def test_filter_by_user_empty(self, context_filter):
        """测试无记录用户"""
        memories = context_filter.filter_by_user("user_nonexistent")
        assert len(memories) == 0

    def test_filter_by_user_with_limit(self, context_filter, conv_mgr):
        """测试带limit的用户过滤"""
        for i in range(5):
            conv_mgr.record_message("sess_001", "user_test", "user", f"Message {i}")

        memories = context_filter.filter_by_user("user_test", limit=3)
        assert len(memories) == 3

    def test_filter_by_user_invalid_id(self, context_filter):
        """测试空user_id"""
        with pytest.raises(ValueError, match="user_id is required"):
            context_filter.filter_by_user("")


class TestSessionFilter:
    """测试会话级过滤"""

    def test_filter_by_session_valid(self, context_filter, conv_mgr):
        """测试有效会话过滤"""
        conv_mgr.record_message("sess_001", "user_alice", "user", "Hello")
        conv_mgr.record_message("sess_001", "user_alice", "assistant", "Hi")
        conv_mgr.record_message("sess_002", "user_alice", "user", "Bye")

        memories = context_filter.filter_by_session("sess_001")
        assert len(memories) == 2
        assert all(m["session_id"] == "sess_001" for m in memories)

    def test_filter_by_session_empty(self, context_filter):
        """测试不存在的会话"""
        memories = context_filter.filter_by_session("sess_nonexistent")
        assert len(memories) == 0

    def test_filter_by_session_with_limit(self, context_filter, conv_mgr):
        """测试带limit的会话过滤"""
        for i in range(5):
            conv_mgr.record_message("sess_test", "user_test", "user", f"Msg {i}")

        memories = context_filter.filter_by_session("sess_test", limit=2)
        assert len(memories) == 2

    def test_filter_by_session_invalid_id(self, context_filter):
        """测试空session_id"""
        with pytest.raises(ValueError, match="session_id is required"):
            context_filter.filter_by_session("")


class TestRoleFilter:
    """测试角色过滤"""

    def test_filter_by_role_user(self, context_filter, conv_mgr):
        """测试过滤user角色"""
        conv_mgr.record_message("sess_001", "user_alice", "user", "Q1")
        conv_mgr.record_message("sess_001", "user_alice", "assistant", "A1")
        conv_mgr.record_message("sess_001", "user_alice", "user", "Q2")

        memories = context_filter.filter_by_role("user_alice", "user")
        assert len(memories) == 2
        assert all(m["role"] == "user" for m in memories)

    def test_filter_by_role_assistant(self, context_filter, conv_mgr):
        """测试过滤assistant角色"""
        conv_mgr.record_message("sess_001", "user_alice", "user", "Q1")
        conv_mgr.record_message("sess_001", "user_alice", "assistant", "A1")

        memories = context_filter.filter_by_role("user_alice", "assistant")
        assert len(memories) == 1
        assert memories[0]["role"] == "assistant"

    def test_filter_by_role_invalid_role(self, context_filter):
        """测试无效角色"""
        with pytest.raises(ValueError, match="role must be"):
            context_filter.filter_by_role("user_test", "invalid_role")

    def test_filter_by_role_invalid_user(self, context_filter):
        """测试空user_id"""
        with pytest.raises(ValueError, match="user_id is required"):
            context_filter.filter_by_role("", "user")


class TestTagFilter:
    """测试标签过滤"""

    def test_filter_by_tags_or_mode(self, context_filter, conv_mgr):
        """测试OR模式标签过滤"""
        conv_mgr.record_message(
            "sess_001",
            "user_alice",
            "user",
            "Msg1",
            metadata={"tags": ["work", "urgent"]},
        )
        conv_mgr.record_message(
            "sess_001",
            "user_alice",
            "user",
            "Msg2",
            metadata={"tags": ["personal"]},
        )
        conv_mgr.record_message(
            "sess_001",
            "user_alice",
            "user",
            "Msg3",
            metadata={"tags": ["work"]},
        )

        memories = context_filter.filter_by_tags(
            "user_alice", ["work", "personal"], match_all=False
        )
        assert len(memories) == 3  # All have work OR personal

    def test_filter_by_tags_and_mode(self, context_filter, conv_mgr):
        """测试AND模式标签过滤"""
        conv_mgr.record_message(
            "sess_001",
            "user_alice",
            "user",
            "Msg1",
            metadata={"tags": ["work", "urgent"]},
        )
        conv_mgr.record_message(
            "sess_001",
            "user_alice",
            "user",
            "Msg2",
            metadata={"tags": ["work"]},
        )

        memories = context_filter.filter_by_tags(
            "user_alice", ["work", "urgent"], match_all=True
        )
        assert len(memories) == 1  # Only Msg1 has both

    def test_filter_by_tags_no_metadata(self, context_filter, conv_mgr):
        """测试无metadata的消息"""
        conv_mgr.record_message("sess_001", "user_alice", "user", "No tags")

        memories = context_filter.filter_by_tags("user_alice", ["work"])
        assert len(memories) == 0

    def test_filter_by_tags_invalid_metadata(self, context_filter, conv_mgr):
        """测试无效JSON metadata"""
        conv_mgr.record_message(
            "sess_001", "user_alice", "user", "Bad JSON", metadata="{invalid json"
        )

        memories = context_filter.filter_by_tags("user_alice", ["work"])
        assert len(memories) == 0

    def test_filter_by_tags_empty_tags(self, context_filter):
        """测试空标签列表"""
        with pytest.raises(ValueError, match="tags cannot be empty"):
            context_filter.filter_by_tags("user_alice", [])


class TestFilterCondition:
    """测试FilterCondition"""

    def test_condition_eq(self):
        """测试等于条件"""
        cond = FilterCondition("role", "eq", "user")
        assert cond.evaluate({"role": "user"}) is True
        assert cond.evaluate({"role": "assistant"}) is False

    def test_condition_ne(self):
        """测试不等于条件"""
        cond = FilterCondition("role", "ne", "user")
        assert cond.evaluate({"role": "assistant"}) is True
        assert cond.evaluate({"role": "user"}) is False

    def test_condition_in(self):
        """测试in条件"""
        cond = FilterCondition("role", "in", ["user", "assistant"])
        assert cond.evaluate({"role": "user"}) is True
        assert cond.evaluate({"role": "system"}) is False

    def test_condition_contains(self):
        """测试contains条件"""
        cond = FilterCondition("content", "contains", "AGI")
        assert cond.evaluate({"content": "Talk about AGI"}) is True
        assert cond.evaluate({"content": "Hello world"}) is False

    def test_condition_contains_list(self):
        """测试contains条件(列表)"""
        cond = FilterCondition("tags", "contains", "work")
        assert cond.evaluate({"tags": ["work", "urgent"]}) is True
        assert cond.evaluate({"tags": ["personal"]}) is False

    def test_condition_startswith(self):
        """测试startswith条件"""
        cond = FilterCondition("content", "startswith", "Hello")
        assert cond.evaluate({"content": "Hello world"}) is True
        assert cond.evaluate({"content": "world Hello"}) is False

    def test_condition_endswith(self):
        """测试endswith条件"""
        cond = FilterCondition("content", "endswith", "world")
        assert cond.evaluate({"content": "Hello world"}) is True
        assert cond.evaluate({"content": "world Hello"}) is False

    def test_condition_negate(self):
        """测试取反条件"""
        cond = FilterCondition("role", "eq", "user", negate=True)
        assert cond.evaluate({"role": "assistant"}) is True
        assert cond.evaluate({"role": "user"}) is False


class TestCompositeFilter:
    """测试CompositeFilter"""

    def test_composite_and_operator(self):
        """测试AND组合"""
        composite = CompositeFilter(
            conditions=[
                FilterCondition("role", "eq", "user"),
                FilterCondition("user_id", "eq", "alice"),
            ],
            operator=FilterOperator.AND,
        )

        assert (
            composite.evaluate({"role": "user", "user_id": "alice"}) is True
        )
        assert (
            composite.evaluate({"role": "assistant", "user_id": "alice"})
            is False
        )

    def test_composite_or_operator(self):
        """测试OR组合"""
        composite = CompositeFilter(
            conditions=[
                FilterCondition("role", "eq", "user"),
                FilterCondition("role", "eq", "assistant"),
            ],
            operator=FilterOperator.OR,
        )

        assert composite.evaluate({"role": "user"}) is True
        assert composite.evaluate({"role": "assistant"}) is True
        assert composite.evaluate({"role": "system"}) is False

    def test_composite_empty(self):
        """测试空复合过滤器"""
        composite = CompositeFilter()
        assert composite.evaluate({"any": "data"}) is True

    def test_filter_by_composite(self, context_filter, conv_mgr):
        """测试使用复合过滤器过滤"""
        conv_mgr.record_message("sess_001", "user_alice", "user", "Q1")
        conv_mgr.record_message("sess_001", "user_alice", "assistant", "A1")
        conv_mgr.record_message("sess_002", "user_bob", "user", "Q2")

        composite = CompositeFilter(
            conditions=[
                FilterCondition("user_id", "eq", "user_alice"),
                FilterCondition("role", "eq", "user"),
            ],
            operator=FilterOperator.AND,
        )

        memories = context_filter.filter_by_composite(composite)
        assert len(memories) == 1
        assert memories[0]["user_id"] == "user_alice"
        assert memories[0]["role"] == "user"


class TestTimeAndUserFilter:
    """测试时间+用户组合过滤"""

    def test_filter_by_time_and_user(self, context_filter, conv_mgr):
        """测试时间+用户过滤"""
        conv_mgr.record_message("sess_001", "user_alice", "user", "Recent")

        memories = context_filter.filter_by_time_and_user("user_alice", hours=24)
        assert len(memories) >= 1

    def test_filter_by_time_and_user_no_temporal(self, conv_mgr):
        """测试无temporal配置"""
        filter_no_temporal = ContextFilter(conv_mgr)

        with pytest.raises(ContextFilterError, match="temporal_retrieval not configured"):
            filter_no_temporal.filter_by_time_and_user("user_test", hours=24)

    def test_filter_by_time_and_user_error(self, context_filter):
        """测试时间+用户过滤错误处理"""
        old_temporal = context_filter.temporal_retrieval
        context_filter.temporal_retrieval.conversation_mgr.db_path = "/invalid/path.db"
        
        with pytest.raises(ContextFilterError, match="Failed to filter by time and user"):
            context_filter.filter_by_time_and_user("user_test", hours=24)
        
        context_filter.temporal_retrieval = old_temporal


class TestErrorHandling:
    """测试错误处理"""

    def test_filter_by_composite_without_user_id(self, context_filter):
        """测试复合过滤器缺少user_id"""
        composite = CompositeFilter(
            conditions=[FilterCondition("role", "eq", "user")], operator=FilterOperator.AND
        )

        # ValueError被包装成ContextFilterError
        with pytest.raises(ContextFilterError, match="base_memories required or user_id must be in conditions"):
            context_filter.filter_by_composite(composite)

    def test_filter_by_composite_invalid_filter(self, context_filter):
        """测试空composite_filter"""
        with pytest.raises(ValueError, match="composite_filter is required"):
            context_filter.filter_by_composite(None)

    def test_filter_by_user_error_handling(self, context_filter):
        """测试filter_by_user错误处理"""
        # 损坏temporal_retrieval来触发异常
        old_temporal = context_filter.temporal_retrieval
        context_filter.temporal_retrieval.conversation_mgr.db_path = "/invalid/path.db"
        
        with pytest.raises(ContextFilterError, match="Failed to filter by user"):
            context_filter.filter_by_user("user_test")
        
        # 恢复
        context_filter.temporal_retrieval = old_temporal

    def test_filter_by_session_error(self, conv_mgr):
        """测试filter_by_session错误处理"""
        # 创建filter但破坏conv_mgr
        filter_mgr = ContextFilter(conv_mgr)
        conv_mgr.db_path = "/invalid/path.db"
        
        with pytest.raises(ContextFilterError, match="Failed to filter by session"):
            filter_mgr.filter_by_session("sess_test")

    def test_filter_by_role_error(self, context_filter):
        """测试filter_by_role错误处理"""
        # 破坏temporal检索
        old_temporal = context_filter.temporal_retrieval
        context_filter.temporal_retrieval.conversation_mgr.db_path = "/invalid/path.db"
        
        with pytest.raises(ContextFilterError, match="Failed to filter by role"):
            context_filter.filter_by_role("user_test", "user")
        
        context_filter.temporal_retrieval = old_temporal

    def test_filter_by_tags_error(self, context_filter):
        """测试filter_by_tags错误处理"""
        old_temporal = context_filter.temporal_retrieval
        context_filter.temporal_retrieval.conversation_mgr.db_path = "/invalid/path.db"
        
        with pytest.raises(ContextFilterError, match="Failed to filter by tags"):
            context_filter.filter_by_tags("user_test", ["work"])
        
        context_filter.temporal_retrieval = old_temporal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
