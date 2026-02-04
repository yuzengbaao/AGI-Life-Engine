"""
跨会话记忆系统 - UserProfileManager单元测试
Cross-Session Memory System - User Profile Manager Tests

版本: 1.0.0
测试UserProfileManager的所有功能
"""

import pytest
import sqlite3
import sys
import json
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_profile_manager import (
    UserProfileManager,
    UserNotFoundError,
    UserProfileManagerError,
)
from cross_session_migration import CrossSessionMigration


class TestUserProfileManager:
    """UserProfileManager单元测试类"""

    @pytest.fixture
    def test_db_path(self, tmp_path):
        """创建临时测试数据库"""
        db_path = tmp_path / "test_users.db"

        # 运行迁移
        migration = CrossSessionMigration(str(db_path))
        migration.migrate_up()

        yield str(db_path)

        # 清理
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def manager(self, test_db_path):
        """创建UserProfileManager实例"""
        return UserProfileManager(test_db_path)

    def test_init_creates_manager_successfully(self, test_db_path):
        """测试: 初始化成功创建管理器"""
        manager = UserProfileManager(test_db_path)
        assert manager.db_path == test_db_path
        assert manager.cache == {}

    def test_create_or_get_user_creates_new_user(self, manager):
        """测试: 创建新用户"""
        user = manager.create_or_get_user("alice@example.com", display_name="Alice")

        assert user["identifier"] == "alice@example.com"
        assert user["display_name"] == "Alice"
        assert user["user_id"].startswith("user_")
        assert "created_at" in user

    def test_create_or_get_user_returns_existing_user(self, manager):
        """测试: 获取已存在的用户"""
        user1 = manager.create_or_get_user("bob@example.com")
        user2 = manager.create_or_get_user("bob@example.com")

        assert user1["user_id"] == user2["user_id"]

    def test_create_or_get_user_with_preferences_and_profile(self, manager):
        """测试: 创建用户时设置偏好和资料"""
        preferences = {"language": "zh-CN", "theme": "dark"}
        profile = {"interests": ["AGI", "ML"]}

        user = manager.create_or_get_user(
            "charlie@example.com", preferences=preferences, profile=profile
        )

        assert user["preferences"] == preferences
        assert user["profile"] == profile

    def test_create_or_get_user_invalid_identifier_raises_error(self, manager):
        """测试: 无效标识符抛出异常"""
        with pytest.raises(ValueError):
            manager.create_or_get_user("")

        with pytest.raises(ValueError):
            manager.create_or_get_user(None)

    def test_get_user_by_id_returns_correct_user(self, manager):
        """测试: 通过ID获取用户"""
        created_user = manager.create_or_get_user("dave@example.com")
        retrieved_user = manager.get_user_by_id(created_user["user_id"])

        assert retrieved_user["user_id"] == created_user["user_id"]
        assert retrieved_user["identifier"] == "dave@example.com"

    def test_get_user_by_id_not_found_raises_error(self, manager):
        """测试: 获取不存在的用户抛出异常"""
        with pytest.raises(UserNotFoundError):
            manager.get_user_by_id("user_nonexistent")

    def test_get_user_by_id_uses_cache(self, manager):
        """测试: 获取用户使用缓存"""
        user = manager.create_or_get_user("eve@example.com")
        user_id = user["user_id"]

        # First call - should cache
        user1 = manager.get_user_by_id(user_id)

        # Second call - should hit cache
        user2 = manager.get_user_by_id(user_id)

        assert user_id in manager.cache
        assert user1 == user2

    def test_update_preferences_merges_by_default(self, manager):
        """测试: 更新偏好默认合并"""
        user = manager.create_or_get_user(
            "frank@example.com", preferences={"key1": "value1"}
        )
        user_id = user["user_id"]

        manager.update_preferences(user_id, {"key2": "value2"})

        updated_user = manager.get_user_by_id(user_id)
        assert updated_user["preferences"]["key1"] == "value1"
        assert updated_user["preferences"]["key2"] == "value2"

    def test_update_preferences_replaces_when_merge_false(self, manager):
        """测试: merge=False时完全替换偏好"""
        user = manager.create_or_get_user(
            "grace@example.com", preferences={"key1": "value1"}
        )
        user_id = user["user_id"]

        manager.update_preferences(user_id, {"key2": "value2"}, merge=False)

        updated_user = manager.get_user_by_id(user_id)
        assert "key1" not in updated_user["preferences"]
        assert updated_user["preferences"]["key2"] == "value2"

    def test_update_preferences_invalidates_cache(self, manager):
        """测试: 更新偏好失效缓存"""
        user = manager.create_or_get_user("henry@example.com")
        user_id = user["user_id"]

        # Cache the user
        manager.get_user_by_id(user_id)
        assert user_id in manager.cache

        # Update preferences
        manager.update_preferences(user_id, {"new_key": "new_value"})

        # Cache should be invalidated
        assert user_id not in manager.cache

    def test_update_stats_updates_statistics(self, manager):
        """测试: 更新统计信息"""
        user = manager.create_or_get_user("iris@example.com")
        user_id = user["user_id"]

        manager.update_stats(user_id, {"total_sessions": 5, "total_messages": 100})

        updated_user = manager.get_user_by_id(user_id)
        assert updated_user["interaction_stats"]["total_sessions"] == 5
        assert updated_user["interaction_stats"]["total_messages"] == 100
        assert "last_interaction" in updated_user["interaction_stats"]

    def test_update_stats_incremental_updates(self, manager):
        """测试: 统计信息增量更新"""
        user = manager.create_or_get_user("jack@example.com")
        user_id = user["user_id"]

        manager.update_stats(user_id, {"total_sessions": 3})
        manager.update_stats(user_id, {"total_messages": 50})

        updated_user = manager.get_user_by_id(user_id)
        assert updated_user["interaction_stats"]["total_sessions"] == 3
        assert updated_user["interaction_stats"]["total_messages"] == 50

    def test_get_user_context_returns_complete_context(self, manager):
        """测试: 获取完整用户上下文"""
        preferences = {"theme": "dark"}
        profile = {"bio": "Test user"}

        user = manager.create_or_get_user(
            "kate@example.com", preferences=preferences, profile=profile
        )
        user_id = user["user_id"]

        context = manager.get_user_context(user_id)

        assert context["user_id"] == user_id
        assert context["preferences"] == preferences
        assert context["profile"] == profile
        assert "interaction_stats" in context

    def test_multiple_users_independence(self, manager):
        """测试: 多用户独立性"""
        user1 = manager.create_or_get_user("user1@example.com")
        user2 = manager.create_or_get_user("user2@example.com")

        manager.update_preferences(user1["user_id"], {"key1": "value1"})
        manager.update_preferences(user2["user_id"], {"key2": "value2"})

        u1 = manager.get_user_by_id(user1["user_id"])
        u2 = manager.get_user_by_id(user2["user_id"])

        assert u1["preferences"]["key1"] == "value1"
        assert "key2" not in u1["preferences"]
        assert u2["preferences"]["key2"] == "value2"
        assert "key1" not in u2["preferences"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
