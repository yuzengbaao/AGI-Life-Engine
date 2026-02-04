"""
跨会话记忆系统 - SessionManager单元测试
Cross-Session Memory System - Session Manager Tests

版本: 1.0.0
测试SessionManager的所有功能
"""

import pytest
import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from session_manager import SessionManager, SessionNotFoundError, SessionManagerError
from cross_session_migration import CrossSessionMigration


class TestSessionManager:
    """SessionManager单元测试类"""
    
    @pytest.fixture
    def test_db_path(self, tmp_path):
        """创建临时测试数据库"""
        db_path = tmp_path / "test_sessions.db"
        
        # 运行迁移
        migration = CrossSessionMigration(str(db_path))
        migration.migrate_up()
        
        yield str(db_path)
        
        # 清理
        if db_path.exists():
            db_path.unlink()
    
    @pytest.fixture
    def manager(self, test_db_path):
        """创建SessionManager实例"""
        return SessionManager(test_db_path)
    
    def test_init_creates_manager_successfully(self, test_db_path):
        """测试: 初始化成功创建管理器"""
        # Act
        manager = SessionManager(test_db_path)
        
        # Assert
        assert manager.db_path == test_db_path
        assert manager.cache == {}
        assert manager.cache_ttl == 1800
    
    def test_init_with_custom_ttl(self, test_db_path):
        """测试: 使用自定义TTL初始化"""
        # Act
        manager = SessionManager(test_db_path, cache_ttl=3600)
        
        # Assert
        assert manager.cache_ttl == 3600
    
    def test_create_session_success_returns_valid_id(self, manager):
        """测试: 创建会话成功返回有效ID"""
        # Arrange
        user_id = "user_123"
        
        # Act
        session_id = manager.create_session(user_id)
        
        # Assert
        assert session_id.startswith("sess_")
        assert len(session_id) > 20
    
    def test_create_session_with_context_and_metadata(self, manager):
        """测试: 创建会话时传入context和metadata"""
        # Arrange
        user_id = "user_456"
        context = {"last_topic": "AGI", "mood": "curious"}
        metadata = {"source": "web", "version": "1.0"}
        
        # Act
        session_id = manager.create_session(user_id, context, metadata)
        session = manager.get_session(session_id)
        
        # Assert
        assert session['context'] == context
        assert session['metadata'] == metadata
    
    def test_create_session_invalid_user_id_raises_error(self, manager):
        """测试: 无效user_id抛出异常"""
        # Act & Assert
        with pytest.raises(ValueError):
            manager.create_session("")
        
        with pytest.raises(ValueError):
            manager.create_session(None)
    
    def test_get_session_returns_correct_data(self, manager):
        """测试: 获取会话返回正确数据"""
        # Arrange
        user_id = "user_789"
        session_id = manager.create_session(user_id)
        
        # Act
        session = manager.get_session(session_id)
        
        # Assert
        assert session['session_id'] == session_id
        assert session['user_id'] == user_id
        assert session['status'] == 'active'
        assert 'start_time' in session
        assert 'created_at' in session
    
    def test_get_session_not_found_raises_error(self, manager):
        """测试: 获取不存在的会话抛出异常"""
        # Act & Assert
        with pytest.raises(SessionNotFoundError):
            manager.get_session("sess_nonexistent_123")
    
    def test_get_session_uses_cache(self, manager):
        """测试: 获取会话使用缓存"""
        # Arrange
        user_id = "user_cache"
        session_id = manager.create_session(user_id)
        
        # First call - should cache
        session1 = manager.get_session(session_id)
        
        # Second call - should hit cache
        session2 = manager.get_session(session_id)
        
        # Assert
        assert session_id in manager.cache
        assert session1 == session2
    
    def test_update_session_updates_context(self, manager):
        """测试: 更新会话context"""
        # Arrange
        user_id = "user_update"
        session_id = manager.create_session(
            user_id,
            context={"topic": "initial"}
        )
        
        # Act
        manager.update_session(
            session_id,
            context={"topic": "updated", "new_key": "new_value"}
        )
        
        # Assert
        session = manager.get_session(session_id)
        assert session['context']['topic'] == "updated"
        assert session['context']['new_key'] == "new_value"
    
    def test_update_session_updates_status(self, manager):
        """测试: 更新会话状态"""
        # Arrange
        user_id = "user_status"
        session_id = manager.create_session(user_id)
        
        # Act
        manager.update_session(session_id, status='paused')
        
        # Assert
        session = manager.get_session(session_id)
        assert session['status'] == 'paused'
    
    def test_update_session_invalid_status_raises_error(self, manager):
        """测试: 无效状态抛出异常"""
        # Arrange
        user_id = "user_invalid"
        session_id = manager.create_session(user_id)
        
        # Act & Assert
        with pytest.raises(ValueError):
            manager.update_session(session_id, status='invalid_status')
    
    def test_update_session_not_found_raises_error(self, manager):
        """测试: 更新不存在的会话抛出异常"""
        # Act & Assert
        with pytest.raises(SessionNotFoundError):
            manager.update_session("sess_nonexistent", context={"key": "value"})
    
    def test_update_session_invalidates_cache(self, manager):
        """测试: 更新会话失效缓存"""
        # Arrange
        user_id = "user_cache_invalidate"
        session_id = manager.create_session(user_id)
        
        # Cache the session
        manager.get_session(session_id)
        assert session_id in manager.cache
        
        # Act
        manager.update_session(session_id, context={"updated": True})
        
        # Assert
        assert session_id not in manager.cache
    
    def test_end_session_sets_status_to_ended(self, manager):
        """测试: 结束会话设置状态为ended"""
        # Arrange
        user_id = "user_end"
        session_id = manager.create_session(user_id)
        
        # Act
        manager.end_session(session_id)
        
        # Assert
        session = manager.get_session(session_id)
        assert session['status'] == 'ended'
    
    def test_get_recent_sessions_returns_list(self, manager):
        """测试: 获取最近会话返回列表"""
        # Arrange
        user_id = "user_recent"
        session_ids = [
            manager.create_session(user_id) for _ in range(5)
        ]
        
        # Act
        recent = manager.get_recent_sessions(user_id, limit=3)
        
        # Assert
        assert len(recent) == 3
        assert all(s['user_id'] == user_id for s in recent)
    
    def test_get_recent_sessions_sorted_by_last_active(self, manager):
        """测试: 最近会话按last_active降序排列"""
        # Arrange
        user_id = "user_sorted"
        session1 = manager.create_session(user_id)
        session2 = manager.create_session(user_id)
        session3 = manager.create_session(user_id)
        
        # Act
        recent = manager.get_recent_sessions(user_id)
        
        # Assert
        assert recent[0]['session_id'] == session3  # Most recent
        assert recent[1]['session_id'] == session2
        assert recent[2]['session_id'] == session1
    
    def test_get_recent_sessions_filter_by_status(self, manager):
        """测试: 按状态过滤最近会话"""
        # Arrange
        user_id = "user_filter"
        session1 = manager.create_session(user_id)
        session2 = manager.create_session(user_id)
        manager.end_session(session1)
        
        # Act
        active_sessions = manager.get_recent_sessions(user_id, status='active')
        ended_sessions = manager.get_recent_sessions(user_id, status='ended')
        
        # Assert
        assert len(active_sessions) == 1
        assert active_sessions[0]['session_id'] == session2
        assert len(ended_sessions) == 1
        assert ended_sessions[0]['session_id'] == session1
    
    def test_restore_context_returns_complete_context(self, manager):
        """测试: 恢复上下文返回完整信息"""
        # Arrange
        user_id = "user_restore"
        context = {"topic": "Memory", "progress": 0.5}
        metadata = {"version": "2.0"}
        session_id = manager.create_session(user_id, context, metadata)
        
        # Act
        restored = manager.restore_context(session_id)
        
        # Assert
        assert restored['session_id'] == session_id
        assert restored['user_id'] == user_id
        assert restored['context'] == context
        assert restored['metadata'] == metadata
        assert 'start_time' in restored
        assert 'last_active' in restored
        assert 'status' in restored
    
    def test_cache_expiration(self, manager):
        """测试: 缓存过期机制"""
        # Arrange
        manager.cache_ttl = 0.1  # 100ms TTL for testing
        user_id = "user_expire"
        session_id = manager.create_session(user_id)
        
        # Cache the session
        manager.get_session(session_id)
        assert session_id in manager.cache
        
        # Wait for cache to expire
        import time
        time.sleep(0.2)
        
        # Act
        cached = manager._get_from_cache(session_id)
        
        # Assert
        assert cached is None
        assert session_id not in manager.cache
    
    def test_multiple_users_sessions(self, manager):
        """测试: 多用户会话隔离"""
        # Arrange
        user1 = "user_alice"
        user2 = "user_bob"
        
        session1_1 = manager.create_session(user1)
        session1_2 = manager.create_session(user1)
        session2_1 = manager.create_session(user2)
        
        # Act
        alice_sessions = manager.get_recent_sessions(user1)
        bob_sessions = manager.get_recent_sessions(user2)
        
        # Assert
        assert len(alice_sessions) == 2
        assert len(bob_sessions) == 1
        assert all(s['user_id'] == user1 for s in alice_sessions)
        assert all(s['user_id'] == user2 for s in bob_sessions)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
