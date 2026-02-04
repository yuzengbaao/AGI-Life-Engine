"""
跨会话记忆系统 - 数据库迁移测试
Cross-Session Memory System - Migration Tests

版本: 1.0.0
测试迁移脚本的正确性
"""

import pytest
import sqlite3
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_session_migration import CrossSessionMigration, MigrationError


class TestCrossSessionMigration:
    """迁移脚本测试类"""
    
    @pytest.fixture
    def test_db_path(self, tmp_path):
        """创建临时测试数据库路径"""
        db_path = tmp_path / "test_migration.db"
        yield str(db_path)
        # 清理
        if db_path.exists():
            db_path.unlink()
    
    @pytest.fixture
    def migration(self, test_db_path):
        """创建迁移管理器实例"""
        return CrossSessionMigration(test_db_path)
    
    def test_migrate_up_creates_all_tables(self, migration, test_db_path):
        """测试: migrate_up创建所有必需的表"""
        # Act
        result = migration.migrate_up()
        
        # Assert
        assert result is True
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table'
        ''')
        tables = {row[0] for row in cursor.fetchall()}
        
        expected_tables = {
            'sessions',
            'user_profiles',
            'conversations',
            'conversations_fts',
            'tasks',
            'migration_history'
        }
        
        assert expected_tables.issubset(tables)
        conn.close()
    
    def test_migrate_up_creates_all_indexes(self, migration, test_db_path):
        """测试: migrate_up创建所有必需的索引"""
        # Arrange & Act
        migration.migrate_up()
        
        # Assert
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        ''')
        indexes = {row[0] for row in cursor.fetchall()}
        
        expected_indexes = {
            'idx_sessions_user_time',
            'idx_sessions_status',
            'idx_users_identifier',
            'idx_conversations_session',
            'idx_conversations_user',
            'idx_tasks_user_status'
        }
        
        assert expected_indexes.issubset(indexes)
        conn.close()
    
    def test_migrate_up_is_idempotent(self, migration):
        """测试: migrate_up可以重复执行(幂等性)"""
        # Act
        result1 = migration.migrate_up()
        result2 = migration.migrate_up()
        
        # Assert
        assert result1 is True
        assert result2 is True
    
    def test_migrate_down_removes_all_tables(self, migration, test_db_path):
        """测试: migrate_down删除所有表"""
        # Arrange
        migration.migrate_up()
        
        # Act
        result = migration.migrate_down()
        
        # Assert
        assert result is True
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table'
        ''')
        tables = {row[0] for row in cursor.fetchall()}
        
        # 只应该存在sqlite内部表
        assert all(t.startswith('sqlite_') for t in tables)
        conn.close()
    
    def test_verify_succeeds_after_migration(self, migration):
        """测试: 迁移后验证成功"""
        # Arrange
        migration.migrate_up()
        
        # Act
        result = migration.verify()
        
        # Assert
        assert result is True
    
    def test_verify_fails_without_migration(self, migration):
        """测试: 未迁移时验证失败"""
        # Act
        result = migration.verify()
        
        # Assert
        assert result is False
    
    def test_get_migration_status_returns_version(self, migration):
        """测试: 获取迁移状态返回版本信息"""
        # Arrange
        migration.migrate_up()
        
        # Act
        status = migration.get_migration_status()
        
        # Assert
        assert status is not None
        assert 'version' in status
        assert status['version'] == '1.0.0'
        assert 'applied_at' in status
        assert 'description' in status
    
    def test_get_migration_status_returns_none_before_migration(self, migration):
        """测试: 迁移前获取状态返回None"""
        # Act
        status = migration.get_migration_status()
        
        # Assert
        assert status is None
    
    def test_sessions_table_has_correct_schema(self, migration, test_db_path):
        """测试: sessions表有正确的schema"""
        # Arrange
        migration.migrate_up()
        
        # Act
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Assert
        expected_columns = {
            'session_id',
            'user_id',
            'start_time',
            'last_active',
            'context',
            'metadata',
            'status',
            'created_at'
        }
        
        assert expected_columns == columns
        conn.close()
    
    def test_user_profiles_table_has_unique_identifier(self, migration, test_db_path):
        """测试: user_profiles表的identifier字段有唯一约束"""
        # Arrange
        migration.migrate_up()
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Act - 插入相同identifier应该失败
        cursor.execute('''
            INSERT INTO user_profiles 
            (user_id, identifier, created_at, updated_at)
            VALUES ('user1', 'test@example.com', '2025-11-14', '2025-11-14')
        ''')
        
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute('''
                INSERT INTO user_profiles 
                (user_id, identifier, created_at, updated_at)
                VALUES ('user2', 'test@example.com', '2025-11-14', '2025-11-14')
            ''')
        
        conn.close()
    
    def test_conversations_has_foreign_key_to_sessions(self, migration, test_db_path):
        """测试: conversations表有指向sessions的外键"""
        # Arrange
        migration.migrate_up()
        
        conn = sqlite3.connect(test_db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        
        # Act - 插入不存在的session_id应该失败
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute('''
                INSERT INTO conversations 
                (message_id, session_id, user_id, role, content, timestamp, created_at)
                VALUES ('msg1', 'nonexistent_session', 'user1', 'user', 'test', 
                        '2025-11-14', '2025-11-14')
            ''')
        
        conn.close()
    
    def test_tasks_has_valid_progress_constraint(self, migration, test_db_path):
        """测试: tasks表的progress字段有0-1范围约束"""
        # Arrange
        migration.migrate_up()
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # 首先创建用户
        cursor.execute('''
            INSERT INTO user_profiles 
            (user_id, identifier, created_at, updated_at)
            VALUES ('user1', 'test@example.com', '2025-11-14', '2025-11-14')
        ''')
        
        # Act - 插入progress > 1.0应该失败
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute('''
                INSERT INTO tasks 
                (task_id, user_id, title, progress, created_at, updated_at)
                VALUES ('task1', 'user1', 'Test Task', 1.5, '2025-11-14', '2025-11-14')
            ''')
        
        conn.close()
    
    def test_fts_table_supports_full_text_search(self, migration, test_db_path):
        """测试: FTS表支持全文检索"""
        # Arrange
        migration.migrate_up()
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # 插入测试数据
        cursor.execute('''
            INSERT INTO conversations_fts (message_id, content)
            VALUES ('msg1', 'This is a test message about AGI')
        ''')
        cursor.execute('''
            INSERT INTO conversations_fts (message_id, content)
            VALUES ('msg2', 'Another message about machine learning')
        ''')
        
        # Act - 全文搜索
        cursor.execute('''
            SELECT message_id FROM conversations_fts 
            WHERE content MATCH 'AGI'
        ''')
        results = cursor.fetchall()
        
        # Assert
        assert len(results) == 1
        assert results[0][0] == 'msg1'
        
        conn.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
