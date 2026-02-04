#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试: 神经记忆生命周期管理器

测试覆盖:
- MemoryRecord 数据类
- MemoryLifecycleManager 核心功能
- 淘汰策略 (LRU, LFU, IMPORTANCE, HYBRID)
- 自动清理流程
- 状态持久化

作者: AGI System
日期: 2026-02-04
"""

import pytest
import time
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# 导入被测试模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.memory_lifecycle_manager import (
    MemoryRecord,
    MemoryLifecycleManager,
    EvictionPolicy,
)


class TestMemoryRecord:
    """测试 MemoryRecord 数据类"""

    def test_memory_record_creation(self):
        """测试记录创建"""
        record = MemoryRecord(
            id="test_001",
            timestamp=time.time(),
            last_accessed=time.time(),
            access_count=1,
            importance_score=0.8,
        )

        assert record.id == "test_001"
        assert record.access_count == 1
        assert record.importance_score == 0.8
        assert not record.compressed
        assert not record.archived

    def test_age_calculation(self):
        """测试年龄计算"""
        now = time.time()
        record = MemoryRecord(
            id="test_002",
            timestamp=now - 100,  # 100秒前创建
            last_accessed=now,
            access_count=1,
            importance_score=0.5,
        )

        age = record.age()
        assert 99 <= age <= 101  # 允许1秒误差

    def test_touch_method(self):
        """测试访问更新"""
        record = MemoryRecord(
            id="test_003",
            timestamp=time.time(),
            last_accessed=time.time() - 100,
            access_count=5,
            importance_score=0.5,
        )

        old_count = record.access_count
        old_access = record.last_accessed

        time.sleep(0.1)  # 等待100ms
        record.touch()

        assert record.access_count == old_count + 1
        assert record.last_accessed > old_access

    def test_expiration_check(self):
        """测试过期检查（通过比较年龄）"""
        # 模拟过期：年龄超过10秒
        old_record = MemoryRecord(
            id="expired",
            timestamp=time.time() - 100,  # 100秒前
            last_accessed=time.time() - 100,
            access_count=1,
            importance_score=0.5,
        )

        # 假设超过10秒就算"过期"（业务逻辑层面）
        is_expired = old_record.age() > 10

        assert is_expired

        # 未过期记录
        fresh_record = MemoryRecord(
            id="fresh",
            timestamp=time.time(),
            last_accessed=time.time(),
            access_count=1,
            importance_score=0.5,
        )

        is_fresh = fresh_record.age() < 10
        assert is_fresh

    def test_to_dict_conversion(self):
        """测试字典转换"""
        record = MemoryRecord(
            id="test_006",
            timestamp=time.time(),
            last_accessed=time.time(),
            access_count=3,
            importance_score=0.9,
            tags=["test", "unit"],
        )

        record_dict = record.to_dict()

        assert record_dict["id"] == "test_006"
        assert record_dict["access_count"] == 3
        assert record_dict["importance_score"] == 0.9
        assert record_dict["tags"] == ["test", "unit"]
        assert "age_seconds" in record_dict
        assert "access_age_seconds" in record_dict


class TestMemoryLifecycleManager:
    """测试 MemoryLifecycleManager 管理器"""

    @pytest.fixture
    def manager(self):
        """创建管理器实例"""
        manager = MemoryLifecycleManager(
            max_records=10,  # 小规模测试
            max_age_days=0.01,  # 约14分钟
            eviction_policy=EvictionPolicy.HYBRID,
            auto_cleanup_interval=5,  # 每5次操作触发清理
            compression_threshold=3,  # 超过3条开始压缩
        )
        return manager

    def test_manager_initialization(self, manager):
        """测试管理器初始化"""
        assert manager.max_records == 10
        assert manager.eviction_policy == EvictionPolicy.HYBRID
        assert manager.auto_cleanup_interval == 5
        assert len(manager.records) == 0
        assert manager.stats["total_added"] == 0

    def test_register_record(self, manager):
        """测试记录注册"""
        record = manager.register_record(
            memory_id="mem_001",
            importance_score=0.8,
            tags=["test", "important"],
        )

        assert record.id == "mem_001"
        assert record.importance_score == 0.8
        assert record.tags == ["test", "important"]
        assert "mem_001" in manager.records
        assert manager.stats["total_added"] == 1

    def test_touch_record(self, manager):
        """测试记录访问更新"""
        manager.register_record("mem_002", importance_score=0.5)
        original_count = manager.records["mem_002"].access_count

        # 更新访问
        touched = manager.touch_record("mem_002")

        assert touched is not None
        assert touched.access_count == original_count + 1

    def test_touch_nonexistent_record(self, manager):
        """测试访问不存在的记录"""
        result = manager.touch_record("nonexistent")
        assert result is None

    def test_auto_cleanup_trigger(self, manager):
        """测试自动清理触发"""
        # 注册5条记录（超过 auto_cleanup_interval=5）
        for i in range(5):
            manager.register_record(f"mem_{i:03d}", importance_score=0.5)

        # 验证清理被触发（operation_count 被重置）
        assert manager.operation_count < 5
        assert manager.stats["cleanup_runs"] > 0

    def test_eviction_lru(self):
        """测试 LRU 淘汰策略"""
        manager = MemoryLifecycleManager(
            max_records=3,
            eviction_policy=EvictionPolicy.LRU,
            auto_cleanup_interval=100,  # 禁用自动清理
        )

        # 添加3条记录
        manager.register_record("mem_a", importance_score=0.5)
        time.sleep(0.1)
        manager.register_record("mem_b", importance_score=0.5)
        time.sleep(0.1)
        manager.register_record("mem_c", importance_score=0.5)

        # 更新 mem_c 的访问时间（使其变为最近使用）
        manager.touch_record("mem_c")

        # 淘汰前：mem_a最老，mem_c最近
        # 手动淘汰1条
        evicted_count = manager.evict(1)
        assert evicted_count == 1

        # 验证：应该剩2条记录
        assert len(manager.records) == 2

        # mem_a 应该被淘汰（最久未使用）
        assert "mem_a" not in manager.records

        # mem_b 和 mem_c 应该保留
        assert "mem_b" in manager.records
        assert "mem_c" in manager.records

    def test_eviction_lfu(self):
        """测试 LFU 淘汰策略"""
        manager = MemoryLifecycleManager(
            max_records=3,
            eviction_policy=EvictionPolicy.LFU,
            auto_cleanup_interval=100,
        )

        # 添加记录并设置不同访问次数
        manager.register_record("mem_a", importance_score=0.5)
        manager.touch_record("mem_a")
        manager.touch_record("mem_a")  # mem_a: 3次访问

        manager.register_record("mem_b", importance_score=0.5)
        manager.touch_record("mem_b")  # mem_b: 2次访问

        manager.register_record("mem_c", importance_score=0.5)
        # mem_c: 只有1次访问

        # 手动淘汰1条
        evicted_count = manager.evict(1)
        assert evicted_count == 1

        # 验证：应该剩2条记录
        assert len(manager.records) == 2

        # mem_c 访问次数最少，应该被淘汰
        assert "mem_c" not in manager.records

        # mem_a 和 mem_b 应该保留（访问次数更多）
        assert "mem_a" in manager.records
        assert "mem_b" in manager.records

    def test_eviction_importance(self):
        """测试基于重要性的淘汰"""
        manager = MemoryLifecycleManager(
            max_records=3,
            eviction_policy=EvictionPolicy.IMPORTANCE,
            auto_cleanup_interval=100,
        )

        manager.register_record("mem_a", importance_score=0.3)
        manager.register_record("mem_b", importance_score=0.9)
        manager.register_record("mem_c", importance_score=0.6)

        # 手动淘汰1条
        evicted_count = manager.evict(1)
        assert evicted_count == 1

        # 验证：应该剩2条记录
        assert len(manager.records) == 2

        # mem_a 重要性最低(0.3)，应该被淘汰
        assert "mem_a" not in manager.records

        # mem_b 和 mem_c 应该保留（重要性更高）
        assert "mem_b" in manager.records
        assert "mem_c" in manager.records

    def test_compress_inactive(self):
        """测试不活跃记录压缩"""
        manager = MemoryLifecycleManager(
            compression_threshold=2,
        )

        # 添加旧记录（模拟不活跃）
        old_record = MemoryRecord(
            id="old_mem",
            timestamp=time.time() - 86400,  # 1天前
            last_accessed=time.time() - 3600,  # 1小时前访问
            access_count=5,
            importance_score=0.5,
        )
        manager.records["old_mem"] = old_record

        # 压缩不活跃记录
        compressed = manager.compress_inactive(threshold_days=0.001)  # 非常低的阈值

        assert compressed >= 1
        assert manager.records["old_mem"].compressed

    def test_archive_old(self):
        """测试归档超时记录"""
        manager = MemoryLifecycleManager(
            max_age_days=0.0001,  # 约8.6秒
        )

        # 添加超时记录
        old_record = MemoryRecord(
            id="ancient_mem",
            timestamp=time.time() - 100,  # 100秒前
            last_accessed=time.time() - 100,
            access_count=1,
            importance_score=0.5,
        )
        manager.records["ancient_mem"] = old_record

        # 执行归档
        archived = manager.archive_old()

        assert archived >= 1
        assert "ancient_mem" not in manager.records  # 已从活动记录中移除

    def test_calculate_importance(self):
        """测试重要性计算"""
        manager = MemoryLifecycleManager()

        # 测试不同类型的重要性
        macro_importance = manager.calculate_importance({"type": "macro"})
        episode_importance = manager.calculate_importance({"type": "episode"})
        tool_call_importance = manager.calculate_importance({
            "type": "tool_call",
            "tool": "rare_tool",  # 罕见工具
            "prototype_ids": ["p1", "p2", "p3"]
        })

        assert macro_importance > episode_importance  # macro 更重要
        assert tool_call_importance > 0.5  # 罕见工具加分

    def test_get_stats(self):
        """测试统计信息获取"""
        manager = MemoryLifecycleManager(max_records=100)

        # 添加一些记录
        for i in range(5):
            manager.register_record(f"mem_{i}", importance_score=0.5)

        stats = manager.get_stats()

        assert stats["total_records"] == 5
        assert stats["active_records"] == 5
        assert stats["max_records"] == 100
        assert stats["usage_ratio"] == 0.05
        # 移除 lifecycle_enabled 检查（BiologicalMemorySystem 有这个，但 Manager 自己的 stats 没有）
        # assert "lifecycle_enabled" in stats

    def test_export_records_for_cleanup(self):
        """测试清理记录导出"""
        import numpy as np

        manager = MemoryLifecycleManager(max_records=5)

        # 注册3条记录
        for i in range(3):
            manager.register_record(f"mem_{i}", importance_score=0.5)

        # 模拟记忆数组
        latents = np.random.randn(3, 64)
        metadata = [
            {"id": "mem_0", "type": "episode"},
            {"id": "mem_1", "type": "episode"},
            {"id": "mem_2", "type": "episode"},
        ]

        # 导出清理后的数组
        cleaned_latents, cleaned_metadata = manager.export_records_for_cleanup(
            latents, metadata
        )

        # 应该保持不变（所有记录都在生命周期管理器中）
        assert len(cleaned_latents) == 3
        assert len(cleaned_metadata) == 3


class TestStatePersistence:
    """测试状态持久化"""

    @pytest.fixture
    def temp_file(self):
        """创建临时文件"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name
        yield filepath
        # 清理
        if os.path.exists(filepath):
            os.unlink(filepath)

    def test_save_and_load_state(self, temp_file):
        """测试状态保存和加载"""
        manager = MemoryLifecycleManager(max_records=5)

        # 添加一些记录
        for i in range(3):
            manager.register_record(f"mem_{i}", importance_score=0.5)

        # 保存状态
        manager.save_state(temp_file)

        # 验证文件存在
        assert os.path.exists(temp_file)

        # 加载状态
        manager2 = MemoryLifecycleManager(max_records=5)
        manager2.load_state(temp_file)

        # 验证恢复成功
        assert len(manager2.records) == 3
        assert "mem_0" in manager2.records
        # stats.total_added 会在注册时累加，但加载后是新的
        # 所以这里验证记录数即可

    def test_load_nonexistent_file(self, temp_file):
        """测试加载不存在的文件"""
        os.unlink(temp_file)

        manager = MemoryLifecycleManager()
        # 不应该抛出异常
        manager.load_state(temp_file)

        # 应该是空状态
        assert len(manager.records) == 0


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_manager_eviction(self):
        """测试空管理器淘汰"""
        manager = MemoryLifecycleManager()
        evicted = manager.evict(5)
        assert evicted == 0

    def test_zero_importance_record(self):
        """测试零重要性记录"""
        manager = MemoryLifecycleManager()
        record = manager.register_record("zero_importance", importance_score=0.0)

        assert record.importance_score == 0.0
        assert "zero_importance" in manager.records

    def test_negative_ttl(self):
        """测试负年龄（边界情况）"""
        # 创建一个"年龄为负"的记录（时钟偏差模拟）
        record = MemoryRecord(
            id="test_005",
            timestamp=time.time() + 100,  # 未来时间（时钟偏差）
            last_accessed=time.time(),
            access_count=1,
            importance_score=0.5,
        )

        # 年龄可能是负数（如果时钟回拨）
        age = record.age()
        # 应该能处理
        assert isinstance(age, float)

    def test_concurrent_registration(self):
        """测试并发注册（简化版）"""
        manager = MemoryLifecycleManager(auto_cleanup_interval=100)  # 禁用自动清理

        # 快速注册多条记录
        records = []
        for i in range(20):
            record = manager.register_record(f"concurrent_{i}", importance_score=0.5)
            records.append(record)

        assert len(records) == 20
        assert len(manager.records) == 20


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
