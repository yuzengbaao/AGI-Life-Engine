#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充测试：提升代码覆盖率

目标：将覆盖率从85%提升到90%+
重点：错误处理路径、边界情况、未覆盖的代码路径

作者: AGI System
日期: 2026-02-04
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool_call_cache import ToolCallCacheOptimized, OptimizedCacheEntry
from core.memory.memory_lifecycle_manager import MemoryLifecycleManager, EvictionPolicy


class TestOptimizedCacheEdgeCases:
    """优化版本缓存边界情况测试"""

    def test_cache_initialization_with_params(self):
        """测试缓存初始化参数"""
        cache = ToolCallCacheOptimized(
            max_size=500,
            default_ttl=1800.0,
            lru_update_interval=5
        )
        assert cache.max_size == 500
        assert cache.default_ttl == 1800.0
        assert cache.lru_update_interval == 5

    def test_batch_increment(self):
        """测试批次号递增"""
        cache = ToolCallCacheOptimized(batch_size=10)

        # 初始批次
        assert cache.current_batch == 0

        # 执行9次操作
        for i in range(9):
            cache.put(f"tool_{i}", {}, {"result": i})

        # 批次号应该还是0
        assert cache.current_batch == 0

        # 第10次操作应该递增批次号
        cache.put("tool_10", {}, {"result": 10})
        assert cache.current_batch == 1

    def test_key_cache_hit(self):
        """测试键缓存命中"""
        cache = ToolCallCacheOptimized()

        # 第一次调用 - 未命中
        params = {"key": "value"}
        key1 = cache.generate_cache_key("test_tool", params)

        assert cache.stats["key_cache_hits"] == 0

        # 第二次调用 - 应该命中
        key2 = cache.generate_cache_key("test_tool", params)

        assert cache.stats["key_cache_hits"] == 1
        assert key1 == key2

    def test_lru_skip_tracking(self):
        """测试LRU更新跳过统计"""
        cache = ToolCallCacheOptimized(lru_update_interval=3)

        params = {"key": "value"}
        cache.put("tool", params, {"result": "data"})

        # 第1次访问 - access_count=1, 1%3=1 != 0, 应该更新LRU
        cache.get("tool", params)
        assert cache.stats.get("lru_skips", 0) == 0

        # 第2次访问 - access_count=2, 2%3=2 != 0, 应该更新LRU
        cache.get("tool", params)
        assert cache.stats.get("lru_skips", 0) == 0

        # 第3次访问 - access_count=3, 3%3=0, 应该跳过LRU更新
        cache.get("tool", params)
        assert cache.stats.get("lru_skips", 0) == 1

        # 第4次访问 - access_count=4, 4%3=1 != 0, 应该更新LRU
        cache.get("tool", params)
        assert cache.stats.get("lru_skips", 0) == 1

    def test_optimized_entry_creation(self):
        """测试优化条目创建"""
        entry = OptimizedCacheEntry(
            cache_key="test_key",
            tool_name="test_tool",
            params={"param": "value"},
            result={"output": "data"},
            timestamp=time.time(),
            last_accessed_batch=5,
            access_count=10,
            importance_score=0.8
        )

        assert entry.cache_key == "test_key"
        assert entry.last_accessed_batch == 5
        assert entry.access_count == 10

    def test_touch_batch_method(self):
        """测试批次touch方法"""
        entry = OptimizedCacheEntry(
            cache_key="test_key",
            tool_name="test_tool",
            params={},
            result={},
            timestamp=time.time(),
            last_accessed_batch=0,
            access_count=0
        )

        # 使用批次10进行touch
        entry.touch_batch(10)

        assert entry.last_accessed_batch == 10
        assert entry.access_count == 1

    def test_is_expired_batch_method(self):
        """测试批次过期检查"""
        entry = OptimizedCacheEntry(
            cache_key="test_key",
            tool_name="test_tool",
            params={},
            result={},
            timestamp=time.time(),
            last_accessed_batch=0,
            access_count=0,
            ttl=100  # 100批次
        )

        # 当前批次50 - 未过期
        assert not entry.is_expired_batch(50)

        # 当前批次200 - 已过期
        assert entry.is_expired_batch(200)


class TestMemoryLifecycleEdgeCases:
    """记忆生命周期管理边界情况测试"""

    def test_manager_with_custom_batch_size(self):
        """测试自定义批次大小"""
        manager = MemoryLifecycleManager(
            max_records=100,
            auto_cleanup_interval=50
        )

        # 添加50个记录
        for i in range(50):
            manager.register_record(f"mem_{i}")

        # 应该触发自动清理
        assert manager.operation_count == 0  # 重置后

    def test_eviction_with_empty_cache(self):
        """测试空缓存淘汰"""
        manager = MemoryLifecycleManager()

        # 尝试从空缓存淘汰
        evicted = manager.evict(10)

        assert evicted == 0

    def test_compress_inactive_with_threshold(self):
        """测试压缩不活跃记录"""
        manager = MemoryLifecycleManager()

        # 添加一个旧记录
        old_record = manager.register_record(
            "old_mem",
            importance_score=0.5
        )
        old_record.timestamp = time.time() - (10 * 86400)  # 10天前
        old_record.last_accessed = time.time() - (10 * 86400)

        # 添加一个新记录
        new_record = manager.register_record(
            "new_mem",
            importance_score=0.5
        )

        # 压缩7天以上不活跃的记录
        compressed = manager.compress_inactive(threshold_days=7)

        assert compressed >= 1
        assert old_record.compressed
        assert not new_record.compressed

    def test_importance_score_calculation(self):
        """测试重要性评分计算"""
        manager = MemoryLifecycleManager()

        # 测试macro类型
        macro_score = manager.calculate_importance({
            "type": "macro",
            "skill": "python",
            "tool": "rare_tool",
            "prototype_ids": ["p1", "p2", "p3"]
        })

        assert macro_score > 0.5

        # 测试episode类型
        episode_score = manager.calculate_importance({
            "type": "episode"
        })

        assert episode_score == 0.5  # 基础分

    def test_get_stats_detailed(self):
        """测试获取详细统计"""
        manager = MemoryLifecycleManager(
            max_records=100,
            eviction_policy=EvictionPolicy.LRU
        )

        # 添加一些记录
        for i in range(10):
            manager.register_record(f"mem_{i}")

        stats = manager.get_stats()

        assert stats["total_records"] == 10
        assert stats["max_records"] == 100
        assert stats["eviction_policy"] == "least_recently_used"
        assert "usage_ratio" in stats


class TestErrorHandlingPaths:
    """错误处理路径测试"""

    def test_cache_get_with_invalid_params(self):
        """测试缓存获取时参数为None"""
        cache = ToolCallCacheOptimized()

        # 空参数字典
        result = cache.get("tool", {})
        assert result is None

        # None参数
        result = cache.get("tool", None)
        assert result is None

    def test_cache_put_with_none_ttl(self):
        """测试缓存存储时使用默认TTL"""
        cache = ToolCallCacheOptimized(default_ttl=100.0)

        # 不指定TTL，使用默认值
        cache.put("tool", {"key": "value"}, {"result": "data"})

        entry = cache.cache.get("tool_tool_")
        assert entry is not None
        assert entry.ttl == 100.0

    def test_memory_load_state_from_invalid_file(self):
        """测试从无效文件加载状态"""
        manager = MemoryLifecycleManager()

        # 尝试从不存在的文件加载
        manager.load_state("nonexistent_file.json")

        # 应该不抛出异常，只是记录警告
        assert len(manager.records) == 0

    def test_cache_invalidate_nonexistent_tool(self):
        """测试失效不存在的工具"""
        cache = ToolCallCacheOptimized()

        # 添加一些记录
        cache.put("tool1", {}, {"result": "data1"})
        cache.put("tool2", {}, {"result": "data2"})

        # 失效不存在的工具应该不影响其他工具
        cache.invalidate("nonexistent_tool")

        assert len(cache.cache) == 2


class TestBoundaryConditions:
    """边界条件测试"""

    def test_cache_with_zero_max_size(self):
        """测试零大小缓存"""
        cache = ToolCallCacheOptimized(max_size=0)

        # 尝试添加记录应该立即淘汰
        cache.put("tool", {}, {"result": "data"})

        # 记录应该被淘汰
        assert len(cache.cache) == 0

    def test_cache_with_very_small_max_size(self):
        """测试非常小的缓存"""
        cache = ToolCallCacheOptimized(max_size=2)

        # 添加3个记录，第3个应该淘汰第1个
        cache.put("tool1", {}, {"result": "data1"})
        cache.put("tool2", {}, {"result": "data2"})
        cache.put("tool3", {}, {"result": "data3"})

        assert len(cache.cache) == 2
        assert "tool1_tool_" not in cache.cache
        assert "tool2_tool_" in cache.cache
        assert "tool3_tool_" in cache.cache

    def test_memory_with_zero_importance(self):
        """测试零重要性记录"""
        manager = MemoryLifecycleManager()

        record = manager.register_record(
            "zero_importance",
            importance_score=0.0
        )

        assert record.importance_score == 0.0

    def test_memory_cleanup_empty_records(self):
        """测试清理空记录"""
        manager = MemoryLifecycleManager()

        # 清理空记录
        result = manager.auto_cleanup()

        assert result["before_count"] == 0
        assert result["after_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
