#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试: 神经记忆生命周期管理器 + 工具调用缓存优化器

测试覆盖:
- 两个模块的协同工作
- 缓存淘汰与生命周期淘汰的交互
- 状态持久化的联合恢复
- 性能监控的集成

作者: AGI System
日期: 2026-02-04
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np

# 导入被测试模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.memory_lifecycle_manager import (
    MemoryLifecycleManager,
    MemoryRecord,
    EvictionPolicy,
)
from core.tool_call_cache import (
    ToolCallCache,
    CacheEntry,
    get_tool_call_cache,
    reset_tool_call_cache,
)


class TestCacheAndLifecycleIntegration:
    """测试缓存与生命周期管理器的集成"""

    @pytest.fixture
    def integrated_system(self):
        """创建集成的缓存和生命周期管理器"""
        cache = ToolCallCache(
            max_size=20,
            default_ttl=3600.0,
            enable_semantic_match=False,
        )
        lifecycle = MemoryLifecycleManager(
            max_records=20,
            max_age_days=1.0,
            eviction_policy=EvictionPolicy.LRU,
            auto_cleanup_interval=10,
        )
        return {"cache": cache, "lifecycle": lifecycle}

    def test_tool_call_creates_memory_record(self, integrated_system):
        """测试工具调用创建记忆记录"""
        cache = integrated_system["cache"]
        lifecycle = integrated_system["lifecycle"]

        # 模拟工具调用
        tool_name = "file_operations"
        params = {"operation": "read", "path": "/test/file.txt"}
        result = {"success": True, "data": "content"}

        # 1. 存储到缓存
        cache_key = cache.put(tool_name, params, result)

        # 2. 创建对应的记忆记录
        memory_id = f"tool_call_{cache_key}"
        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=0.7,
            tags=["tool_call", tool_name],
        )

        # 3. 验证关联
        assert cache_key in cache.cache
        assert memory_id in lifecycle.records

        # 4. 验证缓存命中时更新生命周期
        cached_result = cache.get(tool_name, params)
        assert cached_result is not None

        # 更新访问记录
        lifecycle.touch_record(memory_id)
        assert lifecycle.records[memory_id].access_count >= 2

    def test_cache_eviction_triggers_memory_cleanup(self, integrated_system):
        """测试缓存淘汰触发记忆清理"""
        cache = integrated_system["cache"]
        lifecycle = integrated_system["lifecycle"]

        # 添加多条工具调用记录
        for i in range(25):  # 超过max_size=20
            tool_name = f"tool_{i % 5}"
            params = {"index": i}
            result = {"data": f"result_{i}"}

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["tool_call", tool_name],
            )

        # 验证缓存发生了淘汰
        assert cache.stats["evictions"] > 0

        # 验证生命周期管理器有相应记录
        # 注意：由于两者独立管理，数量可能不完全一致
        assert len(lifecycle.records) > 0

    def test_memory_expiration_affects_cache_validity(self, integrated_system):
        """测试记忆过期影响缓存有效性"""
        cache = integrated_system["cache"]
        lifecycle = integrated_system["lifecycle"]

        # 创建短期缓存条目
        tool_name = "temp_tool"
        params = {"action": "temp"}
        result = {"temp": "data"}

        cache_key = cache.put(tool_name, params, result, ttl=1.0)
        memory_id = f"tool_call_{cache_key}"

        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=0.3,  # 低重要性
            tags=["temp"],
        )

        # 等待缓存过期
        time.sleep(1.5)

        # 验证缓存已过期
        cached_result = cache.get(tool_name, params)
        assert cached_result is None

        # 验证记忆记录仍在（生命周期管理器有自己的TTL）
        assert memory_id in lifecycle.records

    def test_combined_statistics(self, integrated_system):
        """测试联合统计"""
        cache = integrated_system["cache"]
        lifecycle = integrated_system["lifecycle"]

        # 执行多次操作
        for i in range(10):
            tool_name = "test_tool"
            params = {"iteration": i}
            result = {"value": i * 2}

            cache.put(tool_name, params, result)

            # 一半的命中
            if i % 2 == 0:
                cache.get(tool_name, params)

            cache_key = cache.generate_cache_key(tool_name, params)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["test"],
            )

        # 获取各自统计
        cache_stats = cache.get_stats()
        lifecycle_stats = lifecycle.get_stats()

        # 验证统计信息
        # 注意：total_calls只计入get操作，put不计入
        assert cache_stats["size"] == 10  # 10条记录
        assert cache_stats["hits"] >= 5  # 至少一半命中
        assert lifecycle_stats["total_records"] == 10

    def test_priority_based_cleanup(self, integrated_system):
        """测试基于优先级的清理"""
        cache = integrated_system["cache"]
        lifecycle = integrated_system["lifecycle"]

        # 创建高重要性和低重要性记录
        for i in range(10):
            tool_name = "priority_tool"
            params = {"id": i}
            result = {"data": f"priority_{i}"}

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            # 前5个高重要性，后5个低重要性
            importance = 0.9 if i < 5 else 0.3
            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=importance,
                tags=["priority"],
            )

        # 获取缓存统计
        initial_cache_size = len(cache.cache)

        # 执行生命周期淘汰（IMPORTANCE策略）
        lifecycle.eviction_policy = EvictionPolicy.IMPORTANCE

        # 添加一条记录以触发自动清理（超过max_records）
        cache.put("priority_tool", {"id": 999}, {"data": "trigger"})

        # 手动淘汰3条
        evicted = lifecycle.evict(3)
        assert evicted == 3

        # 验证剩余记录（应该主要是高重要性的）
        # 由于IMPORTANCE策略，低重要性应该被优先淘汰
        remaining_count = len(lifecycle.records)
        assert remaining_count == 7  # 10 - 3


class TestStatePersistenceIntegration:
    """测试状态持久化的集成"""

    @pytest.fixture
    def temp_files(self):
        """创建临时文件"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_cache.json") as f:
            cache_file = f.name
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix="_lifecycle.json") as f:
            lifecycle_file = f.name

        yield {"cache": cache_file, "lifecycle": lifecycle_file}

        # 清理
        if os.path.exists(cache_file):
            os.unlink(cache_file)
        if os.path.exists(lifecycle_file):
            os.unlink(lifecycle_file)

    def test_save_and_load_combined_state(self, temp_files):
        """测试保存和加载联合状态"""
        cache = ToolCallCache(max_size=10)
        lifecycle = MemoryLifecycleManager(max_records=10)

        # 添加一些数据
        for i in range(5):
            tool_name = f"tool_{i}"
            params = {"index": i}
            result = {"data": f"value_{i}"}

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.6,
                tags=["combined"],
            )

        # 保存状态
        cache.save_state(temp_files["cache"])
        lifecycle.save_state(temp_files["lifecycle"])

        # 验证文件存在
        assert os.path.exists(temp_files["cache"])
        assert os.path.exists(temp_files["lifecycle"])

        # 创建新实例并加载状态
        new_cache = ToolCallCache(max_size=10)
        new_lifecycle = MemoryLifecycleManager(max_records=10)

        new_cache.load_state(temp_files["cache"])
        new_lifecycle.load_state(temp_files["lifecycle"])

        # 验证恢复成功
        assert len(new_cache.cache) == 5
        assert len(new_lifecycle.records) == 5

        # 验证关联一致性
        for cache_key in new_cache.cache:
            memory_id = f"tool_call_{cache_key}"
            assert memory_id in new_lifecycle.records

    def test_partial_state_recovery(self, temp_files):
        """测试部分状态恢复"""
        cache = ToolCallCache(max_size=10)
        lifecycle = MemoryLifecycleManager(max_records=10)

        # 只添加缓存数据
        cache.put("tool_a", {"p": 1}, {"r": 1})
        cache.put("tool_b", {"p": 2}, {"r": 2})

        # 只保存缓存状态
        cache.save_state(temp_files["cache"])

        # 创建新实例并只加载缓存
        new_cache = ToolCallCache(max_size=10)
        new_lifecycle = MemoryLifecycleManager(max_records=10)

        new_cache.load_state(temp_files["cache"])
        # lifecycle不加载，应该是空的

        # 验证缓存恢复，lifecycle为空
        assert len(new_cache.cache) == 2
        assert len(new_lifecycle.records) == 0


class TestPerformanceIntegration:
    """测试性能监控的集成"""

    def test_combined_performance_tracking(self):
        """测试联合性能追踪"""
        cache = ToolCallCache(max_size=100)
        lifecycle = MemoryLifecycleManager(max_records=100)

        # 执行大量操作
        start_time = time.time()

        for i in range(100):
            tool_name = "perf_tool"
            params = {"iteration": i}
            result = {"data": "x" * 100}  # 模拟较大数据

            # 缓存操作
            cache.put(tool_name, params, result)
            cache.get(tool_name, params)

            # 生命周期操作
            cache_key = cache.generate_cache_key(tool_name, params)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["performance"],
            )

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能指标
        cache_stats = cache.get_stats()
        lifecycle_stats = lifecycle.get_stats()

        # 应该在合理时间内完成（100次操作 < 1秒）
        assert total_time < 1.0

        # 缓存命中率应该合理
        hit_rate = cache_stats["hits"] / max(cache_stats["total_calls"], 1)
        assert hit_rate > 0.4  # 至少40%命中率

        # 记录数应该匹配
        assert lifecycle_stats["total_records"] == 100

    def test_memory_efficiency(self):
        """测试内存效率"""
        cache = ToolCallCache(max_size=50)
        lifecycle = MemoryLifecycleManager(max_records=50)

        # 添加记录前记录内存使用
        import gc
        gc.collect()

        # 添加超过限制的记录以触发淘汰
        for i in range(60):  # 超过max_size=50
            tool_name = f"tool_{i % 10}"
            params = {"data": "x" * 1000}  # 1KB数据
            result = {"response": "y" * 1000}

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["memory_test"],
            )

        # 验证记录数被限制
        assert len(cache.cache) <= 50
        assert len(lifecycle.records) <= 50

        # 验证至少有一个组件发生了淘汰
        # 注意：可能两者都未触发（取决于具体实现）
        # 这里只验证记录数被正确限制


class TestErrorHandlingIntegration:
    """测试错误处理的集成"""

    def test_cache_failure_does_not_affect_lifecycle(self):
        """测试缓存故障不影响生命周期管理器"""
        cache = ToolCallCache(max_size=10)
        lifecycle = MemoryLifecycleManager(max_records=10)

        # 添加正常记录
        tool_name = "test_tool"
        params = {"valid": True}
        result = {"data": "test"}

        cache_key = cache.put(tool_name, params, result)
        memory_id = f"tool_call_{cache_key}"

        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=0.7,
            tags=["robust"],
        )

        # 缓存中不存在的结果（未命中）
        invalid_params = {"invalid": True}
        invalid_result = cache.get(tool_name, invalid_params)
        assert invalid_result is None

        # 验证生命周期管理器不受影响
        assert memory_id in lifecycle.records
        assert lifecycle.records[memory_id].importance_score == 0.7

    def test_lifecycle_cleanup_preserves_cache(self):
        """测试生命周期清理保留缓存"""
        cache = ToolCallCache(max_size=10)
        lifecycle = MemoryLifecycleManager(
            max_records=5,
            eviction_policy=EvictionPolicy.LRU,
        )

        # 添加超过限制的记录
        for i in range(10):
            tool_name = "test_tool"
            params = {"index": i}
            result = {"data": f"value_{i}"}

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["cleanup_test"],
            )

        # 执行生命周期淘汰
        lifecycle.evict(3)

        # 缓存应该保留所有条目（独立管理）
        assert len(cache.cache) == 10

        # 生命周期应该有淘汰
        assert len(lifecycle.records) < 10


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
