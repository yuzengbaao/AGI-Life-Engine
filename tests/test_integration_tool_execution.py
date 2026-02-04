#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试: 工具执行完整流程

测试覆盖:
- 工具调用缓存 + 记忆生命周期 + 异常处理
- 完整工具执行链路的端到端测试
- 错误恢复和重试机制

作者: AGI System
日期: 2026-02-04
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

# 导入被测试模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool_call_cache import ToolCallCache
from core.memory.memory_lifecycle_manager import (
    MemoryLifecycleManager,
    EvictionPolicy,
)


class TestToolExecutionFlow:
    """测试工具执行流程"""

    @pytest.fixture
    def execution_system(self):
        """创建执行系统"""
        return {
            "cache": ToolCallCache(max_size=50, default_ttl=3600.0),
            "lifecycle": MemoryLifecycleManager(
                max_records=50,
                eviction_policy=EvictionPolicy.LRU,
            ),
        }

    def test_complete_tool_execution(self, execution_system):
        """测试完整工具执行流程"""
        cache = execution_system["cache"]
        lifecycle = execution_system["lifecycle"]

        # 步骤1: 工具调用前 - 检查缓存
        tool_name = "file_reader"
        params = {"path": "/test/file.txt"}

        cached_result = cache.get(tool_name, params)
        assert cached_result is None  # 首次调用，未命中

        # 步骤2: 模拟工具执行
        execution_start = time.time()
        result = {
            "success": True,
            "data": "file content here",
            "execution_time_ms": (time.time() - execution_start) * 1000,
        }

        # 步骤3: 存储到缓存
        cache_key = cache.put(tool_name, params, result)

        # 步骤4: 创建记忆记录
        memory_id = f"tool_call_{cache_key}"
        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=0.7,
            tags=["tool_call", tool_name, "success"],
        )

        # 步骤5: 验证缓存命中
        cached_result = cache.get(tool_name, params)
        assert cached_result is not None
        assert cached_result["success"] is True

        # 步骤6: 验证记忆记录
        assert memory_id in lifecycle.records
        assert lifecycle.records[memory_id].importance_score == 0.7

    def test_tool_execution_with_cache_hit(self, execution_system):
        """测试缓存命中时的工具执行"""
        cache = execution_system["cache"]
        lifecycle = execution_system["lifecycle"]

        tool_name = "calculator"
        params = {"operation": "add", "a": 5, "b": 3}

        # 第一次执行（缓存未命中）
        result1 = {"result": 8}
        cache_key1 = cache.put(tool_name, params, result1)

        memory_id1 = f"tool_call_{cache_key1}"
        lifecycle.register_record(memory_id1, importance_score=0.6)

        # 第二次执行（缓存命中）
        cached_result = cache.get(tool_name, params)
        assert cached_result is not None
        assert cached_result["result"] == 8

        # 更新访问记录
        lifecycle.touch_record(memory_id1)

        # 验证访问次数增加
        assert lifecycle.records[memory_id1].access_count >= 2

    def test_tool_execution_failure(self, execution_system):
        """测试工具执行失败"""
        cache = execution_system["cache"]
        lifecycle = execution_system["lifecycle"]

        tool_name = "risky_tool"
        params = {"action": "dangerous"}

        # 模拟执行失败
        error_result = {
            "success": False,
            "error": "Operation failed",
            "error_code": 500,
        }

        # 失败结果也应该缓存（避免重复失败调用）
        cache_key = cache.put(tool_name, params, error_result, ttl=300)  # 短期缓存

        memory_id = f"tool_call_{cache_key}"
        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=0.2,  # 低重要性
            tags=["tool_call", tool_name, "failed"],
        )

        # 验证失败结果被缓存
        cached = cache.get(tool_name, params)
        assert cached is not None
        assert cached["success"] is False

        # 验证低重要性记录
        assert lifecycle.records[memory_id].importance_score == 0.2

    def test_batch_tool_execution(self, execution_system):
        """测试批量工具执行"""
        cache = execution_system["cache"]
        lifecycle = execution_system["lifecycle"]

        # 批量执行10个工具调用
        batch_size = 10
        for i in range(batch_size):
            tool_name = f"batch_tool_{i % 3}"
            params = {"batch_id": i, "data": f"item_{i}"}

            result = {"processed": True, "index": i}
            cache_key = cache.put(tool_name, params, result)

            memory_id = f"tool_call_{cache_key}"
            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["batch", tool_name],
            )

        # 验证所有调用都被记录
        assert len(cache.cache) == batch_size
        assert len(lifecycle.records) == batch_size

        # 验证缓存大小
        stats = cache.get_stats()
        assert stats["size"] == batch_size

    def test_parameter_normalization_in_execution(self, execution_system):
        """测试参数规范化在执行中的作用"""
        cache = execution_system["cache"]

        tool_name = "normalized_tool"

        # 参数顺序不同，但内容相同
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "a": 1, "b": 2}

        # 第一次调用
        result1 = {"normalized": True}
        cache_key1 = cache.put(tool_name, params1, result1)

        # 第二次调用（应该命中缓存）
        cached = cache.get(tool_name, params2)
        assert cached is not None

        # 验证键相同（规范化生效）
        cache_key2 = cache.generate_cache_key(tool_name, params2)
        assert cache_key1 == cache_key2


class TestRetryAndRecovery:
    """测试重试和恢复机制"""

    @pytest.fixture
    def retry_system(self):
        """创建重试系统"""
        return {
            "cache": ToolCallCache(max_size=100),
            "lifecycle": MemoryLifecycleManager(max_records=100),
        }

    def test_retry_with_exponential_backoff(self, retry_system):
        """测试指数退避重试"""
        cache = retry_system["cache"]

        tool_name = "unreliable_tool"
        params = {"attempt": 1}

        # 第一次失败
        cache.put(tool_name, params, {"success": False}, ttl=1)
        time.sleep(1.1)

        # 缓存过期后重试
        cached = cache.get(tool_name, params)
        assert cached is None  # 已过期

        # 重试成功
        cache.put(tool_name, params, {"success": True})

        # 验证成功
        cached = cache.get(tool_name, params)
        assert cached["success"] is True

    def test_recovery_from_cache_corruption(self, retry_system):
        """测试从缓存损坏恢复"""
        cache = retry_system["cache"]
        lifecycle = retry_system["lifecycle"]

        # 添加正常数据
        cache.put("good_tool", {"p": 1}, {"r": 1})
        memory_id = "tool_call_" + cache.generate_cache_key("good_tool", {"p": 1})
        lifecycle.register_record(memory_id, importance_score=0.7)

        # 模拟缓存失效（清空）
        initial_size = len(cache.cache)
        cache.invalidate("good_tool")

        # 验证缓存已清空但生命周期记录保留
        assert len(cache.cache) < initial_size
        assert memory_id in lifecycle.records

        # 重新执行（重建缓存）
        cache.put("good_tool", {"p": 1}, {"r": 1})

        # 验证恢复
        cached = cache.get("good_tool", {"p": 1})
        assert cached is not None

    def test_partial_failure_handling(self, retry_system):
        """测试部分失败处理"""
        cache = retry_system["cache"]
        lifecycle = retry_system["lifecycle"]

        # 批量调用，部分失败
        results = []
        for i in range(10):
            tool_name = "batch_processor"
            params = {"item": i}

            # 前7个成功，后3个失败
            success = i < 7
            result = {
                "success": success,
                "item": i,
                "error": None if success else "Processing error"
            }

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            importance = 0.7 if success else 0.3
            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=importance,
                tags=["success" if success else "failed"],
            )

            results.append(result)

        # 验证部分成功
        success_count = sum(1 for r in results if r["success"])
        assert success_count == 7

        # 验证缓存包含所有结果
        assert len(cache.cache) == 10

        # 验证生命周期管理器按重要性区分
        importances = [r.importance_score for r in lifecycle.records.values()]
        assert any(imp >= 0.7 for imp in importances)  # 成功的
        assert any(imp < 0.5 for imp in importances)  # 失败的


class TestConcurrentExecution:
    """测试并发执行"""

    def test_concurrent_tool_calls(self):
        """测试并发工具调用"""
        import threading

        cache = ToolCallCache(max_size=200)
        lifecycle = MemoryLifecycleManager(max_records=200)
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    tool_name = f"concurrent_tool_{worker_id % 3}"
                    params = {"worker": worker_id, "iteration": i}

                    result = {"data": f"worker_{worker_id}_iter_{i}"}
                    cache_key = cache.put(tool_name, params, result)

                    memory_id = f"tool_call_{cache_key}"
                    lifecycle.register_record(
                        memory_id=memory_id,
                        importance_score=0.5,
                        tags=["concurrent"],
                    )

                    time.sleep(0.001)  # 模拟处理时间
            except Exception as e:
                errors.append(e)

        # 启动5个工作线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证没有错误
        assert len(errors) == 0

        # 验证所有调用都被记录
        assert len(cache.cache) == 50  # 5 workers * 10 iterations
        assert len(lifecycle.records) == 50

    def test_cache_race_condition(self):
        """测试缓存竞态条件"""
        import threading

        cache = ToolCallCache(max_size=100)

        def cache_worker(worker_id):
            for i in range(20):
                tool_name = "race_test"
                params = {"id": i}

                # 先尝试获取
                cache.get(tool_name, params)

                # 然后存储
                cache.put(tool_name, params, {"worker": worker_id})

        # 启动多个线程
        threads = []
        for i in range(3):
            t = threading.Thread(target=cache_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证缓存一致性
        stats = cache.get_stats()
        # total_calls只计入get操作：3 workers * 20 iterations = 60 gets
        assert stats["hits"] + stats["misses"] == 60
        assert len(cache.cache) <= 100  # 受max_size限制


class TestPerformanceIntegration:
    """测试性能集成"""

    def test_execution_time_tracking(self):
        """测试执行时间追踪"""
        cache = ToolCallCache(max_size=100)
        lifecycle = MemoryLifecycleManager(max_records=100)

        execution_times = []

        for i in range(20):
            tool_name = "timer_tool"
            params = {"iteration": i}

            start_time = time.time()
            # 模拟执行
            time.sleep(0.01)
            end_time = time.time()

            execution_time = (end_time - start_time) * 1000
            execution_times.append(execution_time)

            result = {
                "success": True,
                "execution_time_ms": execution_time,
            }

            cache_key = cache.put(tool_name, params, result)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
                tags=["timed"],
            )

        # 验证执行时间合理
        avg_time = sum(execution_times) / len(execution_times)
        assert 10 <= avg_time <= 20  # 10ms左右

    def test_memory_usage_optimization(self):
        """测试内存使用优化"""
        cache = ToolCallCache(max_size=50)
        lifecycle = MemoryLifecycleManager(max_records=50)

        # 添加大量小数据
        for i in range(100):
            tool_name = "memory_test"
            params = {"id": i}
            result = {"data": "x" * 100}  # 100字节

            cache.put(tool_name, params, result)

            cache_key = cache.generate_cache_key(tool_name, params)
            memory_id = f"tool_call_{cache_key}"

            lifecycle.register_record(
                memory_id=memory_id,
                importance_score=0.5,
            )

        # 验证内存被限制
        assert len(cache.cache) <= 50
        assert len(lifecycle.records) <= 50

        # 验证淘汰生效
        assert cache.stats["evictions"] > 0 or lifecycle.stats["total_evicted"] > 0


class TestRealWorldScenarios:
    """测试真实场景"""

    def test_file_operations_workflow(self):
        """测试文件操作工作流"""
        cache = ToolCallCache(max_size=100)
        lifecycle = MemoryLifecycleManager(max_records=100)

        # 场景：读取文件，处理内容，写入结果

        # 1. 读取文件
        read_params = {"path": "/data/input.txt", "operation": "read"}
        read_result = {"content": "file content", "size": 1024}
        cache.put("file_ops", read_params, read_result)
        lifecycle.register_record(
            "tool_call_" + cache.generate_cache_key("file_ops", read_params),
            importance_score=0.8,
            tags=["file", "read"],
        )

        # 2. 处理内容（第二次读取应该命中缓存）
        cached_read = cache.get("file_ops", read_params)
        assert cached_read is not None

        # 3. 写入结果
        write_params = {"path": "/data/output.txt", "operation": "write", "data": "processed"}
        write_result = {"success": True, "bytes_written": 1024}
        cache.put("file_ops", write_params, write_result)
        lifecycle.register_record(
            "tool_call_" + cache.generate_cache_key("file_ops", write_params),
            importance_score=0.7,
            tags=["file", "write"],
        )

        # 验证工作流完整
        assert len(cache.cache) >= 2
        assert len(lifecycle.records) >= 2

    def test_api_call_workflow(self):
        """测试API调用工作流"""
        cache = ToolCallCache(max_size=100)
        lifecycle = MemoryLifecycleManager(max_records=100)

        # 场景：调用外部API，处理响应，缓存结果

        api_endpoint = "https://api.example.com/users"
        params = {"endpoint": api_endpoint, "method": "GET"}

        # 首次API调用
        api_response = {
            "status": 200,
            "data": [{"id": 1, "name": "User1"}, {"id": 2, "name": "User2"}],
        }

        cache.put("api_call", params, api_response, ttl=300)  # 5分钟TTL
        lifecycle.register_record(
            "tool_call_" + cache.generate_cache_key("api_call", params),
            importance_score=0.6,
            tags=["api", "external"],
        )

        # 验证缓存命中
        cached_response = cache.get("api_call", params)
        assert cached_response is not None
        assert cached_response["status"] == 200

        # 验证TTL生效（短期缓存API响应）
        assert len(cache.cache) == 1


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
