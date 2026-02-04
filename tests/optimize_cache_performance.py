#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache GET性能优化分析

分析当前性能问题并提出优化方案

问题: Cache GET(hit) 平均 0.631ms，比 GET(miss) 的 0.002ms 慢 300 倍

根本原因分析：
1. 每次GET(hit)都执行 time.time() (在 entry.touch() 中)
2. 每次GET(hit)都执行 OrderedDict.move_to_end() 更新LRU顺序
3. 键生成虽然快(0.002ms)但在100条记录的缓存中查找仍有开销

优化方案：
1. 延迟时间戳更新（批量更新）
2. 减少move_to_end调用频率
3. 优化哈希算法（如果需要）

作者: AGI System
日期: 2026-02-04
"""

import sys
import time
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, replace

# 设置Windows控制台编码
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class OptimizedCacheEntry:
    """优化的缓存条目"""
    cache_key: str
    tool_name: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: float
    last_accessed_batch: int  # 批次号（代替精确时间戳）
    access_count: int
    ttl: float = 3600.0

    def age(self) -> float:
        """缓存年龄（秒）"""
        return time.time() - self.timestamp

    def is_expired(self, batch_timestamp: int) -> bool:
        """检查是否过期（使用批次号）"""
        # 简化：假设每个批次约1秒，检查是否超过TTL批次
        age_batches = batch_timestamp - self.last_accessed_batch
        return age_batches > self.ttl

    def touch_batch(self, current_batch: int):
        """批次更新（不调用time.time()）"""
        self.last_accessed_batch = current_batch
        self.access_count += 1


class OptimizedToolCallCache:
    """
    优化的工具调用缓存器

    优化点：
    1. 延迟时间戳更新：使用批次号代替精确时间戳
    2. 减少move_to_end调用：每N次才更新LRU顺序
    3. 优化键生成：缓存已生成的键
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        lru_update_interval: int = 10,  # 每N次命中才更新LRU顺序
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lru_update_interval = lru_update_interval

        self.cache: OrderedDict[str, OptimizedCacheEntry] = OrderedDict()
        self.key_cache: Dict[tuple, str] = {}  # 缓存生成的键

        # 批次管理
        self.current_batch = 0
        self.batch_size = 100  # 每100次操作更新批次号

        # 统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "total_calls": 0,
            "lru_skips": 0,  # 跳过的LRU更新次数
        }

    def _increment_batch(self):
        """递增批次号"""
        self.current_batch += 1

    def generate_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        优化的缓存键生成（带缓存）

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            缓存键
        """
        # 创建缓存键元组（不可变，可哈希）
        params_tuple = tuple(sorted(params.items()))

        # 检查键缓存
        cache_key = self.key_cache.get((tool_name, params_tuple))
        if cache_key:
            return cache_key

        # 未命中，生成新键
        normalized_params = self._normalize_params(params)
        cache_input = {
            "tool": tool_name,
            "params": normalized_params,
        }
        json_str = json.dumps(cache_input, sort_keys=True, ensure_ascii=False)
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))
        cache_key = f"{tool_name}_{hash_obj.hexdigest()[:16]}"

        # 缓存键
        self.key_cache[(tool_name, params_tuple)] = cache_key

        return cache_key

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """规范化参数"""
        normalized = {}
        for key in sorted(params.keys()):
            value = params[key]
            if value is None:
                continue
            if isinstance(value, (set, frozenset)):
                value = list(value)
            if isinstance(value, dict):
                value = self._normalize_params(value)
            normalized[key] = value
        return normalized

    def get(self, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        优化的缓存获取

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            缓存的结果，如果未命中则返回 None
        """
        self.stats["total_calls"] += 1

        # 定期更新批次号
        if self.stats["total_calls"] % self.batch_size == 0:
            self._increment_batch()

        # 使用缓存的键生成
        cache_key = self.generate_cache_key(tool_name, params)

        # 精确匹配
        if cache_key in self.cache:
            entry = self.cache[cache_key]

            # 检查过期（使用批次号）
            if entry.is_expired(self.current_batch):
                del self.cache[cache_key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                return None

            # 命中！
            self.stats["hits"] += 1

            # 优化：只有部分命中时才更新LRU顺序
            if entry.access_count % self.lru_update_interval == 0:
                entry.touch_batch(self.current_batch)
                self.cache.move_to_end(cache_key)
            else:
                # 只更新访问计数，不更新LRU顺序
                entry.access_count += 1
                self.stats["lru_skips"] += 1

            return entry.result

        # 未命中
        self.stats["misses"] += 1
        return None

    def put(self, tool_name: str, params: Dict[str, Any], result: Dict[str, Any], ttl: Optional[float] = None) -> str:
        """
        存储到缓存

        Args:
            tool_name: 工具名称
            params: 工具参数
            result: 执行结果
            ttl: 过期时间（秒），None 表示使用默认值

        Returns:
            缓存键
        """
        cache_key = self.generate_cache_key(tool_name, params)

        entry = OptimizedCacheEntry(
            cache_key=cache_key,
            tool_name=tool_name,
            params=params,
            result=result,
            timestamp=time.time(),
            last_accessed_batch=self.current_batch,
            access_count=1,
            ttl=ttl or self.default_ttl,
        )

        # LRU淘汰检查
        if len(self.cache) >= self.max_size:
            # 淘汰最旧的条目（第一个）
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1

        self.cache[cache_key] = entry
        return cache_key

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "size": len(self.cache),
            "hit_rate": f"{(self.stats['hits'] / max(self.stats['total_calls'], 1) * 100):.1f}%"
        }


# ========================================
# 性能对比测试
# ========================================

def benchmark_cache_comparison():
    """对比原始缓存和优化缓存的性能"""
    import numpy as np

    from core.tool_call_cache import ToolCallCache

    print("=" * 60)
    print("📊 缓存性能对比测试")
    print("=" * 60)
    print()

    iterations = 1000

    # 测试1: 原始缓存 GET(hit)
    print("测试原始缓存 GET(hit)...")
    cache_original = ToolCallCache(max_size=1000)

    # 预填充
    for i in range(100):
        cache_original.put("tool", {"id": i}, {"result": i})

    times_original = []
    for _ in range(iterations):
        cache_id = np.random.randint(0, 100)
        start = time.perf_counter()
        cache_original.get("tool", {"id": cache_id})
        end = time.perf_counter()
        times_original.append((end - start) * 1000)

    avg_original = sum(times_original) / len(times_original)

    print(f"  原始缓存: {avg_original:.3f}ms")

    # 测试2: 优化缓存 GET(hit)
    print("测试优化缓存 GET(hit)...")
    cache_optimized = OptimizedToolCallCache(max_size=1000)

    # 预填充
    for i in range(100):
        cache_optimized.put("tool", {"id": i}, {"result": i})

    times_optimized = []
    for _ in range(iterations):
        cache_id = np.random.randint(0, 100)
        start = time.perf_counter()
        cache_optimized.get("tool", {"id": cache_id})
        end = time.perf_counter()
        times_optimized.append((end - start) * 1000)

    avg_optimized = sum(times_optimized) / len(times_optimized)

    print(f"  优化缓存: {avg_optimized:.3f}ms")

    # 对比结果
    print()
    print("=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"原始缓存:     {avg_original:.3f}ms")
    print(f"优化缓存:     {avg_optimized:.3f}ms")
    print(f"性能提升:     {(avg_original / avg_optimized):.2f}x")
    print(f"性能改善:     {((avg_original - avg_optimized) / avg_original * 100):.1f}%")

    improvement_ratio = avg_original / avg_optimized
    if improvement_ratio > 1.5:
        print(f"\n✅ 优化成功！性能提升 {improvement_ratio:.2f}倍")
        return True
    else:
        print(f"\n⚠️  优化效果不明显，可能需要其他方案")
        return False


if __name__ == "__main__":
    benchmark_cache_comparison()
