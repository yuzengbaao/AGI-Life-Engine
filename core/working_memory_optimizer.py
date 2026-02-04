"""
工作记忆优化模块 - P0修复
解决工作记忆冷却机制过度触发问题
"""

import time
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any


class WorkingMemoryOptimizer:
    """
    工作记忆缓存优化器
    
    优化策略:
    1. 基础TTL: 100 ticks (原20)
    2. 动态TTL: 频繁访问的缓存更久
    3. LRU淘汰: 缓存上限1000条
    4. 访问计数: 热数据保持更久
    """
    
    # 配置参数
    BASE_TTL = 100              # 基础TTL 100 ticks (原20)
    MAX_CACHE_SIZE = 1000       # 最大缓存条目数
    ACCESS_BOOST_FACTOR = 0.5   # 每次访问增加的TTL系数
    MAX_BOOST = 5.0             # 最大TTL倍数
    
    def __init__(self):
        # 使用OrderedDict实现LRU
        self._cache: OrderedDict = OrderedDict()
        self._access_count: Dict = {}
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "ttl_extensions": 0
        }
    
    def should_skip_thought(self, thought_key: Tuple, current_tick: int) -> Tuple[bool, str]:
        """
        判断是否应该跳过thought处理（使用缓存）
        
        Returns:
            (should_skip, reason)
        """
        self._stats["total_requests"] += 1
        
        if thought_key not in self._cache:
            self._stats["cache_misses"] += 1
            return False, "not_in_cache"
        
        last_tick, access_count = self._cache[thought_key], self._access_count.get(thought_key, 0)
        
        # 计算动态TTL
        dynamic_ttl = self._calculate_dynamic_ttl(access_count)
        
        if current_tick - last_tick < dynamic_ttl:
            # 缓存命中
            self._stats["cache_hits"] += 1
            self._access_count[thought_key] = access_count + 1
            self._stats["ttl_extensions"] += 1

            # 移动到OrderedDict末尾（LRU更新）
            self._cache.move_to_end(thought_key)

            return True, f"cache_hit(ttl={dynamic_ttl},age={current_tick-last_tick})"        
        # 缓存过期
        del self._cache[thought_key]
        if thought_key in self._access_count:
            del self._access_count[thought_key]
        
        return False, "cache_expired"
    
    def record_thought(self, thought_key: Tuple, current_tick: int):
        """记录已处理的thought"""
        # 检查是否需要LRU淘汰
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            self._evict_lru()
        
        self._cache[thought_key] = current_tick
        self._access_count[thought_key] = 1
        
        # 移动到末尾（最新）
        self._cache.move_to_end(thought_key)
    
    def _calculate_dynamic_ttl(self, access_count: int) -> int:
        """计算动态TTL"""
        # 访问越多，TTL越长
        boost = min(access_count * self.ACCESS_BOOST_FACTOR, self.MAX_BOOST)
        return int(self.BASE_TTL * (1 + boost))
    
    def _evict_lru(self):
        """LRU淘汰最旧的条目"""
        if len(self._cache) > 0:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            if oldest_key in self._access_count:
                del self._access_count[oldest_key]
            self._stats["evictions"] += 1
    
    def cleanup_expired(self, current_tick: int) -> int:
        """清理过期缓存，返回清理数量"""
        expired_keys = []
        for key, last_tick in self._cache.items():
            access_count = self._access_count.get(key, 0)
            dynamic_ttl = self._calculate_dynamic_ttl(access_count)
            if current_tick - last_tick >= dynamic_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        total = stats["total_requests"]
        if total > 0:
            stats["hit_rate"] = stats["cache_hits"] / total
        else:
            stats["hit_rate"] = 0.0
        
        stats["cache_size"] = len(self._cache)
        stats["current_ttl"] = self.BASE_TTL
        
        return stats
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "ttl_extensions": 0
        }
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_count.clear()
        self.reset_stats()


# 便捷函数
def create_working_memory_optimizer() -> WorkingMemoryOptimizer:
    """创建工作记忆优化器实例"""
    return WorkingMemoryOptimizer()


# 测试代码
if __name__ == "__main__":
    optimizer = WorkingMemoryOptimizer()
    
    print("工作记忆优化器测试:")
    print("-" * 60)
    
    # 模拟tick
    for tick in range(200):
        thought_key = ("explore", f"concept_{tick % 10}")  # 循环10个概念
        
        should_skip, reason = optimizer.should_skip_thought(thought_key, tick)
        
        if not should_skip:
            # 模拟处理
            optimizer.record_thought(thought_key, tick)
        
        # 每50 tick打印统计
        if tick % 50 == 0 and tick > 0:
            stats = optimizer.get_stats()
            print(f"Tick {tick}: hit_rate={stats['hit_rate']:.1%}, "
                  f"cache_size={stats['cache_size']}, "
                  f"evictions={stats['evictions']}")
    
    print("-" * 60)
    final_stats = optimizer.get_stats()
    print(f"最终统计:")
    print(f"  总请求: {final_stats['total_requests']}")
    print(f"  缓存命中: {final_stats['cache_hits']}")
    print(f"  缓存未命中: {final_stats['cache_misses']}")
    print(f"  命中率: {final_stats['hit_rate']:.1%}")
    print(f"  LRU淘汰: {final_stats['evictions']}")
    print(f"  当前缓存: {final_stats['cache_size']}")
