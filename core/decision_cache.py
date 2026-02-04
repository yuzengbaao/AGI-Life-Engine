#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策缓存系统 (Decision Cache System)
========================

功能：
1. 基于向量相似度的意图缓存
2. LRU缓存策略
3. 缓存命中率统计
4. 自动失效机制

目标：
- 缓存命中率 > 60%
- 意图识别延迟 < 50ms（从200-2000ms降低）
- LLM调用率降低50%以上

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import numpy as np
import hashlib
import logging
import time
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    intent: str
    confidence: float
    timestamp: float
    embedding: np.ndarray
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, ttl_seconds: float = 3600) -> bool:
        """检查是否过期"""
        age = time.time() - self.timestamp
        return age > ttl_seconds


class DecisionCache:
    """
    决策缓存系统

    核心功能：
    1. 向量相似度检索（余弦相似度）
    2. LRU缓存淘汰策略
    3. 缓存命中率统计
    4. 自动失效机制

    使用示例：
    ```python
    cache = DecisionCache(max_size=1000)

    # 存储决策
    embedding = get_embedding("读取文件")
    cache.put(embedding, "file_read", confidence=0.95)

    # 检索决策
    result = cache.get(query_embedding)
    if result:
        intent, confidence = result
    ```
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        ttl_seconds: float = 3600,
        enable_stats: bool = True
    ):
        """
        初始化决策缓存

        Args:
            max_size: 最大缓存条目数
            similarity_threshold: 相似度阈值（余弦相似度）
            ttl_seconds: 缓存条目生存时间（秒）
            enable_stats: 是否启用统计
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.enable_stats = enable_stats

        # LRU缓存（OrderedDict按访问顺序维护）
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # 统计信息
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

        logger.info(
            f"[决策缓存] 初始化完成 "
            f"(max_size={max_size}, "
            f"similarity_threshold={similarity_threshold}, "
            f"ttl={ttl_seconds}s)"
        )

    def get(self, text_embedding: np.ndarray) -> Optional[Tuple[str, float, Dict]]:
        """
        基于向量相似度检索缓存

        Args:
            text_embedding: 文本嵌入向量

        Returns:
            (intent, confidence, metadata) 或 None
        """
        if not self.enable_stats:
            # 统计禁用时直接检索
            return self._search_by_similarity(text_embedding)

        start_time = time.time()

        # 1. 尝试精确匹配（基于embedding hash）
        embedding_hash = self._hash_embedding(text_embedding)
        if embedding_hash in self.cache:
            entry = self.cache[embedding_hash]

            # 检查是否过期
            if entry.is_expired(self.ttl_seconds):
                del self.cache[embedding_hash]
                self.expirations += 1
                self.misses += 1
                logger.debug(f"[决策缓存] 缓存条目已过期: {entry.intent}")
                return None

            # 更新访问顺序（LRU）
            self.cache.move_to_end(embedding_hash)
            entry.access_count += 1

            self.hits += 1
            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"[决策缓存] 命中 "
                f"(intent={entry.intent}, "
                f"confidence={entry.confidence:.3f}, "
                f"latency={latency_ms:.2f}ms)"
            )

            return (entry.intent, entry.confidence, entry.metadata)

        # 2. 尝试相似度匹配
        result = self._search_by_similarity(text_embedding)
        if result:
            self.hits += 1
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"[决策缓存] 相似度命中 "
                f"(intent={result[0]}, "
                f"similarity={result[1]:.3f}, "
                f"latency={latency_ms:.2f}ms)"
            )
            return result

        # 3. 未命中
        self.misses += 1
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"[决策缓存] 未命中 (latency={latency_ms:.2f}ms)")
        return None

    def put(
        self,
        text_embedding: np.ndarray,
        intent: str,
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        存储决策结果到缓存

        Args:
            text_embedding: 文本嵌入向量
            intent: 意图分类
            confidence: 置信度
            metadata: 额外元数据
        """
        # 只缓存高置信度的结果
        if confidence < 0.7:
            logger.debug(f"[决策缓存] 置信度过低，不缓存: {confidence:.3f} < 0.7")
            return

        # 生成embedding hash
        embedding_hash = self._hash_embedding(text_embedding)

        # 创建缓存条目
        entry = CacheEntry(
            intent=intent,
            confidence=confidence,
            timestamp=time.time(),
            embedding=text_embedding.copy(),
            access_count=1,
            metadata=metadata or {}
        )

        # 检查缓存大小，必要时淘汰
        if len(self.cache) >= self.max_size and embedding_hash not in self.cache:
            self._evict_lru()

        # 存储到缓存
        self.cache[embedding_hash] = entry
        self.cache.move_to_end(embedding_hash)

        logger.debug(
            f"[决策缓存] 已存储 "
            f"(intent={intent}, "
            f"confidence={confidence:.3f}, "
            f"cache_size={len(self.cache)})"
        )

    def _search_by_similarity(
        self,
        query_embedding: np.ndarray
    ) -> Optional[Tuple[str, float, Dict]]:
        """
        基于余弦相似度搜索

        Args:
            query_embedding: 查询向量

        Returns:
            (intent, similarity, metadata) 或 None
        """
        if not self.cache:
            return None

        best_match = None
        best_similarity = 0.0

        # 遍历所有缓存条目
        for embedding_hash, entry in self.cache.items():
            # 检查是否过期
            if entry.is_expired(self.ttl_seconds):
                continue

            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

                # 提前终止：已经足够相似
                if similarity >= 0.95:
                    break

        # 检查是否达到阈值
        if best_match and best_similarity >= self.similarity_threshold:
            # 更新访问顺序
            embedding_hash = self._hash_embedding(best_match.embedding)
            self.cache.move_to_end(embedding_hash)
            best_match.access_count += 1

            return (
                best_match.intent,
                best_similarity,  # 返回相似度而非置信度
                best_match.metadata
            )

        return None

    def _evict_lru(self) -> None:
        """淘汰最久未使用的缓存条目"""
        if not self.cache:
            return

        # 获取最旧的条目（FIFO）
        oldest_key, oldest_entry = next(iter(self.cache.items()))

        # 从缓存中移除
        del self.cache[oldest_key]
        self.evictions += 1

        logger.debug(
            f"[决策缓存] LRU淘汰 "
            f"(intent={oldest_entry.intent}, "
            f"access_count={oldest_entry.access_count})"
        )

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度（[-1, 1]，1表示完全相同）
        """
        # 确保向量是2D的
        if vec1.ndim == 1:
            vec1 = vec1.reshape(1, -1)
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2.T)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # 如果是2x2矩阵，取标量值
        if similarity.shape == (1, 1):
            similarity = similarity[0, 0]

        return float(similarity)

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """
        生成embedding的哈希值

        Args:
            embedding: 嵌入向量

        Returns:
            MD5哈希字符串
        """
        # 使用向量的字节表示生成哈希
        embedding_bytes = embedding.tobytes()
        return hashlib.md5(embedding_bytes).hexdigest()

    def clear(self) -> None:
        """清空缓存"""
        size_before = len(self.cache)
        self.cache.clear()
        logger.info(f"[决策缓存] 已清空缓存 (清除{size_before}条目)")

    def cleanup_expired(self) -> int:
        """
        清理过期条目

        Returns:
            清理的条目数
        """
        expired_keys = []

        for key, entry in self.cache.items():
            if entry.is_expired(self.ttl_seconds):
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            self.expirations += 1

        if expired_keys:
            logger.info(f"[决策缓存] 清理过期条目: {len(expired_keys)}个")

        return len(expired_keys)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl_seconds,
            'avg_access_count': np.mean([
                e.access_count for e in self.cache.values()
            ]) if self.cache else 0.0
        }

    def optimize_threshold(
        self,
        recent_hits: List[float],
        target_hit_rate: float = 0.60
    ) -> float:
        """
        基于近期命中率动态调整相似度阈值

        Args:
            recent_hits: 近期相似度列表
            target_hit_rate: 目标命中率

        Returns:
            调整后的阈值
        """
        if not recent_hits:
            return self.similarity_threshold

        current_hit_rate = np.mean([1.0 if s >= self.similarity_threshold else 0.0 for s in recent_hits])

        # 命中率过低 → 降低阈值
        if current_hit_rate < target_hit_rate * 0.8:
            new_threshold = max(0.70, self.similarity_threshold - 0.05)
            logger.info(
                f"[决策缓存] 降低阈值 "
                f"{self.similarity_threshold:.3f} → {new_threshold:.3f} "
                f"(命中率={current_hit_rate:.2%})"
            )
            self.similarity_threshold = new_threshold

        # 命中率过高 → 提高阈值（提升准确性）
        elif current_hit_rate > target_hit_rate * 1.2:
            new_threshold = min(0.95, self.similarity_threshold + 0.05)
            logger.info(
                f"[决策缓存] 提高阈值 "
                f"{self.similarity_threshold:.3f} → {new_threshold:.3f} "
                f"(命中率={current_hit_rate:.2%})"
            )
            self.similarity_threshold = new_threshold

        return self.similarity_threshold


# ==================== 便捷函数 ====================

_cache_instance: Optional[DecisionCache] = None


def get_decision_cache(
    max_size: int = 1000,
    similarity_threshold: float = 0.85
) -> DecisionCache:
    """获取或创建决策缓存单例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DecisionCache(
            max_size=max_size,
            similarity_threshold=similarity_threshold
        )
    return _cache_instance


def clear_decision_cache() -> None:
    """清空全局决策缓存"""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.clear()
