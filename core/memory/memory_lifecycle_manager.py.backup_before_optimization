#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经记忆生命周期管理器
Neural Memory Lifecycle Manager

P2-1 修复: 解决 630,931+ 条记录持续增长的存储压力

功能:
1. 记录老化追踪 (age tracking)
2. 多策略淘汰 (LRU, LFU, importance-based)
3. 自动压缩与合并
4. 内存压力监控与动态清理

作者: AGI System
日期: 2026-02-04
"""

import time
import logging
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """淘汰策略枚举"""
    LRU = "least_recently_used"  # 最近最少使用
    LFU = "least_frequently_used"  # 最少使用频率
    IMPORTANCE = "importance_based"  # 基于重要性评分
    HYBRID = "hybrid"  # 混合策略


@dataclass
class MemoryRecord:
    """增强的记忆记录"""
    id: str
    timestamp: float
    last_accessed: float
    access_count: int
    importance_score: float  # 0.0-1.0
    compressed: bool = False
    archived: bool = False
    tags: List[str] = field(default_factory=list)

    def age(self) -> float:
        """计算记录年龄（秒）"""
        return time.time() - self.timestamp

    def access_age(self) -> float:
        """计算距离上次访问时间（秒）"""
        return time.time() - self.last_accessed

    def touch(self):
        """更新访问时间和计数"""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "compressed": self.compressed,
            "archived": self.archived,
            "tags": self.tags,
            "age_seconds": self.age(),
            "access_age_seconds": self.access_age()
        }


class MemoryLifecycleManager:
    """
    神经记忆生命周期管理器

    核心职责:
    1. 追踪记忆记录的生命周期状态
    2. 根据策略淘汰低价值记录
    3. 压缩和归档长期不活跃记录
    4. 监控内存压力并自动清理
    """

    def __init__(
        self,
        max_records: int = 100000,  # 最大记录数
        max_age_days: float = 30.0,  # 最大保留天数
        eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
        auto_cleanup_interval: int = 100,  # 每 N 次操作自动清理
        compression_threshold: int = 10000,  # 超过此阈值开始压缩
        archive_ratio: float = 0.1,  # 归档比例 (10% 最不活跃)
    ):
        self.max_records = max_records
        self.max_age_seconds = max_age_days * 86400
        self.eviction_policy = eviction_policy
        self.auto_cleanup_interval = auto_cleanup_interval
        self.compression_threshold = compression_threshold
        self.archive_ratio = archive_ratio

        # 记录索引: {memory_id: MemoryRecord}
        self.records: OrderedDict[str, MemoryRecord] = OrderedDict()

        # 操作计数器（用于触发自动清理）
        self.operation_count = 0

        # 统计信息
        self.stats = {
            "total_added": 0,
            "total_evicted": 0,
            "total_compressed": 0,
            "total_archived": 0,
            "cleanup_runs": 0,
        }

        logger.info(
            f"🧠 MemoryLifecycleManager 初始化: "
            f"max_records={max_records}, "
            f"max_age_days={max_age_days}, "
            f"policy={eviction_policy.value}"
        )

    def register_record(
        self,
        memory_id: str,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> MemoryRecord:
        """
        注册新的记忆记录

        Args:
            memory_id: 记忆唯一标识
            importance_score: 重要性评分 (0.0-1.0)
            tags: 标签列表

        Returns:
            创建的 MemoryRecord 对象
        """
        now = time.time()
        record = MemoryRecord(
            id=memory_id,
            timestamp=now,
            last_accessed=now,
            access_count=1,
            importance_score=importance_score,
            tags=tags or [],
        )

        self.records[memory_id] = record
        self.stats["total_added"] += 1
        self.operation_count += 1

        # 触发自动清理
        if self.operation_count >= self.auto_cleanup_interval:
            self.auto_cleanup()

        return record

    def touch_record(self, memory_id: str) -> Optional[MemoryRecord]:
        """
        更新记录的访问时间和计数

        Args:
            memory_id: 记忆ID

        Returns:
            更新后的记录，如果不存在则返回 None
        """
        record = self.records.get(memory_id)
        if record:
            record.touch()
            self.operation_count += 1
        return record

    def auto_cleanup(self) -> Dict[str, Any]:
        """
        自动清理流程

        执行步骤:
        1. 检查是否超过最大记录数
        2. 淘汰最不活跃的记录
        3. 压缩长期不活跃记录
        4. 归档超时记录

        Returns:
            清理结果统计
        """
        logger.info("🧹 开始自动清理记忆...")

        result = {
            "before_count": len(self.records),
            "evicted": 0,
            "compressed": 0,
            "archived": 0,
            "after_count": 0,
        }

        # 1. 淘汰超量记录
        if len(self.records) > self.max_records:
            excess = len(self.records) - self.max_records
            evicted = self.evict(excess)
            result["evicted"] = evicted

        # 2. 压缩不活跃记录
        if len(self.records) > self.compression_threshold:
            compressed = self.compress_inactive()
            result["compressed"] = compressed

        # 3. 归档超时记录
        archived = self.archive_old()
        result["archived"] = archived

        result["after_count"] = len(self.records)
        self.stats["cleanup_runs"] += 1
        self.operation_count = 0

        logger.info(
            f"✅ 清理完成: "
            f"淘汰={result['evicted']}, "
            f"压缩={result['compressed']}, "
            f"归档={result['archived']}, "
            f"剩余={result['after_count']}"
        )

        return result

    def evict(self, count: int) -> int:
        """
        根据策略淘汰记录

        Args:
            count: 要淘汰的记录数量

        Returns:
            实际淘汰的记录数
        """
        if count <= 0 or not self.records:
            return 0

        # 根据策略选择要淘汰的记录
        to_evict = self._select_for_eviction(count)

        # 执行淘汰
        for memory_id in to_evict:
            del self.records[memory_id]
            self.stats["total_evicted"] += 1

        logger.info(f"🗑️ 淘汰了 {len(to_evict)} 条记录 (策略: {self.eviction_policy.value})")
        return len(to_evict)

    def _select_for_eviction(self, count: int) -> List[str]:
        """根据淘汰策略选择记录"""
        records = list(self.records.values())

        if self.eviction_policy == EvictionPolicy.LRU:
            # 最近最少使用: 按 access_age 排序
            scored = [(r.id, r.access_age()) for r in records]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [r[0] for r in scored[:count]]

        elif self.eviction_policy == EvictionPolicy.LFU:
            # 最少使用频率: 按 access_count 排序
            scored = [(r.id, r.access_count) for r in records]
            scored.sort(key=lambda x: x[1])
            return [r[0] for r in scored[:count]]

        elif self.eviction_policy == EvictionPolicy.IMPORTANCE:
            # 基于重要性: 按 importance_score 排序
            scored = [(r.id, r.importance_score) for r in records]
            scored.sort(key=lambda x: x[1])
            return [r[0] for r in scored[:count]]

        else:  # HYBRID
            # 混合策略: 综合考虑访问频率、年龄和重要性
            # 分数 = (access_count * 0.3) + (importance * 100) - (access_age / 86400)
            scored = []
            for r in records:
                access_age_days = r.access_age() / 86400
                score = (
                    r.access_count * 0.3
                    + r.importance_score * 100
                    - access_age_days
                )
                scored.append((r.id, score))

            scored.sort(key=lambda x: x[1])
            return [r[0] for r in scored[:count]]

    def compress_inactive(self, threshold_days: float = 7.0) -> int:
        """
        压缩长期不活跃的记录

        Args:
            threshold_days: 不活跃天数阈值

        Returns:
            压缩的记录数
        """
        threshold_seconds = threshold_days * 86400
        compressed = 0

        for record in self.records.values():
            if not record.compressed and record.access_age() > threshold_seconds:
                record.compressed = True
                compressed += 1
                self.stats["total_compressed"] += 1

        if compressed > 0:
            logger.info(f"📦 压缩了 {compressed} 条不活跃记录")

        return compressed

    def archive_old(self) -> int:
        """
        归档超时记录

        Returns:
            归档的记录数
        """
        to_archive = []

        for memory_id, record in self.records.items():
            if not record.archived and record.age() > self.max_age_seconds:
                to_archive.append(memory_id)

        for memory_id in to_archive:
            self.records[memory_id].archived = True
            self.stats["total_archived"] += 1

        # 从活动记录中移除归档记录
        # 实际应用中，归档记录应移动到持久存储
        for memory_id in to_archive:
            del self.records[memory_id]

        if to_archive:
            logger.info(f"📁 归档了 {len(to_archive)} 条超时记录")

        return len(to_archive)

    def calculate_importance(self, metadata: Dict[str, Any]) -> float:
        """
        计算记忆记录的重要性评分

        考虑因素:
        - 类型权重 (macro > episode)
        - 工具调用次数
        - 是否有技能关联
        - 连接数 (拓扑图中的度)

        Args:
            metadata: 记忆元数据

        Returns:
            重要性评分 (0.0-1.0)
        """
        score = 0.5  # 基础分

        # 类型权重
        mem_type = metadata.get("type", "episode")
        if mem_type == "macro":
            score += 0.3
        elif mem_type == "tool_call" or mem_type == "skill_call":
            score += 0.1

        # 技能关联
        if metadata.get("skill"):
            score += 0.1

        # 工具调用 (常见工具降低分值，罕见工具提升分值)
        tool = metadata.get("tool")
        common_tools = {"file_operations", "world_model", "metacognition"}
        if tool and tool not in common_tools:
            score += 0.1

        # 原型数量 (macro_induction 产生的 prototype_ids)
        prototype_ids = metadata.get("prototype_ids")
        if isinstance(prototype_ids, list) and len(prototype_ids) > 0:
            score += min(0.2, len(prototype_ids) * 0.05)

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        active = [r for r in self.records.values() if not r.archived]
        compressed = [r for r in self.records.values() if r.compressed]

        return {
            "total_records": len(self.records),
            "active_records": len(active),
            "compressed_records": len(compressed),
            "operation_count": self.operation_count,
            "stats": self.stats.copy(),
            "eviction_policy": self.eviction_policy.value,
            "max_records": self.max_records,
            "usage_ratio": len(self.records) / self.max_records,
        }

    def export_records_for_cleanup(
        self, memory_latents: np.ndarray, memory_metadata: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        导出清理后的记忆数组

        Args:
            memory_latents: 原始潜在向量数组
            memory_metadata: 原始元数据列表

        Returns:
            (清理后的 latents, 清理后的 metadata)
        """
        if not self.records:
            return memory_latents, memory_metadata

        # 获取活动记录的索引
        active_indices = []
        metadata_dict = {m.get("id"): i for i, m in enumerate(memory_metadata)}

        for memory_id in self.records.keys():
            if memory_id in metadata_dict:
                active_indices.append(metadata_dict[memory_id])

        if not active_indices:
            return np.array([]), []

        # 过滤 latents 和 metadata
        cleaned_latents = memory_latents[active_indices]
        cleaned_metadata = [memory_metadata[i] for i in active_indices]

        logger.info(
            f"🧹 记忆清理: "
            f"{len(memory_latents)} -> {len(cleaned_latents)} "
            f"({len(memory_latents) - len(cleaned_latents)} 条被移除)"
        )

        return cleaned_latents, cleaned_metadata

    def save_state(self, filepath: str):
        """保存生命周期管理器状态"""
        state = {
            "records": {mid: r.to_dict() for mid, r in self.records.items()},
            "stats": self.stats,
            "operation_count": self.operation_count,
            "config": {
                "max_records": self.max_records,
                "max_age_days": self.max_age_seconds / 86400,
                "eviction_policy": self.eviction_policy.value,
                "auto_cleanup_interval": self.auto_cleanup_interval,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 生命周期状态已保存: {filepath}")

    def load_state(self, filepath: str):
        """加载生命周期管理器状态"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 恢复记录
            self.records.clear()
            for mid, rdict in state.get("records", {}).items():
                record = MemoryRecord(
                    id=rdict["id"],
                    timestamp=rdict["timestamp"],
                    last_accessed=rdict["last_accessed"],
                    access_count=rdict["access_count"],
                    importance_score=rdict["importance_score"],
                    compressed=rdict.get("compressed", False),
                    archived=rdict.get("archived", False),
                    tags=rdict.get("tags", []),
                )
                self.records[mid] = record

            # 恢复统计
            self.stats = state.get("stats", self.stats.copy())
            self.operation_count = state.get("operation_count", 0)

            logger.info(f"📂 生命周期状态已加载: {len(self.records)} 条记录")

        except FileNotFoundError:
            logger.warning(f"状态文件不存在: {filepath}")
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
