"""
RecursiveSelfMemory - 递归自引用记忆系统

功能边界:
- 输入: 经验/观察 + 元数据 (为何记住、重要性、触发者)
- 输出: 可查询的记忆 + 递归摘要 + "为何记住/遗忘"解释
- 约束: 资源预算<20%、递归深度上限、定期压缩

拓扑连接:
- RecursiveSelfMemory 接收 TheSeed的experience
- RecursiveSelfMemory 发布 memory_created 事件
- InsightValidator 订阅记忆事件并提取洞察
- CriticAgent 使用记忆进行决策

记忆层级:
- L0: 事件记忆 (raw experience)
- L1: 记忆过程 (为什么记住、置信度、重要性)
- L2: 记忆摘要 (100条→1条摘要)
- L3: 策略记忆 (记忆规则的演化)

设计原则:
1. 递归自指: 记忆系统本身也被记忆
2. 可解释: 每条记忆都有"为何记住"的理由
3. 资源受限: 强制配额+定期压缩
4. 深度限制: 避免无限递归
"""

import time
import logging
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举和数据结构
# ============================================================================

class MemoryLayer(Enum):
    """记忆层级"""
    L0_EVENT = "l0_event"         # 事件记忆
    L1_PROCESS = "l1_process"     # 记忆过程
    L2_SUMMARY = "l2_summary"     # 记忆摘要
    L3_STRATEGY = "l3_strategy"   # 策略记忆


class MemoryImportance(Enum):
    """记忆重要性"""
    CRITICAL = "critical"  # 关键记忆 (不可删除)
    HIGH = "high"         # 高重要性
    MEDIUM = "medium"     # 中等重要性
    LOW = "low"           # 低重要性
    EPHEMERAL = "ephemeral"  # 短暂记忆 (优先删除)


class ForgettingReason(Enum):
    """遗忘原因"""
    LOW_IMPORTANCE = "low_importance"        # 重要性低
    RESOURCE_PRESSURE = "resource_pressure"  # 资源压力
    REDUNDANT = "redundant"                  # 冗余信息
    OUTDATED = "outdated"                    # 过时信息
    INCONSISTENT = "inconsistent"            # 与新知识矛盾
    COMPRESSION = "compression"              # 压缩到摘要


@dataclass
class MemoryMetadata:
    """
    记忆元数据 - L1层级

    记录"为什么记住"以及记忆的过程
    """
    memory_id: str
    created_at: float
    importance: MemoryImportance

    # 为什么记住
    why_remembered: str  # 记住的原因
    confidence: float    # 置信度 (0-1)
    trigger: str         # 触发者 (哪个模块创建的)

    # 访问历史
    access_count: int = 0
    last_accessed: float = 0.0
    access_frequency: float = 0.0  # 访问频率

    # 关联
    related_memories: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    # 资源控制
    size_bytes: int = 0
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        data = asdict(self)
        data['importance'] = self.importance.value
        data['tags'] = list(self.tags)
        return data


@dataclass
class EventMemory:
    """
    事件记忆 - L0层级

    原始经验/观察数据
    """
    id: str
    timestamp: float
    event_type: str  # "experience", "observation", "insight", etc.

    # 事件内容
    content: Dict[str, Any]

    # 元数据 (L1)
    metadata: MemoryMetadata

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'content': self.content,
            'metadata': self.metadata.to_dict()
        }


@dataclass
class MemorySummary:
    """
    记忆摘要 - L2层级

    从多条事件记忆压缩形成的摘要
    """
    id: str
    created_at: float
    layer: MemoryLayer

    # 摘要内容
    summary: str
    key_points: List[str]

    # 源记忆
    source_memory_ids: List[str]
    source_count: int

    # 元数据
    metadata: MemoryMetadata

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'id': self.id,
            'timestamp': self.created_at,
            'layer': self.layer.value,
            'summary': self.summary,
            'key_points': self.key_points,
            'source_memory_ids': self.source_memory_ids,
            'source_count': self.source_count,
            'metadata': self.metadata.to_dict()
        }


@dataclass
class StrategyMemory:
    """
    策略记忆 - L3层级

    记忆系统的策略演化 ("我倾向于记什么/忘什么")
    """
    id: str
    created_at: float
    updated_at: float

    # 策略内容
    strategy_type: str  # "remember_criteria", "forget_criteria", etc.
    strategy_description: str

    # 策略参数
    parameters: Dict[str, Any]

    # 性能评估
    effectiveness_score: float  # 0-1
    usage_count: int

    # 元数据
    metadata: MemoryMetadata

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'id': self.id,
            'timestamp': self.created_at,
            'updated_timestamp': self.updated_at,
            'strategy_type': self.strategy_type,
            'strategy_description': self.strategy_description,
            'parameters': self.parameters,
            'effectiveness_score': self.effectiveness_score,
            'usage_count': self.usage_count,
            'metadata': self.metadata.to_dict()
        }


@dataclass
class ForgettingRecord:
    """
    遗忘记录

    记录"为什么遗忘"某条记忆
    """
    memory_id: str
    forgotten_at: float
    reason: ForgettingReason
    reason_detail: str

    # 原始记忆的摘要 (用于追溯)
    original_summary: str

    # 转移到摘要?
    transferred_to_summary: bool = False
    summary_id: Optional[str] = None


# ============================================================================
# 核心实现
# ============================================================================

class RecursiveSelfMemory:
    """
    递归自引用记忆系统

    核心能力:
    1. remember(): 存储记忆 + 元数据
    2. recall(): 检索记忆
    3. forget(): 遗忘记忆
    4. summarize(): 压缩记忆形成摘要
    5. why_remembered(): 解释为何记住
    6. why_forgotten(): 解释为何遗忘

    特性:
    - 递归: 记忆系统的操作本身也被记忆
    - 自指: 记忆"如何记忆"
    - 资源受限: 强制配额+定期压缩
    """

    # 配置常量
    MAX_L0_MEMORIES = 1000      # L0最大记忆数
    MAX_L1_METADATA_OVERHEAD = 0.2  # L1元数据最大开销20%
    SUMMARY_WINDOW = 100        # 每100条形成摘要
    MAX_RECURSION_DEPTH = 3     # 最大递归深度
    FORGETTING_THRESHOLD = 0.3  # 遗忘阈值 (重要性<0.3)

    def __init__(self, event_bus: Any = None,
                 memory_dir: str = "./data/recursive_self_memory"):
        """
        初始化递归自引用记忆系统

        Args:
            event_bus: 事件总线
            memory_dir: 记忆存储目录
        """
        self.event_bus = event_bus
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 记忆存储
        self.l0_events: Dict[str, EventMemory] = {}  # 事件记忆
        self.l1_metadata: Dict[str, MemoryMetadata] = {}  # 记忆过程元数据
        self.l2_summaries: List[MemorySummary] = []  # 记忆摘要
        self.l3_strategies: Dict[str, StrategyMemory] = {}  # 策略记忆

        # 遗忘记录
        self.forgetting_records: List[ForgettingRecord] = []

        # 统计
        self._total_remembered = 0
        self._total_forgotten = 0
        self._total_summarized = 0
        self._metadata_size_bytes = 0  # 元数据总大小 (MemoryMetadata对象开销)
        self._content_size_bytes = 0   # 内容总大小 (实际content数据大小)

        # 记忆策略 (L3)
        self._init_default_strategies()

        # 递归自指: 记忆系统自身的操作
        self._self_memory_enabled = True

        # 防递归标志
        self._in_summarize = False
        self._in_check_limits = False

        logger.info(f"🧠 RecursiveSelfMemory initialized (memory_dir={self.memory_dir})")

    # ========================================================================
    # 核心接口
    # ========================================================================

    def remember(self, event_type: str,
                 content: Dict[str, Any],
                 importance: MemoryImportance = MemoryImportance.MEDIUM,
                 why: str = "",
                 confidence: float = 1.0,
                 trigger: str = "system",
                 _is_meta: bool = False) -> str:
        """
        存储记忆

        Args:
            event_type: 事件类型 ("experience", "observation", "insight", etc.)
            content: 事件内容
            importance: 重要性
            why: 为什么记住
            confidence: 置信度 (0-1)
            trigger: 触发者
            _is_meta: 内部参数,是否为元记忆 (防止无限递归)

        Returns:
            记忆ID
        """
        # 生成记忆ID
        memory_id = hashlib.sha256(
            f"{event_type}{time.time()}{str(content)}".encode()
        ).hexdigest()[:16]

        # 计算实际内容大小
        content_bytes = len(str(content).encode())

        # 估计元数据大小 (MemoryMetadata对象开销)
        metadata_bytes_estimate = 500

        # 创建元数据 (L1)
        metadata = MemoryMetadata(
            memory_id=memory_id,
            created_at=time.time(),
            importance=importance,
            why_remembered=why or f"Auto-remembered {event_type}",
            confidence=confidence,
            trigger=trigger,
            size_bytes=metadata_bytes_estimate
        )

        # 创建事件记忆 (L0)
        event = EventMemory(
            id=memory_id,
            timestamp=time.time(),
            event_type=event_type,
            content=content,
            metadata=metadata
        )

        # 存储
        self.l0_events[memory_id] = event
        self.l1_metadata[memory_id] = metadata
        self._metadata_size_bytes += metadata_bytes_estimate
        self._content_size_bytes += content_bytes

        self._total_remembered += 1

        # 递归自指: 仅对非元记忆记录操作 (防止无限递归)
        if self._self_memory_enabled and not _is_meta:
            self._remember_memory_operation("remember", memory_id, event_type, _is_meta=True)

        # 检查资源限制
        self._check_resource_limits()

        # 发布事件
        self._publish_memory_event("memory_created", {
            'memory_id': memory_id,
            'event_type': event_type,
            'importance': importance.value
        })

        logger.debug(f"[RecursiveSelfMemory] 记住: {memory_id} ({event_type})")

        return memory_id

    def recall(self, query: str,
               limit: int = 10,
               min_importance: Optional[MemoryImportance] = None) -> List[EventMemory]:
        """
        检索记忆

        Args:
            query: 查询字符串
            limit: 返回数量限制
            min_importance: 最小重要性

        Returns:
            匹配的事件记忆列表
        """
        results = []

        query_lower = query.lower()

        for memory_id, event in self.l0_events.items():
            # 重要性过滤
            if min_importance:
                importance_order = [
                    MemoryImportance.EPHEMERAL,
                    MemoryImportance.LOW,
                    MemoryImportance.MEDIUM,
                    MemoryImportance.HIGH,
                    MemoryImportance.CRITICAL
                ]
                if importance_order.index(event.metadata.importance) < importance_order.index(min_importance):
                    continue

            # 内容匹配
            content_str = str(event.content).lower()
            if query_lower in content_str or query_lower in event.metadata.why_remembered.lower():
                results.append(event)

                # 更新访问统计
                event.metadata.access_count += 1
                event.metadata.last_accessed = time.time()
                event.metadata.access_frequency = (
                    event.metadata.access_count /
                    (time.time() - event.metadata.created_at + 1)
                )

                if len(results) >= limit:
                    break

        # 按重要性+访问频率排序
        results.sort(key=lambda e: (
            self._importance_score(e.metadata.importance),
            e.metadata.access_frequency
        ), reverse=True)

        logger.debug(f"[RecursiveSelfMemory] 回忆: {query} → {len(results)}条结果")

        return results

    def forget(self, memory_id: str,
               reason: ForgettingReason = ForgettingReason.LOW_IMPORTANCE,
               detail: str = "") -> bool:
        """
        遗忘记忆

        Args:
            memory_id: 记忆ID
            reason: 遗忘原因
            detail: 详细原因

        Returns:
            是否成功遗忘
        """
        if memory_id not in self.l0_events:
            logger.warning(f"记忆不存在: {memory_id}")
            return False

        event = self.l0_events[memory_id]
        metadata = self.l1_metadata[memory_id]

        # 关键记忆不可遗忘
        if metadata.importance == MemoryImportance.CRITICAL:
            logger.warning(f"关键记忆不可遗忘: {memory_id}")
            return False

        # 创建遗忘记录
        forgetting_record = ForgettingRecord(
            memory_id=memory_id,
            forgotten_at=time.time(),
            reason=reason,
            reason_detail=detail or f"{reason.value}: {metadata.why_remembered}",
            original_summary=str(event.content)[:200]  # 前200字符
        )

        # 尝试转移到摘要
        if len(self.l2_summaries) > 0 and self._should_summarize_before_forgetting(metadata):
            # 将关键信息转移到最新摘要
            latest_summary = self.l2_summaries[-1]
            latest_summary.source_memory_ids.append(memory_id)
            latest_summary.source_count += 1
            forgetting_record.transferred_to_summary = True
            forgetting_record.summary_id = latest_summary.id

        # 计算并减去内容大小
        content_bytes = len(str(event.content).encode())

        # 删除记忆
        del self.l0_events[memory_id]
        del self.l1_metadata[memory_id]
        self._metadata_size_bytes -= metadata.size_bytes
        self._content_size_bytes -= content_bytes

        # 记录遗忘
        self.forgetting_records.append(forgetting_record)
        self._total_forgotten += 1

        # 递归自指: 记住"遗忘"这个操作
        if self._self_memory_enabled:
            self._remember_memory_operation("forget", memory_id, reason.value)

        # 发布事件
        self._publish_memory_event("memory_forgotten", {
            'memory_id': memory_id,
            'reason': reason.value
        })

        logger.debug(f"[RecursiveSelfMemory] 遗忘: {memory_id} ({reason.value})")

        return True

    def summarize(self, force: bool = False) -> Optional[MemorySummary]:
        """
        压缩记忆形成摘要 (L2层级)

        从最近的L0事件中选择100条形成摘要

        Args:
            force: 是否强制摘要

        Returns:
            记忆摘要
        """
        # 防止递归
        if self._in_summarize:
            return None

        if not force and len(self.l0_events) < self.SUMMARY_WINDOW:
            return None

        self._in_summarize = True
        try:
            # 选择最近的事件
            recent_events = sorted(
                self.l0_events.values(),
                key=lambda e: e.timestamp,
                reverse=True
            )[:self.SUMMARY_WINDOW]

            # 按类型分组
            event_groups = defaultdict(list)
            for event in recent_events:
                event_groups[event.event_type].append(event)

            # 生成关键点
            key_points = []
            for event_type, events in event_groups.items():
                count = len(events)
                key_points.append(f"{count}x {event_type}")

            # 生成摘要
            summary_text = f"Summary of {len(recent_events)} events: " + ", ".join(key_points)

            # 创建摘要
            summary_id = hashlib.sha256(
                f"summary{time.time()}{summary_text}".encode()
            ).hexdigest()[:16]

            summary = MemorySummary(
                id=summary_id,
                created_at=time.time(),
                layer=MemoryLayer.L2_SUMMARY,
                summary=summary_text,
                key_points=key_points,
                source_memory_ids=[e.id for e in recent_events],
                source_count=len(recent_events),
                metadata=MemoryMetadata(
                    memory_id=summary_id,
                    created_at=time.time(),
                    importance=MemoryImportance.HIGH,
                    why_remembered=f"Summary of {len(recent_events)} events",
                    confidence=0.9,
                    trigger="summarization",
                    size_bytes=500  # 估计元数据大小
                )
            )

            # 存储
            self.l2_summaries.append(summary)

            self._total_summarized += 1

            # 递归自指: 记住"摘要"这个操作 (使用_is_meta防止递归)
            if self._self_memory_enabled:
                self._remember_memory_operation("summarize", summary_id, "l2_summary", _is_meta=True)

            logger.info(f"[RecursiveSelfMemory] 摘要: {summary_id} ({len(recent_events)}条事件)")

            return summary
        finally:
            self._in_summarize = False

    def why_remembered(self, memory_id: str) -> Optional[str]:
        """
        解释为何记住某条记忆

        Args:
            memory_id: 记忆ID

        Returns:
            解释文本
        """
        if memory_id not in self.l1_metadata:
            return f"记忆不存在或已被遗忘: {memory_id}"

        metadata = self.l1_metadata[memory_id]

        explanation = f"""
记忆ID: {memory_id}
记住原因: {metadata.why_remembered}
重要性: {metadata.importance.value}
置信度: {metadata.confidence:.2f}
触发者: {metadata.trigger}
创建时间: {datetime.fromtimestamp(metadata.created_at).strftime('%Y-%m-%d %H:%M:%S')}
访问次数: {metadata.access_count}
访问频率: {metadata.access_frequency:.4f} /秒
"""

        return explanation.strip()

    def why_forgotten(self, memory_id: str) -> Optional[str]:
        """
        解释为何遗忘某条记忆

        Args:
            memory_id: 记忆ID

        Returns:
            解释文本
        """
        # 在遗忘记录中查找
        for record in self.forgetting_records:
            if record.memory_id == memory_id:
                explanation = f"""
记忆ID: {memory_id}
遗忘原因: {record.reason.value}
详细说明: {record.reason_detail}
遗忘时间: {datetime.fromtimestamp(record.forgotten_at).strftime('%Y-%m-%d %H:%M:%S')}
原始摘要: {record.original_summary}
转移到摘要: {'是 (' + record.summary_id + ')' if record.transferred_to_summary else '否'}
"""
                return explanation.strip()

        # 检查是否仍在记忆中
        if memory_id in self.l0_events:
            return f"记忆未被遗忘: {memory_id}"

        return f"未找到遗忘记录: {memory_id}"

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _remember_memory_operation(self, operation: str, target_id: str, detail: str, _is_meta: bool = False):
        """记住记忆系统的操作 (递归自指)"""
        # 创建关于记忆系统自身的记忆 (使用_is_meta防止递归)
        self.remember(
            event_type="memory_operation",
            content={
                'operation': operation,
                'target_id': target_id,
                'detail': detail,
                'timestamp': time.time()
            },
            importance=MemoryImportance.LOW,  # 元记忆低优先级
            why=f"Memory system performed {operation} on {target_id}",
            confidence=1.0,
            trigger="RecursiveSelfMemory.self",
            _is_meta=True  # 防止递归
        )

    def _check_resource_limits(self):
        """检查资源限制"""
        # 防止递归
        if self._in_check_limits:
            return

        self._in_check_limits = True
        try:
            # 检查L0记忆数量
            if len(self.l0_events) > self.MAX_L0_MEMORIES:
                logger.warning("L0记忆数量超限,触发遗忘")
                self._trigger_forgetting_for_resources()

            # 检查元数据开销
            total_size = sum(e.metadata.size_bytes for e in self.l0_events.values())
            metadata_ratio = self._metadata_size_bytes / max(total_size, 1)

            if metadata_ratio > self.MAX_L1_METADATA_OVERHEAD:
                logger.warning(f"元数据开销过高 ({metadata_ratio:.1%}), 触发压缩")
                self._compress_metadata()

            # 检查是否需要摘要 (不在summarize中时才检查)
            if len(self.l0_events) >= self.SUMMARY_WINDOW and not self._in_summarize:
                self.summarize()
        finally:
            self._in_check_limits = False

    def _trigger_forgetting_for_resources(self):
        """因资源压力触发遗忘"""
        # 按重要性+访问频率排序,删除最不重要的
        memories = sorted(
            self.l0_events.items(),
            key=lambda x: (
                self._importance_score(x[1].metadata.importance),
                x[1].metadata.access_frequency
            )
        )

        # 删除最不重要的10%
        to_forget = int(len(memories) * 0.1)

        for memory_id, _ in memories[:to_forget]:
            self.forget(memory_id, ForgettingReason.RESOURCE_PRESSURE)

    def _compress_metadata(self):
        """压缩元数据"""
        compressed_count = 0

        for metadata in self.l1_metadata.values():
            if not metadata.compressed and metadata.importance != MemoryImportance.CRITICAL:
                # 简化元数据
                metadata.tags = set(list(metadata.tags)[:5])  # 只保留前5个标签
                metadata.related_memories = metadata.related_memories[:10]  # 只保留前10个关联
                metadata.compressed = True
                compressed_count += 1

        logger.info(f"压缩了 {compressed_count} 条元数据")

    def _should_summarize_before_forgetting(self, metadata: MemoryMetadata) -> bool:
        """判断遗忘前是否应该摘要"""
        # 高重要性且高访问频率的记忆应该摘要
        return (
            metadata.importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL] and
            metadata.access_frequency > 0.01
        )

    def _importance_score(self, importance: MemoryImportance) -> float:
        """重要性转分数"""
        scores = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.5,
            MemoryImportance.LOW: 0.3,
            MemoryImportance.EPHEMERAL: 0.1
        }
        return scores.get(importance, 0.5)

    def _init_default_strategies(self):
        """初始化默认记忆策略 (L3)"""
        # 记住策略
        remember_strategy = StrategyMemory(
            id="strategy_remember_default",
            created_at=time.time(),
            updated_at=time.time(),
            strategy_type="remember_criteria",
            strategy_description="记住重要性>=MEDIUM且置信度>=0.5的事件",
            parameters={
                'min_importance': 'MEDIUM',
                'min_confidence': 0.5
            },
            effectiveness_score=0.7,
            usage_count=0,
            metadata=MemoryMetadata(
                memory_id="strategy_remember_default",
                created_at=time.time(),
                importance=MemoryImportance.HIGH,
                why_remembered="默认记住策略",
                confidence=0.9,
                trigger="system"
            )
        )

        # 遗忘策略
        forget_strategy = StrategyMemory(
            id="strategy_forget_default",
            created_at=time.time(),
            updated_at=time.time(),
            strategy_type="forget_criteria",
            strategy_description="遗忘重要性<LOW且访问频率<0.001的记忆",
            parameters={
                'max_importance': 'LOW',
                'max_access_frequency': 0.001
            },
            effectiveness_score=0.7,
            usage_count=0,
            metadata=MemoryMetadata(
                memory_id="strategy_forget_default",
                created_at=time.time(),
                importance=MemoryImportance.HIGH,
                why_remembered="默认遗忘策略",
                confidence=0.9,
                trigger="system"
            )
        )

        self.l3_strategies[remember_strategy.id] = remember_strategy
        self.l3_strategies[forget_strategy.id] = forget_strategy

    def _publish_memory_event(self, event_type: str, data: Dict[str, Any]):
        """发布记忆事件"""
        if not self.event_bus:
            return

        try:
            from core.event_bus import Event, EventType
            event = Event(
                type=EventType.INFO,
                source="RecursiveSelfMemory",
                message=f"Memory event: {event_type}",
                data=data
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.warning(f"发布记忆事件失败: {e}")

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 正确计算大小: content大小 vs 总大小(content+metadata)
        content_size = self._content_size_bytes  # 实际内容总大小
        total_size = content_size + self._metadata_size_bytes
        metadata_ratio = self._metadata_size_bytes / max(total_size, 1)

        return {
            'l0_event_count': len(self.l0_events),
            'l1_metadata_count': len(self.l1_metadata),
            'l2_summary_count': len(self.l2_summaries),
            'l3_strategy_count': len(self.l3_strategies),
            'total_remembered': self._total_remembered,
            'total_forgotten': self._total_forgotten,
            'total_summarized': self._total_summarized,
            'total_size_bytes': total_size,
            'content_size_bytes': content_size,
            'metadata_size_bytes': self._metadata_size_bytes,
            'metadata_overhead_ratio': metadata_ratio,
            'forgetting_records_count': len(self.forgetting_records)
        }

    def export_memories(self, output_path: str, include_forgotten: bool = False):
        """导出记忆到文件"""
        data = {
            'timestamp': time.time(),
            'statistics': self.get_statistics(),
            'l0_events': [e.to_dict() for e in self.l0_events.values()],
            'l2_summaries': [s.to_dict() for s in self.l2_summaries],
            'l3_strategies': [s.to_dict() for s in self.l3_strategies.values()],
        }

        if include_forgotten:
            data['forgetting_records'] = [
                {
                    'memory_id': r.memory_id,
                    'forgotten_at': r.forgotten_at,
                    'reason': r.reason.value,
                    'reason_detail': r.reason_detail,
                    'original_summary': r.original_summary,
                    'transferred_to_summary': r.transferred_to_summary,
                    'summary_id': r.summary_id
                }
                for r in self.forgetting_records
            ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"记忆已导出: {output_path}")

    def load_memories(self, input_path: str):
        """从文件加载记忆"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 恢复L0事件
            for event_data in data.get('l0_events', []):
                # 重建元数据
                metadata_dict = event_data['metadata']
                metadata = MemoryMetadata(
                    memory_id=metadata_dict['memory_id'],
                    created_at=metadata_dict['created_at'],
                    importance=MemoryImportance(metadata_dict['importance']),
                    why_remembered=metadata_dict['why_remembered'],
                    confidence=metadata_dict['confidence'],
                    trigger=metadata_dict['trigger'],
                    access_count=metadata_dict.get('access_count', 0),
                    last_accessed=metadata_dict.get('last_accessed', 0.0),
                    access_frequency=metadata_dict.get('access_frequency', 0.0),
                    related_memories=metadata_dict.get('related_memories', []),
                    tags=set(metadata_dict.get('tags', [])),
                    size_bytes=metadata_dict.get('size_bytes', 0),
                    compressed=metadata_dict.get('compressed', False)
                )

                # 重建事件
                event = EventMemory(
                    id=event_data['id'],
                    timestamp=event_data['timestamp'],
                    event_type=event_data['event_type'],
                    content=event_data['content'],
                    metadata=metadata
                )

                self.l0_events[event.id] = event
                self.l1_metadata[event.id] = metadata

            # 恢复L2摘要
            for summary_data in data.get('l2_summaries', []):
                metadata_dict = summary_data['metadata']
                metadata = MemoryMetadata(
                    memory_id=metadata_dict['memory_id'],
                    created_at=metadata_dict['created_at'],
                    importance=MemoryImportance(metadata_dict['importance']),
                    why_remembered=metadata_dict['why_remembered'],
                    confidence=metadata_dict['confidence'],
                    trigger=metadata_dict['trigger'],
                    size_bytes=metadata_dict.get('size_bytes', 0)
                )

                summary = MemorySummary(
                    id=summary_data['id'],
                    created_at=summary_data['timestamp'],
                    layer=MemoryLayer(summary_data['layer']),
                    summary=summary_data['summary'],
                    key_points=summary_data['key_points'],
                    source_memory_ids=summary_data['source_memory_ids'],
                    source_count=summary_data['source_count'],
                    metadata=metadata
                )

                self.l2_summaries.append(summary)

            logger.info(f"记忆已从 {input_path} 恢复")

        except Exception as e:
            logger.error(f"加载记忆失败: {e}")


# ============================================================================
# 便捷函数
# ============================================================================

def create_memory(event_type: str,
                  content: Dict[str, Any],
                  importance: str = "MEDIUM",
                  why: str = "",
                  confidence: float = 1.0) -> str:
    """
    便捷函数: 创建记忆

    Args:
        event_type: 事件类型
        content: 内容
        importance: 重要性 ("CRITICAL", "HIGH", "MEDIUM", "LOW", "EPHEMERAL")
        why: 为什么记住
        confidence: 置信度

    Returns:
        记忆ID
    """
    # 创建全局记忆系统实例 (如果不存在)
    if not hasattr(create_memory, '_instance'):
        create_memory._instance = RecursiveSelfMemory()

    importance_enum = MemoryImportance(importance.lower())

    return create_memory._instance.remember(
        event_type=event_type,
        content=content,
        importance=importance_enum,
        why=why,
        confidence=confidence
    )


def recall_memory(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    便捷函数: 回忆记忆

    Args:
        query: 查询
        limit: 数量限制

    Returns:
        记忆列表
    """
    if not hasattr(recall_memory, '_instance'):
        recall_memory._instance = RecursiveSelfMemory()

    results = recall_memory._instance.recall(query, limit)

    return [e.to_dict() for e in results]


def why_remembered(memory_id: str) -> Optional[str]:
    """便捷函数: 为何记住"""
    if not hasattr(why_remembered, '_instance'):
        why_remembered._instance = RecursiveSelfMemory()

    return why_remembered._instance.why_remembered(memory_id)


def why_forgotten(memory_id: str) -> Optional[str]:
    """便捷函数: 为何遗忘"""
    if not hasattr(why_forgotten, '_instance'):
        why_forgotten._instance = RecursiveSelfMemory()

    return why_forgotten._instance.why_forgotten(memory_id)
