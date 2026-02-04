#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一事件总线模块 (Unified Event Bus Module)
============================================

解决M1-M4分形AGI组件的事件发布依赖问题。

此模块提供与 agi_component_coordinator.EventBus 兼容但独立的事件总线实现，
专门供 core 包内的组件（如 GoalQuestioner, RecursiveSelfMemory）使用。

设计原则:
- 同步发布: 与M1-M4组件的预期行为一致
- 单例模式: 全局唯一的事件总线实例
- 容错降级: 订阅者异常不影响其他订阅者
- 完整日志: 所有事件操作可追踪

版本: 1.0.0
创建日期: 2026-01-12
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"
    MEMORY = "memory"
    GOAL = "goal"
    INSIGHT = "insight"
    SYSTEM = "system"


@dataclass
class Event:
    """
    标准事件对象
    
    属性:
        type: 事件类型 (EventType枚举)
        source: 事件源组件名称
        message: 事件描述消息
        data: 事件附加数据
        timestamp: 事件创建时间戳
    """
    type: EventType
    source: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'type': self.type.value,
            'source': self.source,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        return f"[{self.type.value}] {self.source}: {self.message}"


class EventBus:
    """
    同步事件总线 - 供M1-M4组件使用
    
    特性:
    - 单例模式: 通过 get_instance() 获取全局实例
    - 同步发布: publish() 立即调用所有订阅者
    - 通配符订阅: 支持 "prefix.*" 和 "*" 模式
    - 事件历史: 保留最近1000条事件用于调试
    
    使用示例:
        from core.event_bus import EventBus, Event, EventType
        
        bus = EventBus.get_instance()
        
        # 订阅
        def handler(event):
            print(f"Received: {event}")
        bus.subscribe("memory_created", handler)
        
        # 发布
        event = Event(
            type=EventType.MEMORY,
            source="RecursiveSelfMemory",
            message="New memory created",
            data={"memory_id": "mem_123"}
        )
        bus.publish(event)
    """
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化事件总线"""
        if self._initialized:
            return
            
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._history: List[Event] = []
        self._max_history = 1000
        self._stats = {
            'total_published': 0,
            'total_delivered': 0,
            'failed_deliveries': 0
        }
        self._initialized = True
        logger.debug("🔗 core.event_bus.EventBus initialized (singleton)")
    
    @classmethod
    def get_instance(cls) -> 'EventBus':
        """获取全局单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """
        订阅事件
        
        Args:
            event_type: 事件类型字符串，支持通配符:
                - "memory_created": 精确匹配
                - "memory.*": 前缀匹配 (匹配 memory_created, memory_deleted 等)
                - "*": 全局匹配 (接收所有事件)
            handler: 事件处理函数，签名: (event: Event) -> None
        """
        self._subscribers[event_type].append(handler)
        handler_name = getattr(handler, '__name__', str(handler))
        logger.debug(f"📬 Subscribed to '{event_type}': {handler_name}")
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]) -> bool:
        """
        取消订阅
        
        Args:
            event_type: 事件类型字符串
            handler: 要移除的处理函数
            
        Returns:
            是否成功移除
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                return True
            except ValueError:
                return False
        return False
    
    def publish(self, event: Event) -> int:
        """
        同步发布事件
        
        Args:
            event: 要发布的事件对象
            
        Returns:
            成功送达的订阅者数量
        """
        self._stats['total_published'] += 1
        
        # 记录历史
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # 收集匹配的处理器
        handlers = set()
        
        # 1. 精确匹配: event_type
        event_key = event_type_to_key(event)
        handlers.update(self._subscribers.get(event_key, []))
        
        # 2. 简单事件类型匹配
        handlers.update(self._subscribers.get(event.type.value, []))
        
        # 3. 通配符匹配: prefix.*
        for pattern, pattern_handlers in self._subscribers.items():
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if event_key.startswith(prefix) or event.type.value.startswith(prefix):
                    handlers.update(pattern_handlers)
        
        # 4. 全局订阅: *
        handlers.update(self._subscribers.get('*', []))
        
        # 分发事件
        delivered = 0
        for handler in handlers:
            try:
                handler(event)
                delivered += 1
                self._stats['total_delivered'] += 1
            except Exception as e:
                self._stats['failed_deliveries'] += 1
                handler_name = getattr(handler, '__name__', str(handler))
                logger.warning(f"⚠️ Event handler '{handler_name}' failed: {e}")
        
        if delivered > 0:
            logger.debug(f"📢 Event '{event_key}' delivered to {delivered} subscriber(s)")
        
        return delivered
    
    def get_history(self, count: int = 100, event_type: Optional[str] = None) -> List[Event]:
        """
        获取事件历史
        
        Args:
            count: 返回的最大事件数
            event_type: 可选的事件类型过滤
            
        Returns:
            事件列表（最新的在前）
        """
        history = self._history[-count:][::-1]
        if event_type:
            history = [e for e in history if e.type.value == event_type]
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计信息"""
        return {
            **self._stats,
            'subscribers_count': sum(len(h) for h in self._subscribers.values()),
            'event_types_count': len(self._subscribers),
            'history_size': len(self._history)
        }
    
    def clear_history(self) -> None:
        """清空事件历史"""
        self._history.clear()
    
    def reset(self) -> None:
        """重置事件总线（仅用于测试）"""
        self._subscribers.clear()
        self._history.clear()
        self._stats = {
            'total_published': 0,
            'total_delivered': 0,
            'failed_deliveries': 0
        }


def event_type_to_key(event: Event) -> str:
    """
    将事件转换为订阅键
    
    格式: {type}_{source} 或 {type}
    """
    return f"{event.type.value}_{event.source}"


# 便捷函数
def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    return EventBus.get_instance()


def publish_event(event_type: EventType, source: str, message: str, 
                  data: Optional[Dict[str, Any]] = None) -> int:
    """
    便捷的事件发布函数
    
    Args:
        event_type: 事件类型
        source: 事件源
        message: 事件消息
        data: 附加数据
        
    Returns:
        送达的订阅者数量
    """
    event = Event(
        type=event_type,
        source=source,
        message=message,
        data=data or {}
    )
    return get_event_bus().publish(event)


# 模块加载时输出调试信息
logger.debug("✅ core.event_bus module loaded")
