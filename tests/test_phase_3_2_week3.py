"""
Phase 3.2 Week 3 - 广播系统测试套件

测试 Week 3 所有组件:
- BroadcastMessage, BroadcastPriority
- PriorityMessageQueue
- MessageFilter, MessageRouter
- EnhancedBroadcaster
- AsyncBroadcaster
- BroadcastHistory

作者: AGI Project Team
创建时间: 2025-01-17
"""

import asyncio
import unittest
from datetime import datetime, timedelta
import time

from phase3_2_self_awareness.broadcast_system import (
    BroadcastPriority,
    BroadcastMessage,
    PriorityMessageQueue,
    MessageFilter,
    MessageRouter,
    EnhancedBroadcaster
)
from phase3_2_self_awareness.async_broadcast import (
    AsyncBroadcaster,
    AsyncBroadcastResult
)
from phase3_2_self_awareness.broadcast_history import (
    BroadcastRecord,
    BroadcastHistory
)


class TestBroadcastMessage(unittest.TestCase):
    """测试 BroadcastMessage 类"""
    
    def test_message_creation(self):
        """测试消息创建"""
        msg = BroadcastMessage(
            content="Test message",
            priority=BroadcastPriority.HIGH,
            source="test_source",
            message_type="test"
        )
        
        self.assertEqual(msg.content, "Test message")
        self.assertEqual(msg.priority, BroadcastPriority.HIGH)
        self.assertEqual(msg.source, "test_source")
        self.assertEqual(msg.message_type, "test")
        self.assertIsNotNone(msg.message_id)
    
    def test_message_ttl_expiration(self):
        """测试消息 TTL 过期"""
        msg = BroadcastMessage(
            content="Expiring message",
            ttl=0.001  # 1 毫秒
        )
        
        self.assertFalse(msg.is_expired())
        time.sleep(0.002)
        self.assertTrue(msg.is_expired())
    
    def test_message_comparison(self):
        """测试消息优先级比较"""
        msg1 = BroadcastMessage("M1", priority=BroadcastPriority.CRITICAL)
        msg2 = BroadcastMessage("M2", priority=BroadcastPriority.LOW)
        
        self.assertTrue(msg1 < msg2)  # 数值小优先级高
    
    def test_message_to_dict(self):
        """测试消息序列化"""
        msg = BroadcastMessage(
            content="Test",
            priority=BroadcastPriority.NORMAL,
            metadata={'key': 'value'}
        )
        
        data = msg.to_dict()
        self.assertEqual(data['content'], "Test")
        self.assertEqual(data['priority'], "NORMAL")
        self.assertEqual(data['metadata'], {'key': 'value'})


class TestPriorityMessageQueue(unittest.TestCase):
    """测试 PriorityMessageQueue 类"""
    
    def setUp(self):
        """测试前准备"""
        self.queue = PriorityMessageQueue(max_size=10)
    
    def test_queue_push_pop(self):
        """测试入队出队"""
        msg1 = BroadcastMessage("Msg1", priority=BroadcastPriority.LOW)
        msg2 = BroadcastMessage("Msg2", priority=BroadcastPriority.CRITICAL)
        
        self.assertTrue(self.queue.push(msg1))
        self.assertTrue(self.queue.push(msg2))
        self.assertEqual(self.queue.size(), 2)
        
        # 应该先弹出 CRITICAL (优先级最高)
        popped = self.queue.pop()
        self.assertEqual(popped.priority, BroadcastPriority.CRITICAL)
        
        popped = self.queue.pop()
        self.assertEqual(popped.priority, BroadcastPriority.LOW)
    
    def test_queue_capacity_limit(self):
        """测试队列容量限制"""
        # 填满队列
        for i in range(11):
            msg = BroadcastMessage(f"Msg{i}", priority=BroadcastPriority.NORMAL)
            self.queue.push(msg)
        
        # 队列应该最多有 10 条消息
        self.assertEqual(self.queue.size(), 10)
    
    def test_queue_duplicate_prevention(self):
        """测试重复消息过滤"""
        msg = BroadcastMessage("Unique", message_id="test_id_123")
        
        self.assertTrue(self.queue.push(msg))
        self.assertFalse(self.queue.push(msg))  # 重复消息
        self.assertEqual(self.queue.size(), 1)
    
    def test_queue_expired_cleanup(self):
        """测试过期消息清理"""
        msg1 = BroadcastMessage("Expire", ttl=0.001)
        msg2 = BroadcastMessage("Keep", ttl=None)
        
        self.queue.push(msg1)
        self.queue.push(msg2)
        
        time.sleep(0.002)
        
        # peek 会触发清理
        peeked = self.queue.peek()
        self.assertEqual(peeked.content, "Keep")
        self.assertEqual(self.queue.size(), 1)
    
    def test_queue_statistics(self):
        """测试队列统计"""
        self.queue.push(BroadcastMessage("M1", priority=BroadcastPriority.HIGH))
        self.queue.push(BroadcastMessage("M2", priority=BroadcastPriority.HIGH))
        self.queue.push(BroadcastMessage("M3", priority=BroadcastPriority.LOW))
        
        stats = self.queue.get_statistics()
        self.assertEqual(stats['size'], 3)
        self.assertEqual(stats['priority_distribution']['HIGH'], 2)
        self.assertEqual(stats['priority_distribution']['LOW'], 1)


class TestMessageFilter(unittest.TestCase):
    """测试 MessageFilter 类"""
    
    def setUp(self):
        """测试前准备"""
        self.filter = MessageFilter()
    
    def test_priority_filter(self):
        """测试优先级过滤"""
        self.filter.set_priority_filter([BroadcastPriority.CRITICAL, BroadcastPriority.URGENT])
        
        msg_pass = BroadcastMessage("Pass", priority=BroadcastPriority.CRITICAL)
        msg_block = BroadcastMessage("Block", priority=BroadcastPriority.LOW)
        
        self.assertTrue(self.filter.should_pass(msg_pass))
        self.assertFalse(self.filter.should_pass(msg_block))
    
    def test_type_filter(self):
        """测试类型过滤"""
        self.filter.set_type_filter(["important", "critical"])
        
        msg_pass = BroadcastMessage("Pass", message_type="important")
        msg_block = BroadcastMessage("Block", message_type="normal")
        
        self.assertTrue(self.filter.should_pass(msg_pass))
        self.assertFalse(self.filter.should_pass(msg_block))
    
    def test_source_filter(self):
        """测试来源过滤"""
        self.filter.set_source_filter(["trusted_source"])
        
        msg_pass = BroadcastMessage("Pass", source="trusted_source")
        msg_block = BroadcastMessage("Block", source="unknown_source")
        
        self.assertTrue(self.filter.should_pass(msg_pass))
        self.assertFalse(self.filter.should_pass(msg_block))
    
    def test_custom_predicate(self):
        """测试自定义谓词过滤"""
        # 只允许内容长度 > 5 的消息
        self.filter.add_predicate(lambda msg: len(str(msg.content)) > 5)
        
        msg_pass = BroadcastMessage("Long message")
        msg_block = BroadcastMessage("Hi")
        
        self.assertTrue(self.filter.should_pass(msg_pass))
        self.assertFalse(self.filter.should_pass(msg_block))
    
    def test_combined_filters(self):
        """测试组合过滤"""
        self.filter.set_priority_filter([BroadcastPriority.HIGH])
        self.filter.set_type_filter(["test"])
        
        msg_pass = BroadcastMessage("Pass", priority=BroadcastPriority.HIGH, message_type="test")
        msg_block1 = BroadcastMessage("Block1", priority=BroadcastPriority.LOW, message_type="test")
        msg_block2 = BroadcastMessage("Block2", priority=BroadcastPriority.HIGH, message_type="other")
        
        self.assertTrue(self.filter.should_pass(msg_pass))
        self.assertFalse(self.filter.should_pass(msg_block1))
        self.assertFalse(self.filter.should_pass(msg_block2))


class TestMessageRouter(unittest.TestCase):
    """测试 MessageRouter 类"""
    
    def setUp(self):
        """测试前准备"""
        self.router = MessageRouter()
    
    def test_group_management(self):
        """测试订阅组管理"""
        sub1 = lambda msg: None
        sub2 = lambda msg: None
        
        self.router.create_group("group1")
        self.router.add_subscriber_to_group("group1", sub1)
        self.router.add_subscriber_to_group("group1", sub2)
        
        self.assertEqual(len(self.router.groups["group1"]), 2)
        
        self.router.remove_subscriber_from_group("group1", sub1)
        self.assertEqual(len(self.router.groups["group1"]), 1)
    
    def test_routing_rules(self):
        """测试路由规则"""
        sub1 = lambda msg: None
        sub2 = lambda msg: None
        
        self.router.add_subscriber_to_group("critical_group", sub1)
        self.router.add_subscriber_to_group("normal_group", sub2)
        
        # 规则: CRITICAL 消息路由到 critical_group
        self.router.add_rule(
            lambda msg: msg.priority == BroadcastPriority.CRITICAL,
            ["critical_group"]
        )
        
        msg_critical = BroadcastMessage("Critical", priority=BroadcastPriority.CRITICAL)
        msg_normal = BroadcastMessage("Normal", priority=BroadcastPriority.NORMAL)
        
        routed_critical = self.router.route(msg_critical)
        routed_normal = self.router.route(msg_normal)
        
        self.assertEqual(len(routed_critical), 1)
        self.assertIn(sub1, routed_critical)
        self.assertEqual(len(routed_normal), 0)
    
    def test_router_statistics(self):
        """测试路由统计"""
        self.router.create_group("group1")
        self.router.create_group("group2")
        self.router.add_subscriber_to_group("group1", lambda: None)
        self.router.add_rule(lambda msg: True, ["group1"])
        
        stats = self.router.get_statistics()
        self.assertEqual(stats['num_groups'], 2)
        self.assertEqual(stats['num_rules'], 1)


class TestEnhancedBroadcaster(unittest.TestCase):
    """测试 EnhancedBroadcaster 类"""
    
    def setUp(self):
        """测试前准备"""
        self.broadcaster = EnhancedBroadcaster(max_queue_size=100)
        self.received_messages = []
    
    def test_enqueue_and_broadcast(self):
        """测试入队和广播"""
        # 设置订阅者
        self.broadcaster.router.create_group("test_group")
        self.broadcaster.router.add_subscriber_to_group(
            "test_group",
            lambda msg: self.received_messages.append(msg)
        )
        
        # 设置路由规则
        self.broadcaster.router.add_rule(
            lambda msg: msg.message_type == "test",
            ["test_group"]
        )
        
        # 入队并广播
        msg = BroadcastMessage("Test", message_type="test")
        self.broadcaster.enqueue(msg)
        success = self.broadcaster.broadcast_next()
        
        self.assertTrue(success)
        self.assertEqual(len(self.received_messages), 1)
        self.assertEqual(self.received_messages[0].content, "Test")
    
    def test_filter_integration(self):
        """测试过滤器集成"""
        # 设置过滤器: 只允许 HIGH 优先级
        self.broadcaster.filter.set_priority_filter([BroadcastPriority.HIGH])
        
        msg_high = BroadcastMessage("High", priority=BroadcastPriority.HIGH)
        msg_low = BroadcastMessage("Low", priority=BroadcastPriority.LOW)
        
        self.assertTrue(self.broadcaster.enqueue(msg_high))
        self.assertFalse(self.broadcaster.enqueue(msg_low))  # 被过滤
        
        self.assertEqual(self.broadcaster.queue.size(), 1)
    
    def test_batch_broadcast(self):
        """测试批量广播"""
        # 设置订阅者
        self.broadcaster.router.create_group("batch_group")
        self.broadcaster.router.add_subscriber_to_group(
            "batch_group",
            lambda msg: self.received_messages.append(msg)
        )
        self.broadcaster.router.add_rule(lambda msg: True, ["batch_group"])
        
        # 入队多条消息
        for i in range(5):
            msg = BroadcastMessage(f"Msg{i}")
            self.broadcaster.enqueue(msg)
        
        # 批量广播
        count = self.broadcaster.broadcast_all(max_messages=10)
        
        self.assertEqual(count, 5)
        self.assertEqual(len(self.received_messages), 5)
    
    def test_statistics(self):
        """测试统计功能"""
        msg = BroadcastMessage("Test")
        self.broadcaster.enqueue(msg)
        
        stats = self.broadcaster.get_statistics()
        self.assertIn('broadcaster', stats)
        self.assertIn('queue', stats)
        self.assertIn('router', stats)
        self.assertEqual(stats['broadcaster']['total_messages'], 1)


class TestAsyncBroadcaster(unittest.TestCase):
    """测试 AsyncBroadcaster 类"""
    
    def setUp(self):
        """测试前准备"""
        self.broadcaster = AsyncBroadcaster(max_queue_size=100)
        self.received_messages = []
    
    def test_async_enqueue(self):
        """测试异步入队"""
        async def test():
            msg = BroadcastMessage("Async test")
            success = await self.broadcaster.enqueue(msg)
            self.assertTrue(success)
            self.assertEqual(self.broadcaster.queue.size(), 1)
        
        asyncio.run(test())
    
    def test_async_subscriber(self):
        """测试异步订阅者"""
        async def async_subscriber(msg):
            await asyncio.sleep(0.01)
            self.received_messages.append(msg)
        
        async def test():
            # 添加异步订阅者
            self.broadcaster.add_async_subscriber("async_group", async_subscriber)
            
            # 设置路由
            self.broadcaster.router.add_rule(
                lambda msg: True,
                ["async_group"]
            )
            
            # 广播消息
            msg = BroadcastMessage("Async message")
            result = await self.broadcaster.broadcast_message(msg)
            
            self.assertEqual(result.success_count, 1)
            self.assertEqual(len(self.received_messages), 1)
        
        asyncio.run(test())
    
    def test_sync_async_mix(self):
        """测试同步和异步订阅者混合"""
        async def async_sub(msg):
            await asyncio.sleep(0.01)
            self.received_messages.append(("async", msg))
        
        def sync_sub(msg):
            self.received_messages.append(("sync", msg))
        
        async def test():
            # 添加订阅者
            self.broadcaster.add_async_subscriber("async_group", async_sub)
            self.broadcaster.router.add_subscriber_to_group("sync_group", sync_sub)
            
            # 设置路由
            self.broadcaster.router.add_rule(
                lambda msg: True,
                ["async_group", "sync_group"]
            )
            
            # 广播消息
            msg = BroadcastMessage("Mixed message")
            result = await self.broadcaster.broadcast_message(msg)
            
            self.assertEqual(result.success_count, 2)
            self.assertEqual(len(self.received_messages), 2)
        
        asyncio.run(test())
    
    def test_broadcast_timeout(self):
        """测试广播超时"""
        async def slow_subscriber(msg):
            await asyncio.sleep(10)  # 超过默认 timeout (5s)
        
        async def test():
            self.broadcaster.add_async_subscriber("slow_group", slow_subscriber)
            self.broadcaster.router.add_rule(
                lambda msg: True,
                ["slow_group"]
            )
            
            msg = BroadcastMessage("Timeout test")
            result = await self.broadcaster.broadcast_message(msg)
            
            # 应该超时失败
            self.assertEqual(result.failure_count, 1)
        
        asyncio.run(test())


class TestBroadcastHistory(unittest.TestCase):
    """测试 BroadcastHistory 类"""
    
    def setUp(self):
        """测试前准备"""
        self.history = BroadcastHistory(max_records=100)
    
    def test_add_record(self):
        """测试添加记录"""
        msg = BroadcastMessage("Test", priority=BroadcastPriority.HIGH)
        record = BroadcastRecord(
            message=msg,
            subscriber_count=10,
            success_count=8,
            failure_count=2
        )
        
        self.history.add_record(record)
        self.assertEqual(len(self.history.records), 1)
    
    def test_query_by_priority(self):
        """测试按优先级查询"""
        for i in range(5):
            msg = BroadcastMessage(f"M{i}", priority=BroadcastPriority.HIGH)
            self.history.add_record(BroadcastRecord(message=msg))
        
        msg_low = BroadcastMessage("Low", priority=BroadcastPriority.LOW)
        self.history.add_record(BroadcastRecord(message=msg_low))
        
        high_records = self.history.get_records_by_priority(BroadcastPriority.HIGH)
        self.assertEqual(len(high_records), 5)
    
    def test_time_range_query(self):
        """测试时间范围查询"""
        now = datetime.now()
        
        # 添加不同时间的记录
        msg1 = BroadcastMessage("M1")
        record1 = BroadcastRecord(message=msg1, timestamp=now - timedelta(hours=2))
        self.history.add_record(record1)
        
        msg2 = BroadcastMessage("M2")
        record2 = BroadcastRecord(message=msg2, timestamp=now - timedelta(hours=1))
        self.history.add_record(record2)
        
        # 查询最近 1.5 小时的记录
        start = now - timedelta(hours=1, minutes=30)
        end = now
        records = self.history.get_records_in_time_range(start, end)
        
        self.assertEqual(len(records), 1)
    
    def test_pattern_analysis(self):
        """测试模式分析"""
        # 添加多条记录
        for i in range(10):
            msg = BroadcastMessage(
                f"M{i}",
                priority=BroadcastPriority.HIGH if i < 5 else BroadcastPriority.LOW,
                message_type="test"
            )
            record = BroadcastRecord(
                message=msg,
                subscriber_count=10,
                success_count=8
            )
            self.history.add_record(record)
        
        analysis = self.history.analyze_broadcast_patterns()
        
        self.assertEqual(analysis['total_broadcasts'], 10)
        self.assertIn('priority_distribution', analysis)
        self.assertIn('HIGH', analysis['priority_distribution'])
    
    def test_failure_analysis(self):
        """测试失败分析"""
        # 添加成功记录
        msg_success = BroadcastMessage("Success")
        record_success = BroadcastRecord(
            message=msg_success,
            subscriber_count=10,
            success_count=10
        )
        self.history.add_record(record_success)
        
        # 添加失败记录
        msg_fail = BroadcastMessage("Fail", source="bad_source")
        record_fail = BroadcastRecord(
            message=msg_fail,
            subscriber_count=10,
            success_count=2,
            failure_count=8,
            errors=["Error 1", "Error 2"]
        )
        self.history.add_record(record_fail)
        
        failure_analysis = self.history.get_failure_analysis(min_failure_rate=0.5)
        
        self.assertEqual(failure_analysis['total_failures'], 1)
        self.assertIn('bad_source', failure_analysis['source_failures'])
    
    def test_summary_report(self):
        """测试摘要报告"""
        for i in range(5):
            msg = BroadcastMessage(f"M{i}")
            record = BroadcastRecord(
                message=msg,
                subscriber_count=10,
                success_count=9
            )
            self.history.add_record(record)
        
        report = self.history.generate_summary_report()
        
        self.assertEqual(report['summary']['total_broadcasts'], 5)
        self.assertEqual(report['notification_stats']['total_subscribers'], 50)
        self.assertEqual(report['notification_stats']['total_successes'], 45)


class TestIntegrationFlow(unittest.TestCase):
    """测试端到端集成流程"""
    
    def test_full_broadcast_flow(self):
        """测试完整广播流程"""
        # 创建广播器和历史记录器
        broadcaster = EnhancedBroadcaster()
        history = BroadcastHistory()
        received = []
        
        # 设置订阅者
        def subscriber(msg):
            received.append(msg)
        
        broadcaster.router.create_group("main")
        broadcaster.router.add_subscriber_to_group("main", subscriber)
        broadcaster.router.add_rule(lambda msg: True, ["main"])
        
        # 发送不同优先级的消息
        messages = [
            BroadcastMessage("Critical", priority=BroadcastPriority.CRITICAL),
            BroadcastMessage("Normal", priority=BroadcastPriority.NORMAL),
            BroadcastMessage("Low", priority=BroadcastPriority.LOW)
        ]
        
        for msg in messages:
            broadcaster.enqueue(msg)
        
        # 广播所有消息
        count = broadcaster.broadcast_all()
        self.assertEqual(count, 3)
        
        # 验证优先级顺序 (CRITICAL 应该先被广播)
        self.assertEqual(received[0].priority, BroadcastPriority.CRITICAL)
        self.assertEqual(received[1].priority, BroadcastPriority.NORMAL)
        self.assertEqual(received[2].priority, BroadcastPriority.LOW)
        
        # 记录到历史
        for msg in received:
            record = BroadcastRecord(
                message=msg,
                subscriber_count=1,
                success_count=1
            )
            history.add_record(record)
        
        # 验证历史记录
        self.assertEqual(len(history.records), 3)
        report = history.generate_summary_report()
        self.assertEqual(report['summary']['total_broadcasts'], 3)


if __name__ == '__main__':
    unittest.main()
