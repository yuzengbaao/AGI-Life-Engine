"""
Phase 3.2 Week 4 测试套件 - GWT 模块集成测试

测试范围:
1. EnhancedGlobalWorkspace - 增强型全局工作空间
2. PersistenceManager - 持久化管理器
3. 集成测试 - 完整工作流
4. 性能验证 - 压力测试

作者: AGI Project Team
创建时间: 2025-01-17
版本: 1.0.0
"""

import unittest
import tempfile
import shutil
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta

from phase3_2_self_awareness.enhanced_workspace import EnhancedGlobalWorkspace
from phase3_2_self_awareness.persistence import PersistenceManager
from phase3_2_self_awareness.global_workspace import ConsciousnessLevel
from phase3_2_self_awareness.broadcast_system import BroadcastPriority, BroadcastMessage
from phase3_2_self_awareness.performance_test import PerformanceTester


class TestEnhancedGlobalWorkspace(unittest.TestCase):
    """测试增强型全局工作空间"""
    
    def setUp(self):
        """测试前准备"""
        self.workspace = EnhancedGlobalWorkspace(
            capacity=7,
            enable_async=False,  # 使用同步模式便于测试
            enable_history=True,
            max_queue_size=100
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.workspace.broadcaster)
        self.assertIsNotNone(self.workspace.history)
        self.assertEqual(len(self.workspace._consciousness_priority_map), 4)
        
        # 验证默认路由
        stats = self.workspace.broadcaster.router.get_statistics()
        self.assertEqual(stats['num_groups'], 4)
        self.assertEqual(stats['num_rules'], 4)
    
    def test_subscribe_with_group(self):
        """测试分组订阅"""
        received = []
        
        def callback(msg):
            received.append(msg)
        
        # 订阅特定组
        self.workspace.subscribe(callback, group="critical_events")
        
        # 发送高优先级消息 (应该路由到 critical_events)
        self.workspace.broadcast_enhanced(
            content="Critical alert",
            message_type="alert",
            priority=BroadcastPriority.CRITICAL
        )
        
        # 处理队列
        processed = self.workspace.process_broadcast_queue()
        self.assertGreater(processed, 0)
        self.assertGreater(len(received), 0)
    
    def test_broadcast_enhanced(self):
        """测试增强广播"""
        # 测试不同优先级
        priorities = [
            BroadcastPriority.CRITICAL,
            BroadcastPriority.URGENT,
            BroadcastPriority.HIGH,
            BroadcastPriority.NORMAL,
            BroadcastPriority.LOW
        ]
        
        for priority in priorities:
            result = self.workspace.broadcast_enhanced(
                content=f"Message with {priority.name}",
                message_type="test",
                priority=priority
            )
            self.assertTrue(result)
        
        # 验证队列中有消息
        stats = self.workspace.broadcaster.get_statistics()
        self.assertEqual(stats['queue']['size'], 5)
    
    def test_backward_compatible_broadcast(self):
        """测试向后兼容的广播"""
        received = []
        
        def callback(msg):
            received.append(msg)
        
        self.workspace.subscribe(callback)
        
        # 使用原始 broadcast 方法
        self.workspace.broadcast("Test information", priority=0.8)
        
        # 处理队列
        self.workspace.process_broadcast_queue()
        
        # 验证消息被接收 (通过原始订阅者列表)
        self.assertGreater(len(received), 0)
    
    def test_consciousness_priority_mapping(self):
        """测试意识级别到优先级的映射"""
        # 设置不同的意识级别
        levels = [
            (ConsciousnessLevel.UNCONSCIOUS, BroadcastPriority.LOW),
            (ConsciousnessLevel.PRECONSCIOUS, BroadcastPriority.NORMAL),
            (ConsciousnessLevel.CONSCIOUS, BroadcastPriority.HIGH),
            (ConsciousnessLevel.METACONSCIOUS, BroadcastPriority.URGENT)
        ]
        
        for level, expected_priority in levels:
            # 更新意识级别
            self.workspace.current_state.level = level
            
            # 广播 (不指定优先级,应自动使用映射)
            self.workspace.broadcast_enhanced(
                content=f"Message at {level.name}",
                message_type="test"
            )
        
        stats = self.workspace.broadcaster.get_statistics()
        self.assertEqual(stats['queue']['size'], 4)
    
    def test_process_broadcast_queue_with_history(self):
        """测试队列处理和历史记录"""
        # 订阅者
        received_count = 0
        
        def callback(msg):
            nonlocal received_count
            received_count += 1
        
        self.workspace.subscribe(callback)
        
        # 发送消息
        for i in range(10):
            self.workspace.broadcast_enhanced(
                content=f"Message {i}",
                message_type="test",
                priority=BroadcastPriority.NORMAL
            )
        
        # 处理队列
        processed = self.workspace.process_broadcast_queue(max_messages=20)
        self.assertEqual(processed, 10)
        self.assertEqual(received_count, 10)
        
        # 验证历史记录
        history_analysis = self.workspace.get_broadcast_history_analysis()
        self.assertGreater(history_analysis['total_broadcasts'], 0)
    
    def test_update_focus_with_broadcast(self):
        """测试焦点更新并广播"""
        received = []
        
        def callback(msg):
            received.append(msg)
        
        self.workspace.subscribe(callback, group="attention_updates")
        
        # 更新焦点
        new_focus = ["task1", "task2", "task3"]
        self.workspace.update_focus(new_focus, reason="test")
        
        # 处理队列
        self.workspace.process_broadcast_queue()
        
        # 验证焦点更新消息
        attention_messages = [
            msg for msg in received 
            if hasattr(msg, 'content') and isinstance(msg.content, dict) 
            and msg.content.get('type') == 'focus_update'
        ]
        self.assertGreater(len(attention_messages), 0)
    
    def test_allocate_attention_with_broadcast(self):
        """测试注意力分配并广播"""
        received = []
        
        def callback(msg):
            received.append(msg)
        
        self.workspace.subscribe(callback, group="attention_updates")
        
        # 分配注意力
        tasks = {
            "task1": 0.8,
            "task2": 0.6,
            "task3": 0.4,
            "task4": 0.2
        }
        self.workspace.allocate_attention(tasks)
        
        # 处理队列
        self.workspace.process_broadcast_queue()
        
        # 验证权重更新消息
        weight_messages = [
            msg for msg in received 
            if hasattr(msg, 'content') and isinstance(msg.content, dict)
            and msg.content.get('type') == 'attention_allocated'
        ]
        self.assertGreater(len(weight_messages), 0)
    
    def test_get_broadcast_statistics(self):
        """测试获取广播统计"""
        # 发送一些消息
        for i in range(5):
            self.workspace.broadcast_enhanced(
                content=f"Message {i}",
                message_type="test"
            )
        
        stats = self.workspace.get_broadcast_statistics()
        self.assertIn('broadcaster', stats)
        self.assertEqual(stats['queue_size'], 5)
    
    def test_export_history(self):
        """测试导出历史"""
        # 发送并处理消息
        self.workspace.subscribe(lambda msg: None)
        for i in range(3):
            self.workspace.broadcast_enhanced(
                content=f"Message {i}",
                message_type="test"
            )
        self.workspace.process_broadcast_queue()
        
        # 导出历史
        history = self.workspace.export_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
    
    def test_reset(self):
        """测试重置"""
        # 发送消息
        for i in range(5):
            self.workspace.broadcast_enhanced(
                content=f"Message {i}",
                message_type="test"
            )
        
        # 重置
        self.workspace.reset()
        
        # 验证队列清空
        stats = self.workspace.broadcaster.get_statistics()
        self.assertEqual(stats['queue']['size'], 0)


class TestPersistenceManager(unittest.TestCase):
    """测试持久化管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PersistenceManager(storage_dir=self.temp_dir)
        
        # 创建测试工作空间
        self.workspace = EnhancedGlobalWorkspace(
            capacity=5,
            enable_async=False,
            enable_history=True
        )
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_directory_creation(self):
        """测试目录创建"""
        self.assertTrue(self.manager.snapshots_dir.exists())
        self.assertTrue(self.manager.history_dir.exists())
        self.assertTrue(self.manager.checkpoints_dir.exists())
    
    def test_save_and_load_consciousness_state(self):
        """测试保存和加载意识状态"""
        # 更新工作空间状态
        self.workspace.update_focus(["task1", "task2"], reason="test")
        self.workspace.allocate_attention({"task1": 0.7, "task2": 0.3})
        
        state = self.workspace.current_state
        
        # 保存状态
        filename = self.manager.save_consciousness_state(state)
        self.assertIsNotNone(filename)
        
        # 加载状态
        loaded_state = self.manager.load_consciousness_state(filename)
        self.assertEqual(loaded_state.level, state.level)
        self.assertEqual(loaded_state.focus, state.focus)
        self.assertEqual(loaded_state.attention_weights, state.attention_weights)
    
    def test_serialize_complex_objects(self):
        """测试复杂对象序列化"""
        # 添加复杂对象到工作记忆
        self.workspace.current_state.working_memory['complex_obj'] = {
            'data': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        self.workspace.current_state.working_memory['function'] = lambda x: x * 2
        
        state = self.workspace.current_state
        
        # 保存并加载
        filename = self.manager.save_consciousness_state(state)
        loaded_state = self.manager.load_consciousness_state(filename)
        
        # 验证复杂对象
        self.assertEqual(
            loaded_state.working_memory['complex_obj'],
            state.working_memory['complex_obj']
        )
        # Lambda 函数应该被序列化为 pickled 对象
        self.assertIn('function', loaded_state.working_memory)
    
    def test_save_and_load_broadcast_history(self):
        """测试保存和加载广播历史"""
        # 生成一些历史
        self.workspace.subscribe(lambda msg: None)
        for i in range(5):
            self.workspace.broadcast_enhanced(
                content=f"Message {i}",
                message_type="test"
            )
        self.workspace.process_broadcast_queue()
        
        # 保存历史
        filename = self.manager.save_broadcast_history(self.workspace.history)
        self.assertIsNotNone(filename)
        
        # 加载历史
        records = self.manager.load_broadcast_history(filename)
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 5)
    
    def test_save_and_load_attention_weights(self):
        """测试保存和加载注意力权重"""
        weights = {"task1": 0.5, "task2": 0.3, "task3": 0.2}
        
        # 保存权重
        filename = self.manager.save_attention_weights(weights)
        self.assertIsNotNone(filename)
        
        # 加载权重
        loaded_weights = self.manager.load_attention_weights(filename)
        self.assertEqual(loaded_weights, weights)
    
    def test_save_and_load_checkpoint(self):
        """测试保存和加载检查点"""
        # 准备工作空间状态
        self.workspace.update_focus(["task1", "task2"], reason="test")
        self.workspace.allocate_attention({"task1": 0.7, "task2": 0.3})
        self.workspace.subscribe(lambda msg: None)
        for i in range(3):
            self.workspace.broadcast_enhanced(
                content=f"Message {i}",
                message_type="test"
            )
        self.workspace.process_broadcast_queue()
        
        state = self.workspace.current_state
        weights = self.workspace.current_state.attention_weights
        history = self.workspace.history
        
        # 保存检查点
        checkpoint_name = self.manager.save_checkpoint(
            state=state,
            weights=weights,
            history=history,
            checkpoint_name="test_checkpoint"
        )
        self.assertIsNotNone(checkpoint_name)
        
        # 加载检查点
        checkpoint_data = self.manager.load_checkpoint(checkpoint_name)
        self.assertIn('metadata', checkpoint_data)
        self.assertIn('state', checkpoint_data)
        self.assertIn('weights', checkpoint_data)
        self.assertIn('history_records', checkpoint_data)
        
        # 验证数据
        loaded_state = checkpoint_data['state']
        self.assertEqual(loaded_state.level, state.level)
        self.assertEqual(checkpoint_data['weights'], weights)
        # 历史记录可能包含 update_focus 和 allocate_attention 发送的额外消息
        self.assertGreaterEqual(len(checkpoint_data['history_records']), 3)
    
    def test_list_operations(self):
        """测试列表操作"""
        # 保存一些数据
        self.manager.save_consciousness_state(self.workspace.current_state)
        self.manager.save_attention_weights({"task1": 0.5})
        
        # 列表快照
        snapshots = self.manager.list_snapshots()
        self.assertEqual(len(snapshots), 1)
        
        # 列表权重 (在独立 weights 目录中)
        weights_dir = self.manager.storage_dir / "weights"
        if weights_dir.exists():
            weights_files = list(weights_dir.glob("weights_*.json"))
            self.assertGreater(len(weights_files), 0)
    
    def test_delete_operations(self):
        """测试删除操作"""
        # 保存快照
        filename = self.manager.save_consciousness_state(self.workspace.current_state)
        self.assertTrue((self.manager.snapshots_dir / filename).exists())
        
        # 删除快照
        self.manager.delete_snapshot(filename)
        self.assertFalse((self.manager.snapshots_dir / filename).exists())
    
    def test_get_storage_info(self):
        """测试获取存储信息"""
        # 保存一些数据
        self.manager.save_consciousness_state(self.workspace.current_state)
        self.manager.save_checkpoint(
            state=self.workspace.current_state,
            weights={"task1": 0.5}
        )
        
        info = self.manager.get_storage_info()
        self.assertIn('total_size_mb', info)
        self.assertIn('snapshots_count', info)
        self.assertIn('checkpoints_count', info)
        self.assertGreater(info['snapshots_count'], 0)
        self.assertGreater(info['checkpoints_count'], 0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = EnhancedGlobalWorkspace(
            capacity=7,
            enable_async=False,
            enable_history=True
        )
        self.manager = PersistenceManager(storage_dir=self.temp_dir)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 订阅消息
        received_messages = []
        self.workspace.subscribe(lambda msg: received_messages.append(msg))
        
        # 2. 更新焦点
        self.workspace.update_focus(["task1", "task2", "task3"], reason="integration_test")
        
        # 3. 分配注意力
        tasks = {"task1": 0.6, "task2": 0.3, "task3": 0.1}
        self.workspace.allocate_attention(tasks)
        
        # 4. 发送各种优先级消息
        for priority in [BroadcastPriority.CRITICAL, BroadcastPriority.NORMAL, BroadcastPriority.LOW]:
            self.workspace.broadcast_enhanced(
                content=f"Message with {priority.name}",
                message_type="integration_test",
                priority=priority
            )
        
        # 5. 处理队列
        processed = self.workspace.process_broadcast_queue()
        self.assertGreater(processed, 0)
        self.assertGreater(len(received_messages), 0)
        
        # 6. 保存检查点
        checkpoint_name = self.manager.save_checkpoint(
            state=self.workspace.current_state,
            weights=self.workspace.current_state.attention_weights,
            history=self.workspace.history
        )
        self.assertIsNotNone(checkpoint_name)
        
        # 7. 加载检查点
        checkpoint_data = self.manager.load_checkpoint(checkpoint_name)
        self.assertIsNotNone(checkpoint_data['state'])
        self.assertEqual(checkpoint_data['weights'], tasks)
    
    def test_checkpoint_restore(self):
        """测试检查点恢复"""
        # 准备初始状态
        self.workspace.update_focus(["task1", "task2"], reason="checkpoint_test")
        self.workspace.allocate_attention({"task1": 0.7, "task2": 0.3})
        original_level = self.workspace.current_state.level
        
        # 保存检查点
        checkpoint_name = self.manager.save_checkpoint(
            state=self.workspace.current_state,
            weights=self.workspace.current_state.attention_weights
        )
        
        # 修改状态
        self.workspace.update_focus(["task3", "task4"], reason="modification")
        self.workspace.allocate_attention({"task3": 0.5, "task4": 0.5})
        
        # 加载检查点
        checkpoint_data = self.manager.load_checkpoint(checkpoint_name)
        
        # 验证恢复
        self.assertEqual(checkpoint_data['state'].level, original_level)
        self.assertEqual(checkpoint_data['weights'], {"task1": 0.7, "task2": 0.3})
    
    def test_priority_ordering_end_to_end(self):
        """测试端到端优先级排序"""
        received_priorities = []
        
        def callback(msg):
            if hasattr(msg, 'priority'):
                received_priorities.append(msg.priority)
        
        self.workspace.subscribe(callback)
        
        # 发送乱序消息
        priorities = [
            BroadcastPriority.LOW,
            BroadcastPriority.CRITICAL,
            BroadcastPriority.NORMAL,
            BroadcastPriority.URGENT,
            BroadcastPriority.HIGH
        ]
        
        for priority in priorities:
            self.workspace.broadcast_enhanced(
                content=f"Message {priority.name}",
                message_type="priority_test",
                priority=priority
            )
        
        # 处理队列 (应该按优先级顺序)
        self.workspace.process_broadcast_queue()
        
        # 验证顺序 (CRITICAL < URGENT < HIGH < NORMAL < LOW)
        expected_order = [
            BroadcastPriority.CRITICAL,
            BroadcastPriority.URGENT,
            BroadcastPriority.HIGH,
            BroadcastPriority.NORMAL,
            BroadcastPriority.LOW
        ]
        self.assertEqual(received_priorities, expected_order)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.workspace = EnhancedGlobalWorkspace(
            capacity=7,
            enable_async=False,
            enable_history=True,
            max_queue_size=5000
        )
        self.tester = PerformanceTester(self.workspace)
    
    def test_concurrent_tasks_performance(self):
        """测试并发任务性能"""
        metrics = self.tester.test_concurrent_tasks(num_tasks=100)
        
        # 验证性能指标
        self.assertLess(metrics.avg_latency_ms, 100)  # 平均延迟 < 100ms
        self.assertGreater(metrics.throughput, 10)  # 吞吐量 > 10 ops/s
        self.assertLess(metrics.memory_used_mb, 50)  # 内存使用 < 50MB
    
    def test_message_broadcast_performance(self):
        """测试消息广播性能"""
        metrics = self.tester.test_message_broadcast(
            num_messages=1000,
            num_subscribers=10
        )
        
        # 验证性能指标
        self.assertLess(metrics.avg_latency_ms, 50)  # 平均延迟 < 50ms
        self.assertGreater(metrics.throughput, 100)  # 吞吐量 > 100 msg/s
        self.assertLess(metrics.memory_used_mb, 100)  # 内存使用 < 100MB


if __name__ == '__main__':
    # 配置日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    unittest.main(verbosity=2)
