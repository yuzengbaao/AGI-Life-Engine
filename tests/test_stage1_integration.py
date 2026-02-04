"""
Phase 3.2 Stage 1 集成测试 - AGI意识层集成验证

测试范围:
1. AGIConsciousnessLayer初始化
2. 任务注册与意识状态映射
3. 注意力机制优先级计算
4. 意识水平转换
5. 性能指标 (<10%开销)
6. 与AGI主系统集成验证

作者: GitHub Copilot (Claude Sonnet 4.5)
创建时间: 2025-11-22
版本: 1.0.0
"""

import unittest
import sys
import torch
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agi_consciousness_integration import (
    AGIConsciousnessLayer,
    get_consciousness_layer,
    reset_consciousness_layer
)
from phase3_2_self_awareness import ConsciousnessLevel


class TestAGIConsciousnessLayerInitialization(unittest.TestCase):
    """测试AGI意识层初始化"""
    
    def test_initialization(self):
        """测试基础初始化"""
        layer = AGIConsciousnessLayer()
        
        self.assertIsNotNone(layer.workspace)
        self.assertIsNotNone(layer.attention)
        self.assertEqual(layer.workspace.capacity, 7)  # Miller's Law
        self.assertTrue(layer.workspace.enable_async)  # 异步模式
        self.assertEqual(layer.attention.num_heads, 4)  # 4-head Transformer
    
    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        layer = AGIConsciousnessLayer(
            capacity=5,
            enable_async=False,
            enable_history=False,
            attention_state_dim=64,
            attention_heads=2
        )
        
        self.assertEqual(layer.workspace.capacity, 5)
        self.assertFalse(layer.workspace.enable_async)
        self.assertEqual(layer.attention.state_dim, 64)
        self.assertEqual(layer.attention.num_heads, 2)
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        reset_consciousness_layer()
        
        layer1 = get_consciousness_layer()
        layer2 = get_consciousness_layer()
        
        self.assertIs(layer1, layer2)


class TestTaskRegistration(unittest.TestCase):
    """测试任务注册与意识状态映射"""
    
    def setUp(self):
        """测试前准备"""
        self.layer = AGIConsciousnessLayer()
    
    def test_register_single_task(self):
        """测试注册单个任务"""
        state = self.layer.register_task(
            task_id="task_001",
            task_name="Test Task",
            task_type="reasoning",
            importance=0.8
        )
        
        self.assertEqual(state.level, ConsciousnessLevel.CONSCIOUS)
        self.assertIn("task_001", state.focus)
        self.assertEqual(state.attention_weights["task_001"], 0.8)
        self.assertEqual(self.layer.stats['total_tasks'], 1)
    
    def test_register_multiple_tasks(self):
        """测试注册多个任务"""
        tasks = [
            ("task_001", "High Priority", "reasoning", 0.9),
            ("task_002", "Medium Priority", "planning", 0.6),
            ("task_003", "Low Priority", "monitoring", 0.2)
        ]
        
        for task_id, name, task_type, importance in tasks:
            self.layer.register_task(task_id, name, task_type, importance)
        
        self.assertEqual(self.layer.stats['total_tasks'], 3)
        self.assertEqual(len(self.layer.task_to_consciousness), 3)
    
    def test_importance_to_consciousness_level_mapping(self):
        """测试重要性到意识水平的映射"""
        # 极高优先级 -> METACONSCIOUS
        state1 = self.layer.register_task("task_meta", "Meta", "meta", 0.95)
        self.assertEqual(state1.level, ConsciousnessLevel.METACONSCIOUS)
        
        # 高优先级 -> CONSCIOUS
        state2 = self.layer.register_task("task_high", "High", "high", 0.75)
        self.assertEqual(state2.level, ConsciousnessLevel.CONSCIOUS)
        
        # 中优先级 -> PRECONSCIOUS
        state3 = self.layer.register_task("task_med", "Medium", "medium", 0.5)
        self.assertEqual(state3.level, ConsciousnessLevel.PRECONSCIOUS)
        
        # 低优先级 -> UNCONSCIOUS
        state4 = self.layer.register_task("task_low", "Low", "low", 0.1)
        self.assertEqual(state4.level, ConsciousnessLevel.UNCONSCIOUS)
    
    def test_task_metadata(self):
        """测试任务元数据"""
        metadata = {
            'user_id': 'user_123',
            'priority_boost': True,
            'deadline': '2025-11-25'
        }
        
        state = self.layer.register_task(
            task_id="task_meta",
            task_name="Task with Metadata",
            task_type="test",
            importance=0.7,
            metadata=metadata
        )
        
        self.assertIn('metadata', state.working_memory)
        self.assertEqual(state.working_memory['metadata'], metadata)


class TestAttentionMechanism(unittest.TestCase):
    """测试注意力机制集成"""
    
    def setUp(self):
        """测试前准备"""
        self.layer = AGIConsciousnessLayer()
    
    def test_compute_task_priority_with_states(self):
        """测试使用状态向量计算优先级"""
        # 注册任务
        task_ids = ["task_001", "task_002", "task_003"]
        for i, task_id in enumerate(task_ids):
            self.layer.register_task(
                task_id=task_id,
                task_name=f"Task {i+1}",
                task_type="test",
                importance=0.5 + i * 0.1
            )
        
        # 创建状态向量
        task_states = torch.randn(3, 128)
        
        # 计算优先级
        priorities = self.layer.compute_task_priority(task_ids, task_states)
        
        self.assertEqual(len(priorities), 3)
        for task_id in task_ids:
            self.assertIn(task_id, priorities)
            self.assertGreaterEqual(priorities[task_id], 0.0)
            self.assertLessEqual(priorities[task_id], 1.0)
    
    def test_compute_task_priority_without_states(self):
        """测试不提供状态向量时的优先级计算"""
        # 注册任务
        task_ids = ["task_001", "task_002"]
        self.layer.register_task("task_001", "Task 1", "test", 0.8)
        self.layer.register_task("task_002", "Task 2", "test", 0.3)
        
        # 计算优先级 (自动从importance生成状态向量)
        priorities = self.layer.compute_task_priority(task_ids)
        
        self.assertEqual(len(priorities), 2)
        # 验证高importance任务优先级更高
        self.assertGreater(priorities["task_001"], priorities["task_002"])
    
    def test_attention_computation_performance(self):
        """测试注意力计算性能"""
        # 注册10个任务
        task_ids = [f"task_{i:03d}" for i in range(10)]
        for task_id in task_ids:
            self.layer.register_task(task_id, "Task", "test", 0.5)
        
        # 计算优先级
        priorities = self.layer.compute_task_priority(task_ids)
        
        # 验证性能指标
        self.assertEqual(self.layer.stats['attention_computations'], 1)
        self.assertGreater(self.layer.stats['avg_attention_time_ms'], 0.0)
        self.assertLess(self.layer.stats['avg_attention_time_ms'], 100.0)  # <100ms


class TestConsciousnessStateManagement(unittest.TestCase):
    """测试意识状态管理"""
    
    def setUp(self):
        """测试前准备"""
        self.layer = AGIConsciousnessLayer()
    
    def test_update_task_importance(self):
        """测试更新任务重要性"""
        # 注册任务
        self.layer.register_task("task_001", "Task", "test", 0.4)
        
        # 更新重要性
        self.layer.update_task_importance("task_001", 0.9)
        
        # 验证状态变化
        state = self.layer.task_to_consciousness["task_001"]
        self.assertEqual(state.attention_weights["task_001"], 0.9)
        self.assertEqual(state.level, ConsciousnessLevel.METACONSCIOUS)
    
    def test_consciousness_level_transition(self):
        """测试意识水平转换"""
        # 注册低优先级任务
        self.layer.register_task("task_001", "Task", "test", 0.2)
        old_level = self.layer.task_to_consciousness["task_001"].level
        
        # 提升重要性,触发转换
        self.layer.update_task_importance("task_001", 0.85)
        new_level = self.layer.task_to_consciousness["task_001"].level
        
        self.assertEqual(old_level, ConsciousnessLevel.UNCONSCIOUS)
        self.assertEqual(new_level, ConsciousnessLevel.CONSCIOUS)
        self.assertEqual(self.layer.stats['state_transitions'], 1)
    
    def test_remove_task(self):
        """测试移除任务"""
        self.layer.register_task("task_001", "Task", "test", 0.5)
        self.assertIn("task_001", self.layer.task_to_consciousness)
        
        self.layer.remove_task("task_001")
        self.assertNotIn("task_001", self.layer.task_to_consciousness)
    
    def test_get_current_focus(self):
        """测试获取当前焦点"""
        # 注册多个任务
        for i in range(5):
            self.layer.register_task(f"task_{i:03d}", "Task", "test", 0.5)
        
        focus = self.layer.get_current_focus()
        self.assertGreater(len(focus), 0)
        self.assertLessEqual(len(focus), 7)  # 不超过容量


class TestStatisticsAndSummary(unittest.TestCase):
    """测试统计与摘要功能"""
    
    def setUp(self):
        """测试前准备"""
        self.layer = AGIConsciousnessLayer()
    
    def test_statistics_tracking(self):
        """测试统计追踪"""
        # 执行一系列操作
        self.layer.register_task("task_001", "Task 1", "test", 0.7)
        self.layer.register_task("task_002", "Task 2", "test", 0.5)
        self.layer.compute_task_priority(["task_001", "task_002"])
        self.layer.update_task_importance("task_001", 0.9)
        
        # 验证统计
        self.assertEqual(self.layer.stats['total_tasks'], 2)
        self.assertGreater(self.layer.stats['consciousness_updates'], 0)
        self.assertEqual(self.layer.stats['attention_computations'], 1)
        self.assertGreater(self.layer.stats['state_transitions'], 0)
    
    def test_get_consciousness_summary(self):
        """测试获取意识状态摘要"""
        # 注册任务
        self.layer.register_task("task_001", "Task", "test", 0.8)
        
        summary = self.layer.get_consciousness_summary()
        
        self.assertIn('workspace', summary)
        self.assertIn('attention', summary)
        self.assertIn('statistics', summary)
        self.assertIn('active_tasks', summary)
        self.assertIn('timestamp', summary)
        
        self.assertEqual(summary['active_tasks'], 1)
        self.assertEqual(summary['attention']['num_heads'], 4)
        self.assertEqual(summary['attention']['state_dim'], 128)


class TestCapacityLimits(unittest.TestCase):
    """测试容量限制"""
    
    def setUp(self):
        """测试前准备"""
        self.layer = AGIConsciousnessLayer(capacity=7)
    
    def test_7_plus_minus_2_limit(self):
        """测试7±2容量限制"""
        # 注册10个任务 (超过容量)
        for i in range(10):
            self.layer.register_task(
                task_id=f"task_{i:03d}",
                task_name=f"Task {i}",
                task_type="test",
                importance=0.5 + i * 0.05
            )
        
        # 获取焦点
        focus = self.layer.get_current_focus()
        
        # 验证焦点数量不超过容量
        self.assertLessEqual(len(focus), 9)  # 7+2
        self.assertGreaterEqual(len(focus), 5)  # 7-2


class TestPerformance(unittest.TestCase):
    """测试性能指标"""
    
    def setUp(self):
        """测试前准备"""
        self.layer = AGIConsciousnessLayer()
    
    def test_task_registration_performance(self):
        """测试任务注册性能"""
        import time
        
        start = time.time()
        for i in range(100):
            self.layer.register_task(f"task_{i:03d}", "Task", "test", 0.5)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        self.assertLess(avg_time_ms, 10.0)  # 平均<10ms/任务
    
    def test_attention_computation_batch_performance(self):
        """测试批量注意力计算性能"""
        import time
        
        # 注册20个任务
        task_ids = [f"task_{i:03d}" for i in range(20)]
        for task_id in task_ids:
            self.layer.register_task(task_id, "Task", "test", 0.5)
        
        # 批量计算
        start = time.time()
        priorities = self.layer.compute_task_priority(task_ids)
        elapsed = (time.time() - start) * 1000
        
        # 验证性能 (<50ms for 20 tasks)
        self.assertLess(elapsed, 50.0)
        self.assertEqual(len(priorities), 20)


if __name__ == '__main__':
    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAGIConsciousnessLayerInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionMechanism))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessStateManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticsAndSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestCapacityLimits))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("Phase 3.2 Stage 1 集成测试完成")
    print("=" * 70)
    print(f"✅ 测试通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 测试失败: {len(result.failures)}")
    print(f"💥 测试错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过! Stage 1集成验证成功!")
    else:
        print("\n⚠️  部分测试失败,请检查错误信息")
        sys.exit(1)
