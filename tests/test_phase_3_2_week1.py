"""
Phase 3.2 Self-Awareness System Unit Tests
===========================================

测试 Week 1 实现的核心组件:
- ConsciousnessState 数据类
- GlobalWorkspace 意识状态管理
- AttentionMechanism 注意力机制

测试覆盖:
1. 功能性测试 (正确性)
2. 边界条件测试
3. 性能测试
4. 集成测试
"""

import unittest
import time
import torch
from phase3_2_self_awareness.global_workspace import (
    ConsciousnessLevel,
    ConsciousnessState,
    GlobalWorkspace
)
from phase3_2_self_awareness.attention_mechanism import (
    AttentionMechanism,
    SimpleAttentionMechanism
)


class TestConsciousnessState(unittest.TestCase):
    """测试 ConsciousnessState 数据类"""
    
    def setUp(self):
        self.state = ConsciousnessState(
            timestamp=time.time(),
            level=ConsciousnessLevel.CONSCIOUS,
            focus=["task_1", "task_2"],
            attention_weights={"task_1": 0.6, "task_2": 0.4},
            working_memory={"data": "test"},
            metadata={"test": True}
        )
    
    def test_state_creation(self):
        """测试状态创建"""
        self.assertEqual(self.state.level, ConsciousnessLevel.CONSCIOUS)
        self.assertEqual(len(self.state.focus), 2)
        self.assertIn("task_1", self.state.attention_weights)
    
    def test_to_dict(self):
        """测试转换为字典"""
        state_dict = self.state.to_dict()
        self.assertIn("timestamp", state_dict)
        self.assertIn("level", state_dict)
        self.assertIn("focus", state_dict)
        self.assertEqual(state_dict["level_value"], 2)
    
    def test_get_primary_focus(self):
        """测试获取主要焦点"""
        primary = self.state.get_primary_focus()
        self.assertEqual(primary, "task_1")  # 权重最高
    
    def test_get_focus_intensity(self):
        """测试获取焦点强度"""
        intensity = self.state.get_focus_intensity()
        self.assertEqual(intensity, 0.6)  # 最大权重


class TestGlobalWorkspace(unittest.TestCase):
    """测试 GlobalWorkspace 全局工作空间"""
    
    def setUp(self):
        self.workspace = GlobalWorkspace(capacity=7)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.workspace.capacity, 7)
        self.assertEqual(self.workspace.current_state.level, ConsciousnessLevel.PRECONSCIOUS)
        self.assertEqual(len(self.workspace.current_state.focus), 0)
    
    def test_subscription(self):
        """测试订阅机制"""
        received_broadcasts = []
        
        def callback(info):
            received_broadcasts.append(info)
        
        # 订阅
        sub_id = self.workspace.subscribe(callback)
        self.assertIsInstance(sub_id, int)
        
        # 广播
        self.workspace.broadcast({"message": "test"})
        
        # 验证接收
        self.assertEqual(len(received_broadcasts), 1)
        self.assertIn("information", received_broadcasts[0])
    
    def test_unsubscribe(self):
        """测试取消订阅"""
        received = []
        sub_id = self.workspace.subscribe(lambda x: received.append(x))
        
        # 广播 1
        self.workspace.broadcast({"test": 1})
        self.assertEqual(len(received), 1)
        
        # 取消订阅
        self.workspace.unsubscribe(sub_id)
        
        # 广播 2
        self.workspace.broadcast({"test": 2})
        # 应该还是 1 (没收到新广播)
        self.assertEqual(len(received), 1)
    
    def test_update_focus(self):
        """测试更新焦点"""
        new_focus = ["task_1", "task_2", "task_3"]
        self.workspace.update_focus(new_focus, reason="test")
        
        self.assertEqual(self.workspace.current_state.focus, new_focus)
        self.assertEqual(self.workspace.current_state.metadata["last_update_reason"], "test")
    
    def test_focus_capacity_limit(self):
        """测试焦点容量限制"""
        # 创建容量为 3 的工作空间
        ws = GlobalWorkspace(capacity=3)
        
        # 尝试添加 5 个焦点
        new_focus = ["task_1", "task_2", "task_3", "task_4", "task_5"]
        ws.update_focus(new_focus)
        
        # 应该只保留 3 个
        self.assertEqual(len(ws.current_state.focus), 3)
    
    def test_allocate_attention(self):
        """测试注意力分配"""
        priorities = {
            "task_1": 0.5,
            "task_2": 0.3,
            "task_3": 0.2
        }
        
        self.workspace.allocate_attention(priorities)
        
        # 检查归一化
        total_weight = sum(self.workspace.current_state.attention_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
        # 检查最高权重
        max_task = max(
            self.workspace.current_state.attention_weights.items(),
            key=lambda x: x[1]
        )[0]
        self.assertEqual(max_task, "task_1")
    
    def test_consciousness_level_transition(self):
        """测试意识水平转换"""
        # 初始状态: PRECONSCIOUS
        self.assertEqual(self.workspace.current_state.level, ConsciousnessLevel.PRECONSCIOUS)
        
        # 分配高强度注意力 -> CONSCIOUS
        self.workspace.allocate_attention({"important_task": 0.9, "other": 0.1})
        self.workspace.update_focus(["important_task", "other"])
        self.workspace._update_consciousness_level()
        
        self.assertEqual(self.workspace.current_state.level, ConsciousnessLevel.CONSCIOUS)
        
        # 清空焦点 -> UNCONSCIOUS
        self.workspace.update_focus([])
        self.workspace._update_consciousness_level()
        
        self.assertEqual(self.workspace.current_state.level, ConsciousnessLevel.UNCONSCIOUS)
    
    def test_working_memory(self):
        """测试工作记忆"""
        # 更新
        self.workspace.update_working_memory("key1", "value1")
        self.workspace.update_working_memory("key2", 123)
        
        # 获取
        val1 = self.workspace.get_working_memory("key1")
        val2 = self.workspace.get_working_memory("key2")
        
        self.assertEqual(val1, "value1")
        self.assertEqual(val2, 123)
        
        # 默认值
        val3 = self.workspace.get_working_memory("nonexistent", default="default")
        self.assertEqual(val3, "default")
        
        # 清空
        self.workspace.clear_working_memory()
        self.assertEqual(len(self.workspace.current_state.working_memory), 0)
    
    def test_get_state_summary(self):
        """测试状态摘要"""
        self.workspace.update_focus(["task_1", "task_2"])
        self.workspace.allocate_attention({"task_1": 0.7, "task_2": 0.3})
        
        summary = self.workspace.get_state_summary()
        
        self.assertIn("consciousness_level", summary)
        self.assertIn("focus_count", summary)
        self.assertIn("primary_focus", summary)
        self.assertIn("attention_entropy", summary)
        
        self.assertEqual(summary["focus_count"], 2)
        self.assertEqual(summary["primary_focus"], "task_1")
    
    def test_statistics(self):
        """测试统计信息"""
        # 执行一些操作
        self.workspace.subscribe(lambda x: None)
        self.workspace.broadcast({"test": 1})
        self.workspace.update_focus(["task_1"])
        
        stats = self.workspace.get_statistics()
        
        # update_focus 内部也会触发广播,所以是 2 次
        self.assertGreaterEqual(stats["total_broadcasts"], 1)
        self.assertEqual(stats["subscribers_count"], 1)
        self.assertGreater(stats["total_updates"], 0)
    
    def test_reset(self):
        """测试重置"""
        # 执行操作
        self.workspace.update_focus(["task_1"])
        self.workspace.update_working_memory("key", "value")
        self.workspace.broadcast({"test": 1})
        
        # 重置
        self.workspace.reset()
        
        # 验证
        self.assertEqual(len(self.workspace.current_state.focus), 0)
        self.assertEqual(len(self.workspace.current_state.working_memory), 0)
        self.assertEqual(len(self.workspace.broadcast_history), 0)


class TestAttentionMechanism(unittest.TestCase):
    """测试 AttentionMechanism 注意力机制"""
    
    def setUp(self):
        self.model = AttentionMechanism(state_dim=128, num_heads=4)
        self.model.eval()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.model.state_dim, 128)
        self.assertEqual(self.model.num_heads, 4)
        self.assertEqual(self.model.head_dim, 32)
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 2
        num_tasks = 10
        states = torch.randn(batch_size, num_tasks, 128)
        
        with torch.no_grad():
            output = self.model(states)
        
        # 检查输出键
        self.assertIn("attention_weights", output)
        self.assertIn("importance_scores", output)
        self.assertIn("attended_states", output)
        self.assertIn("normalized_importance", output)
        
        # 检查形状
        self.assertEqual(output["attention_weights"].shape, (batch_size, 4, num_tasks, num_tasks))
        self.assertEqual(output["importance_scores"].shape, (batch_size, num_tasks))
        self.assertEqual(output["attended_states"].shape, (batch_size, num_tasks, 128))
    
    def test_attention_weights_valid(self):
        """测试注意力权重有效性"""
        states = torch.randn(1, 5, 128)
        
        with torch.no_grad():
            output = self.model(states)
        
        attention = output["attention_weights"]  # [1, 4, 5, 5]
        
        # 每行应该求和为 1 (softmax)
        row_sums = attention.sum(dim=-1)  # [1, 4, 5]
        
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
    
    def test_importance_scores_range(self):
        """测试重要性分数范围"""
        states = torch.randn(2, 10, 128)
        
        with torch.no_grad():
            output = self.model(states)
        
        scores = output["importance_scores"]
        
        # Sigmoid 输出应该在 [0, 1]
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_select_focus(self):
        """测试焦点选择"""
        importance_scores = torch.tensor([
            [0.9, 0.8, 0.3, 0.2, 0.1],
            [0.5, 0.6, 0.4, 0.7, 0.3]
        ])
        
        focus_indices = self.model.select_focus(importance_scores, capacity=3)
        
        # 应该选择前 3 个
        self.assertEqual(len(focus_indices), 2)  # 2 个 batch
        self.assertEqual(len(focus_indices[0]), 3)
        self.assertEqual(len(focus_indices[1]), 3)
        
        # Batch 0 应该是 [0, 1, 2] (前三高)
        self.assertIn(0, focus_indices[0])
        self.assertIn(1, focus_indices[0])
    
    def test_select_focus_with_threshold(self):
        """测试带阈值的焦点选择"""
        importance_scores = torch.tensor([[0.9, 0.8, 0.3, 0.2, 0.1]])
        
        focus_indices = self.model.select_focus(
            importance_scores,
            capacity=5,
            threshold=0.5
        )
        
        # 只有 0.9 和 0.8 高于阈值
        self.assertEqual(len(focus_indices[0]), 2)
    
    def test_compute_attention_diversity(self):
        """测试注意力多样性计算"""
        # 创建均匀分布的注意力 (高多样性)
        uniform_attention = torch.ones(1, 4, 5, 5) / 5
        
        # 创建集中的注意力 (低多样性)
        concentrated_attention = torch.zeros(1, 4, 5, 5)
        concentrated_attention[:, :, :, 0] = 1.0
        
        diversity_uniform = self.model.compute_attention_diversity(uniform_attention)
        diversity_concentrated = self.model.compute_attention_diversity(concentrated_attention)
        
        # 均匀分布应该有更高的多样性
        self.assertGreater(diversity_uniform.item(), diversity_concentrated.item())
    
    def test_get_attention_summary(self):
        """测试注意力摘要"""
        states = torch.randn(2, 5, 128)
        
        with torch.no_grad():
            output = self.model(states)
        
        summary = self.model.get_attention_summary(
            output["attention_weights"],
            output["importance_scores"],
            task_names=["task_1", "task_2", "task_3", "task_4", "task_5"]
        )
        
        self.assertIn("batch_size", summary)
        self.assertIn("num_heads", summary)
        self.assertIn("num_tasks", summary)
        self.assertIn("top_tasks", summary)
        self.assertIn("top_task_names", summary)
        
        self.assertEqual(summary["batch_size"], 2)
        self.assertEqual(summary["num_heads"], 4)
        self.assertEqual(len(summary["top_task_names"]), 2)
    
    def test_mask_support(self):
        """测试掩码支持"""
        states = torch.randn(1, 5, 128)
        mask = torch.tensor([[1, 1, 1, 0, 0]])  # 只考虑前 3 个任务
        
        with torch.no_grad():
            output = self.model(states, mask=mask)
        
        importance = output["normalized_importance"]
        
        # 后两个任务的重要性应该为 0 (或极小)
        self.assertLess(importance[0, 3].item(), 1e-5)
        self.assertLess(importance[0, 4].item(), 1e-5)


class TestSimpleAttentionMechanism(unittest.TestCase):
    """测试 SimpleAttentionMechanism 简化版"""
    
    def setUp(self):
        self.mechanism = SimpleAttentionMechanism(capacity=7)
    
    def test_compute_importance(self):
        """测试重要性计算"""
        task_features = {
            "task_1": {"urgency": 0.9, "difficulty": 0.5, "value": 0.8},
            "task_2": {"urgency": 0.3, "difficulty": 0.7, "value": 0.4},
            "task_3": {"urgency": 0.6, "difficulty": 0.6, "value": 0.6}
        }
        
        scores = self.mechanism.compute_importance(task_features)
        
        self.assertIn("task_1", scores)
        self.assertIn("task_2", scores)
        self.assertIn("task_3", scores)
        
        # task_1 应该有最高分 (urgency 最高)
        self.assertGreater(scores["task_1"], scores["task_2"])
    
    def test_select_focus(self):
        """测试焦点选择"""
        importance_scores = {
            "task_1": 0.9,
            "task_2": 0.8,
            "task_3": 0.5,
            "task_4": 0.2,
            "task_5": 0.1
        }
        
        selected = self.mechanism.select_focus(importance_scores, threshold=0.4)
        
        # 应该选择 task_1, task_2, task_3
        self.assertEqual(len(selected), 3)
        self.assertIn("task_1", selected)
        self.assertIn("task_2", selected)
        self.assertIn("task_3", selected)
    
    def test_capacity_limit(self):
        """测试容量限制"""
        mechanism = SimpleAttentionMechanism(capacity=3)
        
        importance_scores = {f"task_{i}": 0.5 for i in range(10)}
        
        selected = mechanism.select_focus(importance_scores, threshold=0.0)
        
        # 应该只选择 3 个
        self.assertEqual(len(selected), 3)


class TestIntegration(unittest.TestCase):
    """测试组件集成"""
    
    def test_workspace_with_attention_mechanism(self):
        """测试工作空间与注意力机制集成"""
        workspace = GlobalWorkspace(capacity=7)
        attention = AttentionMechanism(state_dim=128, num_heads=4)
        attention.eval()
        
        # 模拟任务状态
        num_tasks = 10
        task_states = torch.randn(1, num_tasks, 128)
        
        # 使用注意力机制计算重要性
        with torch.no_grad():
            output = attention(task_states)
        
        importance_scores = output["importance_scores"][0]  # [num_tasks]
        
        # 选择焦点
        focus_indices = attention.select_focus(
            importance_scores.unsqueeze(0),
            capacity=workspace.capacity
        )[0]
        
        # 更新工作空间
        task_names = [f"task_{i}" for i in range(num_tasks)]
        focus_names = [task_names[i] for i in focus_indices]
        
        workspace.update_focus(focus_names, reason="attention_mechanism")
        
        # 分配注意力权重
        attention_weights = {
            task_names[i]: importance_scores[i].item()
            for i in focus_indices
        }
        workspace.allocate_attention(attention_weights)
        
        # 验证
        self.assertEqual(len(workspace.current_state.focus), len(focus_indices))
        self.assertGreater(len(workspace.current_state.attention_weights), 0)
        
        # 获取摘要
        summary = workspace.get_state_summary()
        self.assertIsNotNone(summary["primary_focus"])


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessState))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalWorkspace))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionMechanism))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleAttentionMechanism))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3.2 Self-Awareness - Week 1 Unit Tests")
    print("=" * 60)
    print()
    
    success = run_tests()
    
    print()
    print("=" * 60)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败,请检查日志")
    print("=" * 60)
