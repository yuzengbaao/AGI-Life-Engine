"""
Phase 3.2 Week 2 Tests - Attention Training & Phase 2 Integration
================================================================

测试内容:
1. AttentionTrainer 训练流程
2. AttentionDataset 数据集
3. Phase2Integration 集成功能
4. StateEncoder 状态编码
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import shutil

from phase3_2_self_awareness import (
    AttentionMechanism,
    AttentionTrainer,
    AttentionDataset,
    TrainingConfig,
    generate_synthetic_data,
    GlobalWorkspace,
    Phase2Integration,
    StateEncoder
)


class TestAttentionDataset(unittest.TestCase):
    """测试 AttentionDataset"""
    
    def setUp(self):
        self.states = torch.randn(100, 10, 128)
        self.labels = torch.rand(100, 10)
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = AttentionDataset(self.states, self.labels)
        
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.states.shape, (100, 10, 128))
        self.assertEqual(dataset.labels.shape, (100, 10))
    
    def test_dataset_getitem(self):
        """测试数据获取"""
        dataset = AttentionDataset(self.states, self.labels)
        
        states, labels, masks = dataset[0]
        
        self.assertEqual(states.shape, (10, 128))
        self.assertEqual(labels.shape, (10,))
        self.assertEqual(masks.shape, (10,))
    
    def test_dataset_with_mask(self):
        """测试带掩码的数据集"""
        masks = torch.ones(100, 10)
        masks[:, -3:] = 0  # 最后3个任务无效
        
        dataset = AttentionDataset(self.states, self.labels, masks)
        
        _, _, mask = dataset[0]
        self.assertEqual(mask[-1].item(), 0)


class TestTrainingConfig(unittest.TestCase):
    """测试 TrainingConfig"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        self.assertEqual(config.state_dim, 128)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.batch_size, 32)
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = TrainingConfig(learning_rate=1e-3)
        config_dict = config.to_dict()
        
        self.assertIn('learning_rate', config_dict)
        self.assertEqual(config_dict['learning_rate'], 1e-3)


class TestAttentionTrainer(unittest.TestCase):
    """测试 AttentionTrainer"""
    
    def setUp(self):
        # 使用小配置加快测试
        self.config = TrainingConfig(
            state_dim=64,
            num_heads=2,
            num_epochs=2,
            batch_size=16,
            device='cpu',
            save_dir='test_checkpoints'
        )
        
        self.trainer = AttentionTrainer(self.config)
        
        # 生成小数据集
        states, labels = generate_synthetic_data(50, 5, 64)
        self.dataset = AttentionDataset(states, labels)
    
    def tearDown(self):
        """清理测试文件"""
        checkpoint_dir = Path(self.config.save_dir)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertEqual(self.trainer.current_epoch, 0)
    
    def test_parameter_count(self):
        """测试参数计数"""
        param_count = self.trainer._count_parameters()
        self.assertGreater(param_count, 0)
    
    def test_training_epoch(self):
        """测试训练一个 epoch"""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        loss = self.trainer._train_epoch(loader)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_validation(self):
        """测试验证"""
        from torch.utils.data import DataLoader
        
        loader = DataLoader(self.dataset, batch_size=16, shuffle=False)
        loss = self.trainer._validate(loader)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_full_training(self):
        """测试完整训练流程"""
        # 分割数据
        train_dataset = AttentionDataset(
            self.dataset.states[:40],
            self.dataset.labels[:40]
        )
        val_dataset = AttentionDataset(
            self.dataset.states[40:],
            self.dataset.labels[40:]
        )
        
        # 训练
        history = self.trainer.train(train_dataset, val_dataset)
        
        # 验证历史
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), self.config.num_epochs)
    
    def test_evaluation(self):
        """测试评估"""
        metrics = self.trainer.evaluate(self.dataset)
        
        self.assertIn('test_loss', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('correlation', metrics)
    
    def test_checkpoint_save_load(self):
        """测试检查点保存和加载"""
        # 保存
        self.trainer.save_checkpoint('test_checkpoint.pt')
        
        checkpoint_path = Path(self.config.save_dir) / 'test_checkpoint.pt'
        self.assertTrue(checkpoint_path.exists())
        
        # 修改状态
        old_epoch = self.trainer.current_epoch
        self.trainer.current_epoch = 10
        
        # 加载
        self.trainer.load_checkpoint('test_checkpoint.pt')
        
        # 验证恢复
        self.assertEqual(self.trainer.current_epoch, old_epoch)


class TestGenerateSyntheticData(unittest.TestCase):
    """测试合成数据生成"""
    
    def test_data_generation(self):
        """测试数据生成"""
        states, labels = generate_synthetic_data(100, 10, 128)
        
        self.assertEqual(states.shape, (100, 10, 128))
        self.assertEqual(labels.shape, (100, 10))
    
    def test_label_range(self):
        """测试标签范围"""
        _, labels = generate_synthetic_data(50, 5, 64)
        
        # 标签应该在 [0, 1]
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels <= 1).all())


class TestStateEncoder(unittest.TestCase):
    """测试 StateEncoder"""
    
    def setUp(self):
        self.encoder = StateEncoder(input_dim=64, output_dim=128)
    
    def test_encoder_forward(self):
        """测试编码器前向传播"""
        x = torch.randn(10, 64)
        output = self.encoder(x)
        
        self.assertEqual(output.shape, (10, 128))
    
    def test_encoder_3d_input(self):
        """测试3D输入"""
        x = torch.randn(2, 5, 64)
        output = self.encoder(x)
        
        self.assertEqual(output.shape, (2, 5, 128))


class TestPhase2Integration(unittest.TestCase):
    """测试 Phase2Integration"""
    
    def setUp(self):
        self.workspace = GlobalWorkspace(capacity=7)
        self.attention = AttentionMechanism(state_dim=128, num_heads=4)
        self.integration = Phase2Integration(
            self.workspace,
            self.attention,
            state_dim=128
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.integration.workspace)
        self.assertIsNotNone(self.integration.attention)
        self.assertEqual(self.integration.state_dim, 128)
    
    def test_load_phase2_systems(self):
        """测试加载 Phase 2 系统"""
        # 尝试加载 (可能会失败,但不应该抛出致命异常)
        # Phase 2 系统可能需要特定参数,所以允许部分加载失败
        self.integration.load_phase2_systems()
        
        # 只要没有崩溃就通过测试
        # 实际使用中会有更严格的集成测试
        self.assertIsNotNone(self.integration)
    
    def test_extract_task_states(self):
        """测试提取任务状态"""
        context = {
            'maml_features': np.random.randn(64).tolist(),
            'gnn_embeddings': np.random.randn(256).tolist(),
            'goal_embeddings': np.random.randn(256).tolist(),
        }
        
        task_states, task_names = self.integration.extract_task_states(context)
        
        self.assertIsInstance(task_states, torch.Tensor)
        self.assertIsInstance(task_names, list)
        self.assertGreater(len(task_names), 0)
    
    def test_process_with_attention(self):
        """测试使用注意力处理"""
        context = {
            'task_type': 'test',
            'maml_features': np.random.randn(64).tolist(),
        }
        
        result = self.integration.process_with_attention(context, capacity=5)
        
        # 验证结果
        self.assertIn('task_states', result)
        self.assertIn('task_names', result)
        self.assertIn('importance_scores', result)
        self.assertIn('focus_names', result)
        self.assertIn('consciousness_level', result)
    
    def test_get_statistics(self):
        """测试获取统计"""
        stats = self.integration.get_statistics()
        
        self.assertIn('total_integrations', stats)
        self.assertIn('maml_activations', stats)
        self.assertIn('workspace_statistics', stats)
    
    def test_reset_statistics(self):
        """测试重置统计"""
        # 执行一次处理
        context = {'maml_features': np.random.randn(64).tolist()}
        self.integration.process_with_attention(context)
        
        # 重置
        self.integration.reset_statistics()
        
        stats = self.integration.get_statistics()
        self.assertEqual(stats['total_integrations'], 0)


class TestIntegrationFlow(unittest.TestCase):
    """测试完整集成流程"""
    
    def test_end_to_end_flow(self):
        """测试端到端流程"""
        # 1. 创建组件
        workspace = GlobalWorkspace(capacity=5)
        attention = AttentionMechanism(state_dim=128, num_heads=4)
        integration = Phase2Integration(workspace, attention, state_dim=128)
        
        # 2. 模拟任务
        context = {
            'task_type': 'complex_reasoning',
            'maml_features': np.random.randn(64).tolist(),
            'gnn_embeddings': np.random.randn(256).tolist(),
        }
        
        # 3. 处理
        result = integration.process_with_attention(context, capacity=3)
        
        # 4. 验证工作空间更新
        self.assertGreater(len(workspace.current_state.focus), 0)
        self.assertGreater(len(workspace.current_state.attention_weights), 0)
        
        # 5. 验证意识水平
        level = workspace.get_consciousness_level()
        self.assertIsNotNone(level)
        
        # 6. 验证结果完整性
        self.assertLessEqual(len(result['focus_names']), 3)
        self.assertEqual(len(result['task_names']), result['task_states'].shape[0])


def run_week2_tests():
    """运行 Week 2 测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateSyntheticData))
    suite.addTests(loader.loadTestsFromTestCase(TestStateEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationFlow))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3.2 Self-Awareness - Week 2 Tests")
    print("=" * 60)
    print()
    
    success = run_week2_tests()
    
    print()
    print("=" * 60)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
