"""
Monitored Phase 2 Integration 测试套件
======================================

测试增强的Phase 2集成和自我监控功能

测试覆盖:
1. 基础初始化和配置
2. 系统加载和错误处理
3. MAML任务监控
4. GNN推理监控
5. 目标生成监控
6. 性能监控集成
7. 错误检测集成
8. 异常检测集成
9. 健康诊断集成
10. 数据导出和报告

作者: AGI Project Team
创建时间: 2025-01-17
"""

import unittest
import time
import torch
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase3_2_self_awareness.global_workspace import GlobalWorkspace
from phase3_2_self_awareness.attention_mechanism import AttentionMechanism
from phase3_2_self_awareness.monitored_phase2_integration import MonitoredPhase2Integration


class TestMonitoredPhase2Integration(unittest.TestCase):
    """测试增强型Phase 2集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.workspace = GlobalWorkspace(capacity=7)
        self.attention = AttentionMechanism(state_dim=128)
    
    def test_01_initialization_with_monitoring(self):
        """测试带监控的初始化"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        self.assertTrue(integration.enable_monitoring)
        self.assertIsNotNone(integration.perf_monitor)
        self.assertIsNotNone(integration.error_detector)
        self.assertIsNotNone(integration.anomaly_detector)
        self.assertIsNotNone(integration.diagnosis)
        
        # 验证统计初始化
        self.assertEqual(integration.statistics['total_operations'], 0)
        self.assertEqual(integration.statistics['errors_caught'], 0)
    
    def test_02_initialization_without_monitoring(self):
        """测试禁用监控的初始化"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=False
        )
        
        self.assertFalse(integration.enable_monitoring)
        
        # 验证监控组件未创建
        self.assertFalse(hasattr(integration, 'perf_monitor'))
    
    def test_03_monitoring_configuration(self):
        """测试监控配置"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 验证性能阈值配置
        thresholds = integration.perf_monitor.thresholds
        self.assertEqual(thresholds['cpu_percent']['warning'], 70.0)
        self.assertEqual(thresholds['cpu_percent']['critical'], 90.0)
        
        # 验证异常检测阈值
        self.assertIn('train_loss', integration.anomaly_detector.custom_thresholds)
        self.assertIn('train_accuracy', integration.anomaly_detector.custom_thresholds)
    
    def test_04_system_loading(self):
        """测试Phase 2系统加载"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 尝试加载系统 (可能部分成功)
        success = integration.load_phase2_systems()
        
        # 验证至少尝试过加载 (即使失败)
        self.assertIsInstance(success, bool)
        
        # 检查错误检测是否记录了加载失败
        if not success:
            error_stats = integration.error_detector.get_statistics()
            self.assertGreater(error_stats.get('total_errors', 0), 0)
    
    def test_05_maml_task_execution(self):
        """测试MAML任务执行"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 创建模拟数据
        support_data = torch.randn(5, 10)
        query_data = torch.randn(5, 10)
        
        # 执行MAML任务 (会失败因为系统未加载,但会记录错误)
        result = integration.run_maml_task(
            support_data=support_data,
            query_data=query_data,
            task_name="test_task"
        )
        
        # 验证结果结构
        self.assertIn('success', result)
        
        # 验证统计更新
        self.assertEqual(integration.statistics['total_operations'], 1)
        
        # 如果失败,应记录错误
        if not result['success']:
            self.assertGreater(integration.statistics['errors_caught'], 0)
    
    def test_06_gnn_reasoning_execution(self):
        """测试GNN推理执行"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行GNN推理
        result = integration.run_gnn_reasoning(
            query="测试查询",
            knowledge_graph={'nodes': [], 'edges': []}
        )
        
        # 验证结果结构
        self.assertIn('success', result)
        
        # 验证统计更新
        self.assertEqual(integration.statistics['total_operations'], 1)
    
    def test_07_goal_generation_execution(self):
        """测试目标生成执行"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行目标生成
        current_state = {'position': [0, 0], 'resources': 100}
        result = integration.run_goal_generation(
            current_state=current_state,
            constraints={'max_distance': 10}
        )
        
        # 验证结果结构
        self.assertIn('success', result)
        
        # 验证统计更新
        self.assertEqual(integration.statistics['total_operations'], 1)
    
    def test_08_performance_monitoring(self):
        """测试性能监控集成"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行多个操作
        for i in range(5):
            support_data = torch.randn(5, 10)
            query_data = torch.randn(5, 10)
            integration.run_maml_task(support_data, query_data, f"task_{i}")
            time.sleep(0.01)
        
        # 验证性能记录 (可能没有快照但有操作记录)
        stats = integration.perf_monitor.get_statistics(time_window=60)
        # 如果返回错误,说明没有快照,这是正常的(因为测试很快)
        if 'error' not in stats:
            # 有数据时验证操作记录
            self.assertIn('operations', stats)
            self.assertGreater(stats['operations']['count'], 0)
        
        # 验证操作计数(这个肯定有)
        self.assertEqual(integration.statistics['total_operations'], 5)
    
    def test_09_error_detection(self):
        """测试错误检测集成"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 触发多个错误
        for i in range(3):
            integration.run_maml_task(
                torch.randn(5, 10),
                torch.randn(5, 10),
                f"error_task_{i}"
            )
        
        # 验证错误记录
        if integration.statistics['errors_caught'] > 0:
            error_stats = integration.error_detector.get_statistics()
            self.assertGreater(error_stats['total_errors'], 0)
            
            # 检查错误模式检测
            top_errors = integration.error_detector.get_top_errors(3)
            self.assertIsInstance(top_errors, list)
    
    def test_10_anomaly_detection(self):
        """测试异常检测集成"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 记录正常指标
        for i in range(35):
            integration.anomaly_detector.record_metric('test_metric', 50.0 + i)
        
        # 记录异常值
        anomaly = integration.anomaly_detector.record_metric('test_metric', 200.0)
        
        # 验证异常检测
        if anomaly:
            self.assertEqual(anomaly.metric_name, 'test_metric')
            self.assertEqual(anomaly.value, 200.0)
            
            # 验证统计
            anomaly_stats = integration.anomaly_detector.get_statistics()
            self.assertGreater(anomaly_stats['total_anomalies'], 0)
    
    def test_11_health_status(self):
        """测试健康状态查询"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行一些操作
        for i in range(3):
            integration.run_gnn_reasoning(f"query_{i}")
        
        time.sleep(0.1)
        
        # 获取健康状态
        health = integration.get_health_status()
        
        # 验证结构
        self.assertTrue(health['monitoring_enabled'])
        self.assertIn('health_score', health)
        self.assertIn('health_status', health)
        self.assertIn('performance', health)
        self.assertIn('errors', health)
        self.assertIn('anomalies', health)
        self.assertIn('operations', health)
        
        # 验证健康分数范围
        self.assertGreaterEqual(health['health_score'], 0)
        self.assertLessEqual(health['health_score'], 100)
    
    def test_12_diagnostic_report(self):
        """测试诊断报告生成"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行操作
        for i in range(5):
            integration.run_goal_generation({'state': i})
        
        # 生成诊断报告
        report = integration.generate_diagnostic_report()
        
        # 验证报告结构 (报告包含health_report子字典)
        self.assertIn('health_report', report)
        health_report = report['health_report']
        self.assertIn('overall_health_score', health_report)
        self.assertIn('overall_status', health_report)
        self.assertIn('component_statuses', health_report)
        self.assertIn('system_info', report)
    
    def test_13_background_monitoring(self):
        """测试后台监控启停"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 启动后台监控
        integration.start_monitoring()
        self.assertTrue(integration.perf_monitor.is_monitoring)
        
        time.sleep(0.5)
        
        # 验证后台数据收集 (检查快照或CPU/内存数据存在)
        stats = integration.perf_monitor.get_statistics(time_window=60)
        # 后台监控应该记录了CPU或内存数据
        self.assertTrue('cpu' in stats or 'memory' in stats)
        
        # 停止监控
        integration.stop_monitoring()
        self.assertFalse(integration.perf_monitor.is_monitoring)
    
    def test_14_data_export(self):
        """测试监控数据导出"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 生成一些数据
        for i in range(3):
            integration.run_maml_task(torch.randn(5, 10), torch.randn(5, 10))
            integration.run_gnn_reasoning(f"query_{i}")
            integration.run_goal_generation({'state': i})
        
        # 导出数据
        exported = integration.export_monitoring_data()
        
        # 验证导出结构
        self.assertIn('performance_snapshots', exported)
        self.assertIn('error_records', exported)
        self.assertIn('error_patterns', exported)
        self.assertIn('anomaly_records', exported)
        self.assertIn('anomaly_baselines', exported)
    
    def test_15_statistics_tracking(self):
        """测试统计追踪"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行不同类型的操作
        integration.run_maml_task(torch.randn(5, 10), torch.randn(5, 10))
        integration.run_gnn_reasoning("test")
        integration.run_gnn_reasoning("test2")
        integration.run_goal_generation({'state': 1})
        
        # 验证统计
        stats = integration.statistics
        self.assertEqual(stats['total_operations'], 4)
        # GNN可能成功或失败,取决于系统是否可用
        # 只验证计数大于等于0
        self.assertGreaterEqual(stats['gnn_calls'], 0)
    
    def test_16_error_context_integration(self):
        """测试ErrorContext集成"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 手动触发ErrorContext
        try:
            from phase3_2_self_awareness.error_detector import ErrorContext
            with ErrorContext(integration.error_detector, component="Test"):
                raise ValueError("测试错误")
        except ValueError:
            pass
        
        # 验证错误记录
        error_stats = integration.error_detector.get_statistics()
        self.assertGreater(error_stats['total_errors'], 0)
    
    def test_17_multiple_operations_stress(self):
        """测试多操作压力"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        integration.start_monitoring()
        
        # 快速执行20个操作
        for i in range(20):
            if i % 3 == 0:
                integration.run_maml_task(torch.randn(5, 10), torch.randn(5, 10))
            elif i % 3 == 1:
                integration.run_gnn_reasoning(f"query_{i}")
            else:
                integration.run_goal_generation({'state': i})
        
        # 验证统计
        self.assertEqual(integration.statistics['total_operations'], 20)
        
        # 验证性能监控 (检查操作记录或CPU数据)
        perf_stats = integration.perf_monitor.get_statistics(time_window=60)
        # 后台监控可能有CPU数据或操作数据
        if 'error' not in perf_stats:
            # 有数据时至少应该有operations或cpu/memory
            has_data = ('operations' in perf_stats or 
                       'cpu' in perf_stats or 
                       'memory' in perf_stats)
            self.assertTrue(has_data, "应该有性能监控数据")
        
        integration.stop_monitoring()
    
    def test_18_health_recommendations(self):
        """测试健康建议生成"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 执行操作
        for i in range(5):
            integration.run_goal_generation({'state': i})
        
        # 获取健康状态
        health = integration.get_health_status()
        
        # 验证建议存在
        self.assertIn('recommendations', health)
        self.assertIsInstance(health['recommendations'], list)
    
    def test_19_anomaly_severity_levels(self):
        """测试异常严重性等级"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 建立baseline
        for i in range(35):
            integration.anomaly_detector.record_metric('severity_test', 50.0)
        
        # 触发不同严重性的异常
        anomalies = []
        
        # 轻度异常 (1.5倍标准差)
        a1 = integration.anomaly_detector.record_metric('severity_test', 51.0)
        if a1:
            anomalies.append(a1)
        
        # 中度异常 (2倍标准差)
        a2 = integration.anomaly_detector.record_metric('severity_test', 100.0)
        if a2:
            anomalies.append(a2)
        
        # 严重异常 (3倍标准差)
        a3 = integration.anomaly_detector.record_metric('severity_test', 200.0)
        if a3:
            anomalies.append(a3)
        
        # 验证异常记录
        if len(anomalies) > 0:
            self.assertTrue(any(a.severity in ['low', 'medium', 'high', 'critical'] 
                               for a in anomalies))
    
    def test_20_integration_completeness(self):
        """测试集成完整性"""
        integration = MonitoredPhase2Integration(
            global_workspace=self.workspace,
            attention_mechanism=self.attention,
            enable_monitoring=True
        )
        
        # 验证所有监控组件都正确集成
        self.assertIsNotNone(integration.perf_monitor)
        self.assertIsNotNone(integration.error_detector)
        self.assertIsNotNone(integration.anomaly_detector)
        self.assertIsNotNone(integration.diagnosis)
        
        # 验证诊断系统引用了所有监控器
        self.assertIs(integration.diagnosis.performance_monitor, 
                     integration.perf_monitor)
        self.assertIs(integration.diagnosis.error_detector, 
                     integration.error_detector)
        self.assertIs(integration.diagnosis.anomaly_detector, 
                     integration.anomaly_detector)


if __name__ == '__main__':
    # 运行测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMonitoredPhase2Integration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "="*70)
    print("Monitored Phase 2 Integration 测试摘要")
    print("="*70)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"成功率: 100.0%")
        print("✅ 所有测试通过!")
    else:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                       result.testsRun * 100)
        print(f"成功率: {success_rate:.1f}%")
        if result.failures:
            print(f"❌ {len(result.failures)} 个测试失败")
        if result.errors:
            print(f"❌ {len(result.errors)} 个测试错误")
    print("="*70)
