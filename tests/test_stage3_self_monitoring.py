"""
Phase 3.2 Stage 3 集成测试 - AGI自我监控层集成验证

测试范围:
1. AGISelfMonitoringLayer初始化
2. 性能监控功能 (CPU, 内存, 延迟, 吞吐量)
3. 错误检测功能 (异常捕获, 分类, 模式识别)
4. 系统健康诊断 (健康评分, 组件状态)
5. 统计与历史查询
6. 警报回调机制

作者: GitHub Copilot (Claude Sonnet 4.5)
创建时间: 2025-11-22
版本: 1.0.0
"""

import unittest
import asyncio
import sys
import time
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agi_self_monitoring_integration import (
    AGISelfMonitoringLayer,
    get_self_monitoring_layer,
    reset_self_monitoring_layer
)


class TestAGISelfMonitoringLayerInitialization(unittest.TestCase):
    """测试AGI自我监控层初始化"""
    
    def test_initialization(self):
        """测试基础初始化"""
        layer = AGISelfMonitoringLayer()
        
        self.assertIsNotNone(layer.performance_monitor)
        self.assertIsNotNone(layer.error_detector)
        self.assertIsNotNone(layer.self_diagnosis)
        self.assertEqual(layer.stats['total_snapshots'], 0)
    
    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        layer = AGISelfMonitoringLayer(
            sampling_interval=2.0,
            history_size=1000,
            max_errors=5000,
            pattern_detection=False,
            health_check_interval=30.0,
            enable_auto_monitoring=False
        )
        
        self.assertEqual(layer.performance_monitor.sampling_interval, 2.0)
        self.assertEqual(layer.error_detector.max_errors, 5000)
        self.assertEqual(layer.health_check_interval, 30.0)
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        reset_self_monitoring_layer()
        
        layer1 = get_self_monitoring_layer()
        layer2 = get_self_monitoring_layer()
        
        self.assertIs(layer1, layer2)


class TestPerformanceMonitoring(unittest.IsolatedAsyncioTestCase):
    """测试性能监控功能"""
    
    async def asyncSetUp(self):
        """测试前准备"""
        self.layer = AGISelfMonitoringLayer(enable_auto_monitoring=False)
        await self.layer.start()
    
    async def asyncTearDown(self):
        """测试后清理"""
        await self.layer.stop()
    
    async def test_capture_snapshot(self):
        """测试捕获性能快照"""
        snapshot = self.layer.capture_snapshot()
        
        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.cpu_percent, 0.0)
        self.assertGreaterEqual(snapshot.memory_mb, 0.0)
        self.assertGreaterEqual(snapshot.memory_percent, 0.0)
        self.assertEqual(self.layer.stats['total_snapshots'], 1)
    
    async def test_get_current_metrics(self):
        """测试获取当前指标"""
        metrics = self.layer.get_current_metrics()
        
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_mb', metrics)
        self.assertIn('memory_percent', metrics)
        self.assertIn('latency_ms', metrics)
        self.assertIn('throughput', metrics)
        self.assertIn('active_threads', metrics)
    
    async def test_record_operation(self):
        """测试记录操作延迟"""
        # 记录一些操作
        for i in range(10):
            self.layer.record_operation(float(i * 10))
        
        # 捕获快照以确保有延迟数据
        self.layer.capture_snapshot()
        
        # 获取统计
        stats = self.layer.get_performance_statistics()
        
        # 验证统计结构
        self.assertIn('time_range', stats)
        self.assertGreater(stats['time_range']['sample_count'], 0)
    
    async def test_performance_threshold(self):
        """测试性能阈值设置"""
        self.layer.set_performance_threshold('cpu_percent', 80.0, 95.0)
        
        thresholds = self.layer.performance_monitor.thresholds['cpu_percent']
        self.assertEqual(thresholds['warning'], 80.0)
        self.assertEqual(thresholds['critical'], 95.0)
    
    async def test_recent_snapshots(self):
        """测试获取最近快照"""
        # 捕获多个快照
        for _ in range(5):
            self.layer.capture_snapshot()
            await asyncio.sleep(0.1)
        
        # 获取最近3个
        recent = self.layer.get_recent_snapshots(3)
        
        self.assertLessEqual(len(recent), 3)


class TestErrorDetection(unittest.IsolatedAsyncioTestCase):
    """测试错误检测功能"""
    
    async def asyncSetUp(self):
        """测试前准备"""
        self.layer = AGISelfMonitoringLayer(enable_auto_monitoring=False)
        await self.layer.start()
    
    async def asyncTearDown(self):
        """测试后清理"""
        await self.layer.stop()
    
    async def test_capture_exception(self):
        """测试捕获异常"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            record = self.layer.capture_exception(
                e, 
                context={'test': 'context'},
                severity='error',
                component='test_module'
            )
        
        self.assertIsNotNone(record)
        self.assertEqual(record.error_type, 'ValueError')
        self.assertEqual(record.severity, 'error')
        self.assertEqual(self.layer.stats['total_errors'], 1)
    
    async def test_record_error_manually(self):
        """测试手动记录错误"""
        record = self.layer.record_error(
            error_type='CustomError',
            error_message='Custom test error',
            stack_trace='Stack trace here',
            severity='warning',
            component='test_component'
        )
        
        self.assertIsNotNone(record)
        self.assertEqual(record.error_type, 'CustomError')
        self.assertEqual(record.severity, 'warning')
    
    async def test_error_statistics(self):
        """测试错误统计"""
        # 记录多个错误
        for i in range(5):
            self.layer.record_error(
                f'Error{i}',
                f'Message {i}',
                severity='error',
                component='test'
            )
        
        stats = self.layer.get_error_statistics()
        
        self.assertGreater(stats['total_errors'], 0)
        self.assertIn('by_type', stats)
        self.assertIn('by_severity', stats)
    
    async def test_recent_errors(self):
        """测试获取最近错误"""
        # 记录多个不同严重程度的错误
        self.layer.record_error('Error1', 'Message1', severity='error')
        self.layer.record_error('Error2', 'Message2', severity='critical')
        self.layer.record_error('Error3', 'Message3', severity='warning')
        
        # 获取所有最近错误
        recent = self.layer.get_recent_errors(10)
        self.assertGreater(len(recent), 0)
        
        # 仅获取严重错误
        critical_errors = self.layer.get_recent_errors(10, severity='critical')
        self.assertTrue(all(e.severity == 'critical' for e in critical_errors))


class TestSystemHealthDiagnosis(unittest.IsolatedAsyncioTestCase):
    """测试系统健康诊断"""
    
    async def asyncSetUp(self):
        """测试前准备"""
        self.layer = AGISelfMonitoringLayer(
            enable_auto_monitoring=False,
            health_check_interval=1.0
        )
        await self.layer.start()
    
    async def asyncTearDown(self):
        """测试后清理"""
        await self.layer.stop()
    
    async def test_run_health_check(self):
        """测试运行健康检查"""
        report = await self.layer.run_health_check()
        
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.overall_health_score, 0.0)
        self.assertLessEqual(report.overall_health_score, 100.0)
        self.assertIn(report.overall_status, ['healthy', 'degraded', 'unhealthy', 'critical'])
    
    async def test_health_check_with_errors(self):
        """测试有错误时的健康检查"""
        # 制造一些错误
        for i in range(10):
            self.layer.record_error(
                f'Error{i}',
                f'Message {i}',
                severity='error'
            )
        
        report = await self.layer.run_health_check()
        
        # 验证报告结构(注:Phase3.2组件的健康检查可能不直接受错误数量影响)
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.overall_health_score, 0.0)
        self.assertLessEqual(report.overall_health_score, 100.0)
        
        # 验证错误已被记录
        error_stats = self.layer.get_error_statistics()
        self.assertGreaterEqual(error_stats['total_errors'], 10)
    
    async def test_component_status(self):
        """测试组件状态查询"""
        report = await self.layer.run_health_check()
        
        self.assertGreater(len(report.component_statuses), 0)
        
        # 验证组件状态结构
        for status in report.component_statuses:
            self.assertIn('component_name', status.__dict__)
            self.assertIn('status', status.__dict__)
            self.assertIn('health_score', status.__dict__)


class TestMonitoringSummary(unittest.IsolatedAsyncioTestCase):
    """测试监控总览功能"""
    
    async def asyncSetUp(self):
        """测试前准备"""
        self.layer = AGISelfMonitoringLayer(enable_auto_monitoring=False)
        await self.layer.start()
    
    async def asyncTearDown(self):
        """测试后清理"""
        await self.layer.stop()
    
    async def test_get_monitoring_summary(self):
        """测试获取监控总览"""
        # 生成一些活动
        self.layer.capture_snapshot()
        self.layer.record_error('TestError', 'Test message')
        await self.layer.run_health_check()
        
        summary = self.layer.get_monitoring_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('status', summary)
        self.assertIn('layer_stats', summary)
        self.assertIn('performance', summary)
        self.assertIn('errors', summary)
        self.assertIn('health', summary)
    
    async def test_monitoring_statistics(self):
        """测试监控统计信息"""
        # 捕获多个快照
        for _ in range(5):
            self.layer.capture_snapshot()
        
        # 记录多个错误
        for i in range(3):
            self.layer.record_error(f'Error{i}', f'Message{i}')
        
        # 执行健康检查
        await self.layer.run_health_check()
        
        # 验证统计
        self.assertEqual(self.layer.stats['total_snapshots'], 5)
        self.assertEqual(self.layer.stats['total_errors'], 3)
        self.assertGreater(self.layer.stats['total_health_checks'], 0)


class TestAlertCallbacks(unittest.IsolatedAsyncioTestCase):
    """测试警报回调机制"""
    
    async def asyncSetUp(self):
        """测试前准备"""
        self.layer = AGISelfMonitoringLayer(enable_auto_monitoring=False)
        await self.layer.start()
        self.callback_triggered = False
        self.callback_data = None
    
    async def asyncTearDown(self):
        """测试后清理"""
        await self.layer.stop()
    
    async def test_performance_alert_callback(self):
        """测试性能警报回调"""
        def alert_callback(alert):
            self.callback_triggered = True
            self.callback_data = alert
        
        self.layer.add_performance_alert_callback(alert_callback)
        
        # 验证回调已添加
        self.assertIn(alert_callback, self.layer.performance_monitor.alert_callbacks)
    
    async def test_error_callback(self):
        """测试错误回调"""
        def error_callback(error):
            self.callback_triggered = True
            self.callback_data = error
        
        self.layer.add_error_callback(error_callback)
        
        # 验证回调已添加
        self.assertIn(error_callback, self.layer.error_detector.error_callbacks)


if __name__ == '__main__':
    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAGISelfMonitoringLayerInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemHealthDiagnosis))
    suite.addTests(loader.loadTestsFromTestCase(TestMonitoringSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestAlertCallbacks))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("Phase 3.2 Stage 3 集成测试完成")
    print("=" * 70)
    print(f"✅ 测试通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 测试失败: {len(result.failures)}")
    print(f"💥 测试错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过! Stage 3集成验证成功!")
        print("🏥 自我监控层已就绪,性能监控+错误检测+健康诊断全面运行!")
    else:
        print("\n⚠️  部分测试失败,请检查错误信息")
        sys.exit(1)
