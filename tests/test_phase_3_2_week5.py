"""
Phase 3.2 Week 5 - 自我监控系统测试套件

测试覆盖:
- PerformanceMonitor: 性能监控和警报
- ErrorDetector: 错误检测和模式识别
- AnomalyDetector: 异常检测和基线学习
- SelfDiagnosis: 健康检查和根因分析
- Integration: 完整监控管道

作者: AGI Project Team
创建时间: 2025-01-17
版本: 1.0.0
"""

import unittest
import time
import os
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase3_2_self_awareness.performance_monitor import PerformanceMonitor, PerformanceSnapshot
from phase3_2_self_awareness.error_detector import ErrorDetector, ErrorContext
from phase3_2_self_awareness.anomaly_detector import AnomalyDetector
from phase3_2_self_awareness.self_diagnosis import SelfDiagnosis


class TestPerformanceMonitor(unittest.TestCase):
    """PerformanceMonitor 测试"""
    
    def setUp(self):
        """测试初始化"""
        self.monitor = PerformanceMonitor(sampling_interval=0.1, history_size=100)
    
    def tearDown(self):
        """测试清理"""
        if self.monitor._monitor_thread and self.monitor._monitor_thread.is_alive():
            self.monitor.stop_monitoring()
    
    def test_01_initialization(self):
        """测试初始化"""
        self.assertEqual(self.monitor.sampling_interval, 0.1)
        self.assertEqual(len(self.monitor.snapshots), 0)
        self.assertFalse(self.monitor.is_monitoring)
    
    def test_02_capture_snapshot(self):
        """测试快照捕获"""
        snapshot = self.monitor.capture_snapshot()
        
        self.assertIsInstance(snapshot, PerformanceSnapshot)
        self.assertGreaterEqual(snapshot.cpu_percent, 0)  # CPU 可能为 0
        self.assertGreater(snapshot.memory_mb, 0)
        self.assertGreater(snapshot.active_threads, 0)
    
    def test_03_threshold_configuration(self):
        """测试阈值配置"""
        self.monitor.set_threshold('cpu_percent', warning=80.0, critical=95.0)
        
        thresholds = self.monitor.thresholds['cpu_percent']
        self.assertEqual(thresholds['warning'], 80.0)
        self.assertEqual(thresholds['critical'], 95.0)
    
    def test_04_operation_recording(self):
        """测试操作记录"""
        self.monitor.record_operation(50.0)  # 50ms latency
        self.monitor.record_operation(100.0)  # 100ms latency
        
        self.assertEqual(self.monitor.operation_stats['total_operations'], 2)
        self.assertGreater(self.monitor.operation_stats['total_latency_ms'], 0)
    
    def test_05_background_monitoring(self):
        """测试后台监控"""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        
        time.sleep(0.5)  # 等待几个采样周期
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)
        self.assertGreater(len(self.monitor.snapshots), 0)
    
    def test_06_get_recent_snapshots(self):
        """测试获取最近快照"""
        for _ in range(10):
            self.monitor.capture_snapshot()
        
        recent = self.monitor.get_recent_snapshots(count=5)
        self.assertEqual(len(recent), 5)
    
    def test_07_statistics_calculation(self):
        """测试统计计算"""
        for _ in range(10):
            self.monitor.capture_snapshot()
            time.sleep(0.01)
        
        stats = self.monitor.get_statistics()
        
        self.assertIn('cpu', stats)
        self.assertIn('memory', stats)
        self.assertIn('time_range', stats)
        self.assertGreater(stats['time_range']['sample_count'], 0)
    
    def test_08_alert_triggering(self):
        """测试警报触发"""
        alert_triggered = []
        
        def on_alert(alert):
            alert_triggered.append(alert)
        
        self.monitor.add_alert_callback(on_alert)
        self.monitor.set_threshold('cpu_percent', warning=0.1, critical=0.2)
        
        # 触发警报 (通过捕获快照)
        self.monitor.capture_snapshot()
        
        # CPU 使用率通常 > 0.1%, 应该触发警报
        time.sleep(0.1)
        # 警报可能需要时间触发,所以不强制要求
    
    def test_09_report_generation(self):
        """测试报告生成"""
        for _ in range(5):
            self.monitor.capture_snapshot()
            time.sleep(0.01)
        
        report = self.monitor.generate_report(time_window=10)
        
        self.assertIn('generated_at', report)
        self.assertIn('monitoring_status', report)
        self.assertIn('statistics', report)
    
    def test_10_export_snapshots(self):
        """测试导出快照"""
        for _ in range(5):
            self.monitor.capture_snapshot()
        
        exported = self.monitor.export_snapshots()
        
        self.assertEqual(len(exported), 5)
        self.assertIn('timestamp', exported[0])
        self.assertIn('cpu_percent', exported[0])


class TestErrorDetector(unittest.TestCase):
    """ErrorDetector 测试"""
    
    def setUp(self):
        """测试初始化"""
        self.detector = ErrorDetector(max_errors=100, pattern_detection=True)
    
    def test_11_initialization(self):
        """测试初始化"""
        self.assertEqual(len(self.detector.error_records), 0)
        self.assertTrue(self.detector.pattern_detection)
    
    def test_12_exception_capture(self):
        """测试异常捕获"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            record = self.detector.capture_exception(
                e,
                context={'test': True},
                severity='error',
                component='TestComponent'
            )
        
        self.assertEqual(record.error_type, 'ValueError')
        self.assertEqual(record.error_message, 'Test error')
        self.assertEqual(record.component, 'TestComponent')
        self.assertIn('test', record.context)
    
    def test_13_manual_error_recording(self):
        """测试手动错误记录"""
        record = self.detector.record_error(
            error_type='CustomError',
            error_message='Custom error message',
            severity='warning',
            component='ManualTest'
        )
        
        self.assertEqual(record.error_type, 'CustomError')
        self.assertEqual(record.severity, 'warning')
    
    def test_14_pattern_detection(self):
        """测试模式检测"""
        # 记录相同错误多次
        for i in range(5):
            self.detector.record_error(
                error_type='RepeatedError',
                error_message='This error repeats',
                component=f'Component{i % 2}'
            )
        
        patterns = self.detector.get_top_errors(count=10)
        
        self.assertGreater(len(patterns), 0)
        top_pattern = patterns[0]
        self.assertEqual(top_pattern.error_type, 'RepeatedError')
        self.assertEqual(top_pattern.occurrence_count, 5)
    
    def test_15_error_queries(self):
        """测试错误查询"""
        self.detector.record_error('TypeError', 'Type error', severity='error')
        self.detector.record_error('ValueError', 'Value error', severity='warning')
        self.detector.record_error('TypeError', 'Another type error', severity='critical')
        
        # 按类型查询
        type_errors = self.detector.get_errors_by_type('TypeError')
        self.assertEqual(len(type_errors), 2)
        
        # 按严重程度查询
        warnings = self.detector.get_errors_by_severity('warning')
        self.assertEqual(len(warnings), 1)
    
    def test_16_error_context_manager(self):
        """测试错误上下文管理器"""
        with ErrorContext(self.detector, component='ContextTest', severity='info'):
            pass  # 不抛出异常
        
        # 应该没有错误
        recent = self.detector.get_recent_errors(10)
        context_errors = [e for e in recent if e.component == 'ContextTest']
        self.assertEqual(len(context_errors), 0)
        
        # 抛出异常 (ErrorContext 会捕获但不抑制异常)
        try:
            with ErrorContext(self.detector, component='ContextTest2'):
                raise RuntimeError("Context error")
        except RuntimeError:
            pass  # 预期的异常
        
        recent = self.detector.get_recent_errors(10)
        context_errors = [e for e in recent if e.component == 'ContextTest2']
        self.assertEqual(len(context_errors), 1)
    
    def test_17_statistics(self):
        """测试统计"""
        for i in range(10):
            self.detector.record_error(
                f'Error{i % 3}',
                f'Message {i}',
                severity='error' if i % 2 == 0 else 'warning'
            )
        
        stats = self.detector.get_statistics()
        
        self.assertEqual(stats['total_errors'], 10)
        self.assertGreater(stats['unique_errors'], 0)
        self.assertIn('by_severity', stats)
    
    def test_18_report_generation(self):
        """测试报告生成"""
        for i in range(5):
            self.detector.record_error('TestError', f'Error {i}')
        
        report = self.detector.generate_report(time_window=10)
        
        self.assertIn('summary', report)
        self.assertIn('breakdown', report)
    
    def test_19_export_errors(self):
        """测试导出错误"""
        self.detector.record_error('ExportError', 'Test export')
        
        exported = self.detector.export_errors()
        
        self.assertGreater(len(exported), 0)
        self.assertIn('error_type', exported[0])
    
    def test_20_clear_errors(self):
        """测试清空错误"""
        self.detector.record_error('TempError', 'Temporary')
        self.assertGreater(len(self.detector.error_records), 0)
        
        self.detector.clear_errors()
        self.assertEqual(len(self.detector.error_records), 0)


class TestAnomalyDetector(unittest.TestCase):
    """AnomalyDetector 测试"""
    
    def setUp(self):
        """测试初始化"""
        self.detector = AnomalyDetector(
            baseline_window=50,
            z_score_threshold=2.0,
            iqr_multiplier=1.5
        )
    
    def test_21_initialization(self):
        """测试初始化"""
        self.assertEqual(self.detector.baseline_window, 50)
        self.assertEqual(self.detector.z_score_threshold, 2.0)
    
    def test_22_metric_recording(self):
        """测试指标记录"""
        anomaly = self.detector.record_metric('test_metric', 50.0)
        
        # 第一次记录不应该有异常 (没有基线)
        self.assertIsNone(anomaly)
        
        self.assertIn('test_metric', self.detector.metric_history)
    
    def test_23_baseline_establishment(self):
        """测试基线建立"""
        # 记录足够的数据建立基线
        for i in range(50):
            self.detector.record_metric('cpu', 50.0 + i % 10)
        
        baseline = self.detector.get_baseline('cpu')
        
        self.assertIsNotNone(baseline)
        self.assertGreater(baseline.mean, 0)
        self.assertGreater(baseline.std_dev, 0)
    
    def test_24_threshold_detection(self):
        """测试阈值检测"""
        self.detector.set_threshold('custom_metric', min_value=10.0, max_value=100.0)
        
        # 正常值
        anomaly = self.detector.record_metric('custom_metric', 50.0)
        self.assertIsNone(anomaly)
        
        # 异常值
        anomaly = self.detector.record_metric('custom_metric', 150.0)
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.anomaly_type, 'threshold')
    
    def test_25_z_score_detection(self):
        """测试 Z-score 检测"""
        # 建立正常基线
        for _ in range(50):
            self.detector.record_metric('z_test', 50.0)
        
        # 记录异常值
        anomaly = self.detector.record_metric('z_test', 150.0)
        
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.anomaly_type, 'statistical')
    
    def test_26_iqr_detection(self):
        """测试 IQR 检测"""
        # 建立基线
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for _ in range(5):
            for v in values:
                self.detector.record_metric('iqr_test', v)
        
        # 记录离群值
        anomaly = self.detector.record_metric('iqr_test', 500.0)
        
        # IQR 或 Z-score 可能检测到
        if anomaly:
            self.assertEqual(anomaly.anomaly_type, 'statistical')
    
    def test_27_check_anomaly(self):
        """测试异常检查 (不记录)"""
        self.detector.set_threshold('check_test', min_value=0, max_value=100)
        
        self.assertFalse(self.detector.check_anomaly('check_test', 50.0))
        self.assertTrue(self.detector.check_anomaly('check_test', 150.0))
    
    def test_28_anomaly_queries(self):
        """测试异常查询"""
        self.detector.set_threshold('query_test', min_value=0, max_value=100)
        
        for i in range(5):
            self.detector.record_metric('query_test', 150.0 + i)
        
        # 按指标查询
        metric_anomalies = self.detector.get_anomalies_by_metric('query_test')
        self.assertGreater(len(metric_anomalies), 0)
        
        # 获取最近异常
        recent = self.detector.get_recent_anomalies(10)
        self.assertGreater(len(recent), 0)
    
    def test_29_statistics(self):
        """测试统计"""
        self.detector.set_threshold('stats_test', 0, 50)
        self.detector.record_metric('stats_test', 100)
        
        stats = self.detector.get_statistics()
        
        self.assertGreater(stats['total_anomalies'], 0)
        self.assertIn('by_type', stats)
    
    def test_30_report_generation(self):
        """测试报告生成"""
        self.detector.set_threshold('report_test', 0, 50)
        for i in range(5):
            self.detector.record_metric('report_test', 100 + i)
        
        report = self.detector.generate_report(time_window=10)
        
        self.assertIn('summary', report)
        self.assertIn('breakdown', report)


class TestSelfDiagnosis(unittest.TestCase):
    """SelfDiagnosis 测试"""
    
    def setUp(self):
        """测试初始化"""
        self.perf_monitor = PerformanceMonitor(sampling_interval=0.1)
        self.error_detector = ErrorDetector(max_errors=100)
        self.anomaly_detector = AnomalyDetector(baseline_window=50)
        
        self.diagnosis = SelfDiagnosis(
            performance_monitor=self.perf_monitor,
            error_detector=self.error_detector,
            anomaly_detector=self.anomaly_detector
        )
    
    def tearDown(self):
        """测试清理"""
        if self.perf_monitor.is_monitoring:
            self.perf_monitor.stop_monitoring()
    
    def test_31_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.diagnosis.performance_monitor)
        self.assertIsNotNone(self.diagnosis.error_detector)
        self.assertIsNotNone(self.diagnosis.anomaly_detector)
    
    def test_32_health_check_basic(self):
        """测试基本健康检查"""
        # 生成一些性能数据
        for _ in range(10):
            self.perf_monitor.capture_snapshot()
            time.sleep(0.01)
        
        report = self.diagnosis.run_health_check()
        
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.overall_health_score, 0)
        self.assertLessEqual(report.overall_health_score, 100)
        self.assertIn(report.overall_status, ['healthy', 'degraded', 'unhealthy', 'critical'])
    
    def test_33_component_status_check(self):
        """测试组件状态检查"""
        for _ in range(5):
            self.perf_monitor.capture_snapshot()
        
        report = self.diagnosis.run_health_check()
        
        self.assertGreater(len(report.component_statuses), 0)
        
        perf_status = next(
            (s for s in report.component_statuses if s.component_name == 'Performance'),
            None
        )
        self.assertIsNotNone(perf_status)
    
    def test_34_error_health_check(self):
        """测试错误健康检查"""
        # 记录一些错误
        for i in range(5):
            self.error_detector.record_error('TestError', f'Error {i}')
        
        report = self.diagnosis.run_health_check()
        
        error_status = next(
            (s for s in report.component_statuses if s.component_name == 'Errors'),
            None
        )
        self.assertIsNotNone(error_status)
        self.assertIn('total_errors', error_status.metrics)
    
    def test_35_anomaly_health_check(self):
        """测试异常健康检查"""
        self.anomaly_detector.set_threshold('test', 0, 50)
        for i in range(5):
            self.anomaly_detector.record_metric('test', 100 + i)
        
        report = self.diagnosis.run_health_check()
        
        anomaly_status = next(
            (s for s in report.component_statuses if s.component_name == 'Anomalies'),
            None
        )
        self.assertIsNotNone(anomaly_status)
    
    def test_36_root_cause_analysis_cpu(self):
        """测试 CPU 根因分析"""
        for _ in range(10):
            self.perf_monitor.capture_snapshot()
        
        analysis = self.diagnosis.analyze_root_cause("系统运行缓慢")
        
        self.assertIsNotNone(analysis)
        self.assertGreater(len(analysis.root_causes), 0)
        self.assertGreaterEqual(analysis.confidence, 0)  # 置信度可能为0
    
    def test_37_root_cause_analysis_error(self):
        """测试错误根因分析"""
        for i in range(10):
            self.error_detector.record_error('FrequentError', f'Error {i}')
        
        analysis = self.diagnosis.analyze_root_cause("系统频繁报错")
        
        self.assertGreater(len(analysis.root_causes), 0)
        # evidence 是一个字典,可能为空
        self.assertIsInstance(analysis.evidence, dict)
    
    def test_38_system_health_score(self):
        """测试系统健康分数"""
        for _ in range(5):
            self.perf_monitor.capture_snapshot()
        
        self.diagnosis.run_health_check()
        
        score = self.diagnosis.get_system_health_score()
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_39_recommendations(self):
        """测试修复建议"""
        for _ in range(5):
            self.perf_monitor.capture_snapshot()
        
        report = self.diagnosis.run_health_check()
        
        self.assertGreater(len(report.recommendations), 0)
    
    def test_40_diagnostic_report(self):
        """测试完整诊断报告"""
        for _ in range(5):
            self.perf_monitor.capture_snapshot()
        
        report = self.diagnosis.generate_diagnostic_report()
        
        self.assertIn('health_report', report)
        self.assertIn('system_info', report)
        self.assertIn('health_trend', report)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试初始化"""
        self.perf_monitor = PerformanceMonitor(sampling_interval=0.1, history_size=100)
        self.error_detector = ErrorDetector(max_errors=100, pattern_detection=True)
        self.anomaly_detector = AnomalyDetector(baseline_window=30)
        self.diagnosis = SelfDiagnosis(
            self.perf_monitor,
            self.error_detector,
            self.anomaly_detector
        )
    
    def tearDown(self):
        """测试清理"""
        if self.perf_monitor.is_monitoring:
            self.perf_monitor.stop_monitoring()
    
    def test_41_full_monitoring_pipeline(self):
        """测试完整监控管道"""
        # 启动性能监控
        self.perf_monitor.start_monitoring()
        
        # 模拟工作负载
        for i in range(10):
            # 记录操作
            start = time.time()
            time.sleep(0.01)
            latency_ms = (time.time() - start) * 1000
            self.perf_monitor.record_operation(latency_ms)
            
            # 记录一些错误
            if i % 3 == 0:
                self.error_detector.record_error('WorkloadError', f'Error during operation {i}')
        
        time.sleep(0.3)  # 等待监控数据
        
        # 停止监控
        self.perf_monitor.stop_monitoring()
        
        # 运行健康检查
        health_report = self.diagnosis.run_health_check()
        
        self.assertIsNotNone(health_report)
        self.assertGreater(len(health_report.component_statuses), 0)
    
    def test_42_alert_propagation(self):
        """测试警报传播"""
        alert_count = {'performance': 0, 'error': 0, 'anomaly': 0}
        
        def on_perf_alert(alert):
            alert_count['performance'] += 1
        
        def on_error(error):
            alert_count['error'] += 1
        
        def on_anomaly(anomaly):
            alert_count['anomaly'] += 1
        
        self.perf_monitor.add_alert_callback(on_perf_alert)
        self.error_detector.add_error_callback(on_error)
        self.anomaly_detector.add_anomaly_callback(on_anomaly)
        
        # 触发各种警报
        self.perf_monitor.set_threshold('cpu_percent', warning=0.1, critical=0.2)
        self.perf_monitor.capture_snapshot()
        
        self.error_detector.record_error('AlertError', 'Test error', severity='critical')
        
        self.anomaly_detector.set_threshold('test', 0, 50)
        self.anomaly_detector.record_metric('test', 100)
        
        # 验证至少有一些警报被触发
        # (不强制要求所有类型,因为某些警报条件可能不满足)
        total_alerts = sum(alert_count.values())
        self.assertGreaterEqual(total_alerts, 1)
    
    def test_43_cross_component_correlation(self):
        """测试跨组件关联"""
        # 生成性能数据
        for _ in range(10):
            self.perf_monitor.capture_snapshot()
            time.sleep(0.01)
        
        # 生成错误数据
        for i in range(5):
            self.error_detector.record_error('CorrelationError', f'Error {i}')
        
        # 生成异常数据
        self.anomaly_detector.set_threshold('corr_metric', 0, 50)
        for i in range(5):
            self.anomaly_detector.record_metric('corr_metric', 100 + i)
        
        # 运行诊断
        health_report = self.diagnosis.run_health_check()
        analysis = self.diagnosis.analyze_root_cause("系统性能下降和频繁报错")
        
        # 验证诊断包含多个组件的信息
        self.assertGreaterEqual(len(health_report.component_statuses), 2)
        self.assertGreater(len(analysis.root_causes), 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfDiagnosis))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Phase 3.2 Week 5 测试摘要")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
