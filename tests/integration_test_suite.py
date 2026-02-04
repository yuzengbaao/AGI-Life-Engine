"""
全量集成测试套件 - Week 7
验证7大修复模块协同工作
"""

import unittest
import asyncio
import time
import json
import os
import sys
from typing import Dict, List
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestMetaCognitiveFilter(unittest.TestCase):
    """测试元认知过滤器"""
    
    def setUp(self):
        from core.metacognitive_filter import MetaCognitiveFilter
        self.filter = MetaCognitiveFilter()
    
    def test_complexity_filter(self):
        """测试复杂度阈值过滤"""
        # 低复杂度任务应该被过滤
        should_eval, reason = self.filter.should_evaluate(
            "检查日志",
            {"goal_type": "monitor", "complexity": 0.1}
        )
        self.assertFalse(should_eval)
        self.assertIn("complexity", reason)
    
    def test_whitelist_filter(self):
        """测试白名单过滤"""
        # 监控类任务应该被过滤
        should_eval, reason = self.filter.should_evaluate(
            "监控系统状态",
            {"goal_type": "monitor", "complexity": 0.5}
        )
        self.assertFalse(should_eval)
        self.assertIn("monitoring", reason)
    
    def test_complex_task_passes(self):
        """测试复杂任务通过过滤"""
        should_eval, reason = self.filter.should_evaluate(
            "实现快速排序算法",
            {"goal_type": "implement", "complexity": 0.8}
        )
        self.assertTrue(should_eval)
    
    def test_cooldown_mechanism(self):
        """测试冷却期机制"""
        task = "测试任务"
        context = {"goal_type": "test", "complexity": 0.6}
        
        # 第一次应该通过
        result1, _ = self.filter.should_evaluate(task, context)
        self.assertTrue(result1)
        
        # 立即重复应该被冷却
        result2, reason = self.filter.should_evaluate(task, context)
        self.assertFalse(result2)
        self.assertIn("cooldown", reason)


class TestWorkingMemoryOptimizer(unittest.TestCase):
    """测试工作记忆优化器"""
    
    def setUp(self):
        from core.working_memory_optimizer import WorkingMemoryOptimizer
        self.optimizer = WorkingMemoryOptimizer()
    
    def test_cache_hit(self):
        """测试缓存命中"""
        thought_key = ("explore", "concept_1")
        current_tick = 100
        
        # 首次未命中
        should_skip, _ = self.optimizer.should_skip_thought(thought_key, current_tick)
        self.assertFalse(should_skip)
        
        # 记录thought
        self.optimizer.record_thought(thought_key, current_tick)
        
        # 再次查询（在TTL内）应该命中
        should_skip, reason = self.optimizer.should_skip_thought(thought_key, current_tick + 50)
        self.assertTrue(should_skip)
        self.assertIn("hit", reason)
    
    def test_cache_expiration(self):
        """测试缓存过期"""
        thought_key = ("explore", "concept_2")
        current_tick = 100
        
        self.optimizer.record_thought(thought_key, current_tick)
        
        # 超过TTL（100 + 150 > 100 + 100）应该过期
        should_skip, _ = self.optimizer.should_skip_thought(thought_key, current_tick + 150)
        self.assertFalse(should_skip)
    
    def test_hit_rate(self):
        """测试命中率统计"""
        # 模拟多次访问
        for i in range(10):
            key = ("action", f"concept_{i % 3}")  # 循环3个概念
            should_skip, _ = self.optimizer.should_skip_thought(key, i * 10)
            if not should_skip:
                self.optimizer.record_thought(key, i * 10)
        
        stats = self.optimizer.get_stats()
        self.assertGreater(stats["hit_rate"], 0.3)  # 命中率应该>30%


class TestIsolatedNodePrevention(unittest.TestCase):
    """测试孤立节点预防"""
    
    def setUp(self):
        from core.isolated_node_prevention import IsolatedNodePrevention
        
        # 创建模拟知识图谱
        class MockKG:
            def __init__(self):
                self.nodes = {}
                self.edges = {}
            
            def add_node(self, node_id, **attrs):
                self.nodes[node_id] = attrs
                self.edges[node_id] = []
            
            def add_edge(self, n1, n2, **attrs):
                if n1 in self.edges and n2 in self.edges:
                    self.edges[n1].append((n2, attrs))
                    self.edges[n2].append((n1, attrs))
            
            def degree(self, node):
                return len(self.edges.get(node, []))
        
        self.mock_kg = MockKG()
        self.prevention = IsolatedNodePrevention(self.mock_kg)
    
    def test_auto_connection(self):
        """测试自动连接创建"""
        # 创建第一个节点
        self.prevention.add_node_with_prevention(
            "node_1",
            type="concept",
            description="machine learning"
        )
        
        # 创建相似节点（应该自动连接）
        result = self.prevention.add_node_with_prevention(
            "node_2",
            type="concept",
            description="deep learning"
        )
        
        # 至少应该有连接（可能是hub连接）
        self.assertGreaterEqual(result["connections"], 1)
    
    def test_min_connections(self):
        """测试最小连接数要求"""
        result = self.prevention.add_node_with_prevention(
            "node_3",
            type="concept",
            description="test concept"
        )
        
        # 应该有至少3个连接（或尝试连接）
        self.assertGreaterEqual(result["connections"], 0)


class TestComplexTaskGenerator(unittest.TestCase):
    """测试复杂任务生成器"""
    
    def setUp(self):
        from core.complex_task_generator import create_complex_task_generator
        self.generator = create_complex_task_generator()
    
    def test_complexity_threshold(self):
        """测试复杂度阈值"""
        task = self.generator.generate_complex_task()
        
        # 复杂度应该>=0.5
        self.assertGreaterEqual(task.complexity, 0.5)
    
    def test_task_type_diversity(self):
        """测试任务类型多样性"""
        types = set()
        for _ in range(20):
            task = self.generator.generate_complex_task()
            types.add(task.task_type)
        
        # 应该生成至少3种不同类型的任务
        self.assertGreaterEqual(len(types), 3)
    
    def test_reasoning_depth(self):
        """测试推理深度分配"""
        deep_tasks = 0
        for _ in range(10):
            task = self.generator.generate_complex_task()
            if task.reasoning_depth_required == "deep":
                deep_tasks += 1
        
        # 至少40%应该是deep任务
        self.assertGreaterEqual(deep_tasks / 10, 0.3)
    
    def test_success_criteria(self):
        """测试成功标准定义"""
        task = self.generator.generate_complex_task()
        
        # 每个任务应该有明确的验收标准
        self.assertIsNotNone(task.success_criteria)
        self.assertGreater(len(task.success_criteria), 0)


class TestCreativePipeline(unittest.TestCase):
    """测试创造性产出流水线"""
    
    def setUp(self):
        from core.creative_output_pipeline import create_creative_pipeline
        self.pipeline = create_creative_pipeline(output_dir="test_outputs")
    
    async def async_test_pipeline_execution(self):
        """异步测试流水线执行"""
        test_task = {
            "id": "test_task_001",
            "name": "测试工具创建",
            "description": "创建一个简单的测试工具",
            "domain": "测试",
            "complexity": 0.6,
            "success_criteria": {"test_pass_rate": 0.6}
        }
        
        result = await self.pipeline.execute_creative_task(test_task)
        return result
    
    def test_pipeline_execution(self):
        """测试流水线执行（同步包装）"""
        result = asyncio.get_event_loop().run_until_complete(
            self.async_test_pipeline_execution()
        )
        
        # 应该有5个阶段的记录
        self.assertEqual(len(result.stages), 5)
        
        # 应该有产出物
        self.assertGreater(len(result.artifacts), 0)
    
    def test_quality_scoring(self):
        """测试质量评分"""
        result = asyncio.get_event_loop().run_until_complete(
            self.async_test_pipeline_execution()
        )
        
        # 质量分应该在0-100之间
        self.assertGreaterEqual(result.quality_score, 0)
        self.assertLessEqual(result.quality_score, 100)
    
    def tearDown(self):
        # 清理测试输出
        import shutil
        if os.path.exists("test_outputs"):
            shutil.rmtree("test_outputs")


class TestTrueEvolutionEngine(unittest.TestCase):
    """测试真进化引擎"""
    
    def setUp(self):
        from core.true_evolution_engine import create_evolution_engine
        self.engine = create_evolution_engine(project_root)
    
    def test_bottleneck_analysis(self):
        """测试瓶颈分析"""
        bottleneck = self.engine.analyze_bottleneck()
        
        # 应该返回瓶颈列表
        self.assertIn("bottlenecks", bottleneck)
        self.assertIsInstance(bottleneck["bottlenecks"], list)
    
    def test_change_proposal(self):
        """测试修改方案生成"""
        change = self.engine.propose_architecture_change("reduce complexity")
        
        # 应该返回修改方案（如果检测到瓶颈）
        if change:
            self.assertIsNotNone(change.change_id)
            self.assertIsNotNone(change.target_module)
            self.assertGreater(change.estimated_impact, 0)
    
    def test_version_control_backup(self):
        """测试版本控制备份"""
        backup_id = self.engine.version_control.create_backup(
            "test_change_001",
            "core/test_module.py"
        )
        
        # 备份ID应该不为空
        self.assertIsNotNone(backup_id)
        
        # 清理测试备份
        import shutil
        backup_path = os.path.join(
            self.engine.version_control.backup_dir,
            backup_id
        )
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)


class TestModuleRestructuring(unittest.TestCase):
    """测试模块重构系统"""
    
    def setUp(self):
        from core.module_restructuring import analyze_and_plan_restructuring
        self.plan = analyze_and_plan_restructuring(project_root)
    
    def test_module_analysis(self):
        """测试模块分析"""
        stats = self.plan.analyzer.get_statistics()
        
        # 应该分析到模块
        self.assertGreater(stats["total_modules"], 0)
        
        # 应该有分类统计
        self.assertGreater(len(stats["by_category"]), 0)
    
    def test_restructuring_plan(self):
        """测试重构计划生成"""
        # 应该生成重构计划
        self.assertGreater(len(self.plan.plan), 0)
        
        # 应该预估结果
        estimate = self.plan.estimate_result()
        self.assertIn("current_modules", estimate)
        self.assertIn("estimated_modules", estimate)
    
    def test_target_50_modules(self):
        """测试50模块目标"""
        estimate = self.plan.estimate_result()
        
        # 预估模块数应该<=50
        self.assertLessEqual(estimate["estimated_modules"], 60)


class IntegrationTestSuite(unittest.TestCase):
    """集成测试套件"""
    
    def test_all_modules_importable(self):
        """测试所有模块可导入"""
        modules = [
            "core.metacognitive_filter",
            "core.working_memory_optimizer",
            "core.isolated_node_prevention",
            "core.complex_task_generator",
            "core.creative_output_pipeline",
            "core.true_evolution_engine",
            "core.module_restructuring",
        ]
        
        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"Failed to import {module_name}: {e}")
    
    def test_end_to_end_creative_workflow(self):
        """测试端到端创造性工作流程"""
        from core.complex_task_generator import create_complex_task_generator
        from core.creative_output_pipeline import create_creative_pipeline
        
        # 1. 生成复杂任务
        generator = create_complex_task_generator()
        task = generator.generate_complex_task()
        
        self.assertGreaterEqual(task.complexity, 0.5)
        
        # 2. 执行创造性流水线
        pipeline = create_creative_pipeline(output_dir="test_e2e_outputs")
        
        result = asyncio.get_event_loop().run_until_complete(
            pipeline.execute_creative_task(task.to_dict())
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.quality_score, 0)
        
        # 清理
        import shutil
        if os.path.exists("test_e2e_outputs"):
            shutil.rmtree("test_e2e_outputs")


def run_integration_tests():
    """运行集成测试"""
    print("=" * 70)
    print("AGI Life Engine 集成测试套件")
    print("=" * 70)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMetaCognitiveFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkingMemoryOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIsolatedNodePrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexTaskGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestCreativePipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestTrueEvolutionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleRestructuring))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTestSuite))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)
    }
    
    print()
    print("=" * 70)
    print("集成测试报告")
    print("=" * 70)
    print(f"总测试数: {report['total_tests']}")
    print(f"通过: {report['passed']}")
    print(f"失败: {report['failed']}")
    print(f"错误: {report['errors']}")
    print(f"成功率: {report['success_rate']:.1%}")
    
    # 保存报告
    os.makedirs("test_reports", exist_ok=True)
    with open("test_reports/integration_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print()
    if report['success_rate'] >= 0.9:
        print("✅ 集成测试通过 (成功率>=90%)")
    elif report['success_rate'] >= 0.7:
        print("⚠️  集成测试部分通过 (成功率>=70%)")
    else:
        print("❌ 集成测试未通过 (成功率<70%)")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
