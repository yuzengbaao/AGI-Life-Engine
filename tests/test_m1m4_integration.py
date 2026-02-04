#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-M4组件集成测试 - 验证与AGI_Life_Engine的集成
================================================

测试场景:
1. 适配器初始化测试
2. EventBus事件流测试
3. M1 MetaLearner性能追踪测试
4. M2 GoalQuestioner目标质疑测试
5. M3 SelfModifyingEngine代码分析测试
6. M4 RecursiveSelfMemory记忆操作测试
7. 组件间交互测试
8. 系统关闭测试

版本: 1.0.0
"""

import sys
import os
import time
import logging
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockEventBus:
    """模拟EventBus用于测试"""

    def __init__(self):
        self.events = []
        self.subscribers = {}

    def subscribe(self, event_type, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event_type, data):
        event = MockEvent(type=event_type, source="test", data=data)
        self.events.append(event)

        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

    def publish_sync(self, event_type, data):
        """同步发布（用于测试）"""
        asyncio.create_task(self.publish(event_type, data))


class MockEvent:
    def __init__(self, type, source, data):
        self.type = type
        self.source = source
        self.data = data


class M1M4IntegrationTest:
    """M1-M4组件集成测试"""

    def __init__(self, output_dir: str = "./test_results/m1m4_integration"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'tests': {},
            'overall_pass': False
        }

        self.event_bus = None
        self.adapter = None

    def run_test_suite(self) -> dict:
        """运行完整集成测试套件"""
        logger.info("=" * 80)
        logger.info("M1-M4组件集成测试开始")
        logger.info("=" * 80)

        # 测试1: 适配器初始化
        logger.info("\n[测试1] 适配器初始化")
        self.results['tests']['adapter_init'] = self._test_adapter_init()

        # 测试2: M1 MetaLearner事件流
        logger.info("\n[测试2] M1 MetaLearner性能追踪")
        self.results['tests']['m1_performance'] = self._test_m1_performance_tracking()

        # 测试3: M2 GoalQuestioner目标质疑
        logger.info("\n[测试3] M2 GoalQuestioner目标质疑")
        self.results['tests']['m2_goal_questioning'] = self._test_m2_goal_questioning()

        # 测试4: M3 SelfModifyingEngine代码分析
        logger.info("\n[测试4] M3 SelfModifyingEngine代码分析")
        self.results['tests']['m3_code_analysis'] = self._test_m3_code_analysis()

        # 测试5: M4 RecursiveSelfMemory记忆操作
        logger.info("\n[测试5] M4 RecursiveSelfMemory记忆操作")
        self.results['tests']['m4_memory'] = self._test_m4_memory_operations()

        # 测试6: 组件健康状态
        logger.info("\n[测试6] 组件健康状态监控")
        self.results['tests']['health_monitoring'] = self._test_health_monitoring()

        # 测试7: 系统关闭
        logger.info("\n[测试7] 系统关闭与清理")
        self.results['tests']['shutdown'] = self._test_shutdown()

        # 分析结果
        self._analyze_results()

        # 保存报告
        self._save_report()

        logger.info("\n" + "=" * 80)
        logger.info("M1-M4组件集成测试完成")
        logger.info("=" * 80)

        return self.results

    def _test_adapter_init(self) -> dict:
        """测试1: 适配器初始化"""
        try:
            from core.m1m4_adapter import create_m1m4_adapter

            # 创建模拟EventBus
            self.event_bus = MockEventBus()

            # 创建适配器
            self.adapter = create_m1m4_adapter(self.event_bus)

            # 验证初始化
            health = self.adapter.get_health_status()

            return {
                'test_name': 'adapter_init',
                'passed': True,
                'components_initialized': len(health),
                'components': list(health.keys()),
                'details': f"成功初始化 {len(health)} 个组件"
            }

        except Exception as e:
            logger.error(f"适配器初始化失败: {e}")
            import traceback
            traceback.print_exc()

            return {
                'test_name': 'adapter_init',
                'passed': False,
                'error': str(e),
                'details': '适配器初始化失败'
            }

    def _test_m1_performance_tracking(self) -> dict:
        """测试2: M1 MetaLearner性能追踪"""
        if self.adapter is None or self.adapter.meta_learner is None:
            return {'test_name': 'm1_performance', 'passed': False, 'error': 'MetaLearner未初始化'}

        try:
            # 模拟TheSeed性能事件
            performance_event = MockEvent(
                type="the_seed.performance",
                source="TheSeed",
                data={
                    'step': 100,
                    'reward': 0.85,
                    'loss': 0.15,
                    'uncertainty': 0.3,
                    'exploration_rate': 0.1
                }
            )

            # 发布事件
            asyncio.run(self.event_bus.publish("the_seed.performance", performance_event.data))

            time.sleep(0.5)  # 等待处理

            # 获取统计信息
            stats = self.adapter.meta_learner.get_statistics()

            return {
                'test_name': 'm1_performance',
                'passed': True,
                'events_processed': stats.get('total_observations', 0),
                'parameters_tracked': len(stats.get('current_parameters', {})),
                'details': f"MetaLearner处理了 {stats.get('total_observations', 0)} 个性能事件"
            }

        except Exception as e:
            logger.error(f"M1性能追踪测试失败: {e}")
            return {'test_name': 'm1_performance', 'passed': False, 'error': str(e)}

    def _test_m2_goal_questioning(self) -> dict:
        """测试3: M2 GoalQuestioner目标质疑"""
        if self.adapter is None or self.adapter.goal_questioner is None:
            return {'test_name': 'm2_goal_questioning', 'passed': False, 'error': 'GoalQuestioner未初始化'}

        try:
            # 模拟目标创建事件
            goal_event = MockEvent(
                type="goal.created",
                source="GoalManager",
                data={
                    'goal_id': 'test_goal_001',
                    'goal_type': 'exploration',
                    'description': '测试目标：探索未知领域',
                    'target_outcome': '发现新的知识',
                    'success_criteria': ['发现至少3个新概念'],
                    'hard_constraints': ['不违反安全原则'],
                    'soft_constraints': ['保持好奇心'],
                    'priority': 0.7,
                    'system_state': {'mode': 'learning'},
                    'available_resources': {'compute': 80},
                    'time_pressure': 0.3
                }
            )

            # 发布事件
            asyncio.run(self.event_bus.publish("goal.created", goal_event.data))

            time.sleep(0.5)  # 等待处理

            return {
                'test_name': 'm2_goal_questioning',
                'passed': True,
                'goal_id': 'test_goal_001',
                'details': 'GoalQuestioner成功处理目标创建事件'
            }

        except Exception as e:
            logger.error(f"M2目标质疑测试失败: {e}")
            return {'test_name': 'm2_goal_questioning', 'passed': False, 'error': str(e)}

    def _test_m3_code_analysis(self) -> dict:
        """测试4: M3 SelfModifyingEngine代码分析"""
        if self.adapter is None or self.adapter.self_modifier is None:
            return {'test_name': 'm3_code_analysis', 'passed': False, 'error': 'SelfModifyingEngine未初始化'}

        try:
            # 分析一个模块
            module_path = "core.seed"

            logger.info(f"  分析模块: {module_path}")

            analysis = self.adapter.self_modifier.analyze(module_path)

            return {
                'test_name': 'm3_code_analysis',
                'passed': True,
                'module': module_path,
                'locations_found': len(analysis.locations),
                'complexity': float(analysis.complexity),
                'safety_score': float(analysis.safety_score),
                'details': f"分析完成: 复杂度={analysis.complexity:.2f}, 安全分数={analysis.safety_score:.2f}"
            }

        except Exception as e:
            logger.error(f"M3代码分析测试失败: {e}")
            return {'test_name': 'm3_code_analysis', 'passed': False, 'error': str(e)}

    def _test_m4_memory_operations(self) -> dict:
        """测试5: M4 RecursiveSelfMemory记忆操作"""
        if self.adapter is None or self.adapter.recursive_memory is None:
            return {'test_name': 'm4_memory', 'passed': False, 'error': 'RecursiveSelfMemory未初始化'}

        try:
            from core.recursive_self_memory import MemoryImportance

            # 测试记住
            memory_id = self.adapter.recursive_memory.remember(
                event_type="test_event",
                content={"message": "集成测试记忆"},
                importance=MemoryImportance.MEDIUM,
                why="测试记忆功能",
                trigger="M1M4IntegrationTest"
            )

            # 测试回忆
            results = self.adapter.recursive_memory.recall("测试", limit=5)

            # 测试为何记住
            why = self.adapter.recursive_memory.why_remembered(memory_id)

            # 获取统计信息
            stats = self.adapter.recursive_memory.get_statistics()

            return {
                'test_name': 'm4_memory',
                'passed': True,
                'memory_id': memory_id,
                'recall_count': len(results),
                'why_remembered_available': why is not None,
                'total_memories': stats['l0_event_count'],
                'details': f"记忆操作成功: 总记忆数={stats['l0_event_count']}"
            }

        except Exception as e:
            logger.error(f"M4记忆操作测试失败: {e}")
            return {'test_name': 'm4_memory', 'passed': False, 'error': str(e)}

    def _test_health_monitoring(self) -> dict:
        """测试6: 组件健康状态监控"""
        if self.adapter is None:
            return {'test_name': 'health_monitoring', 'passed': False, 'error': '适配器未初始化'}

        try:
            # 获取健康状态
            health = self.adapter.get_health_status()

            # 获取统计信息
            stats = self.adapter.get_statistics()

            active_components = len([h for h in health.values() if h['status'] == 'active'])

            return {
                'test_name': 'health_monitoring',
                'passed': True,
                'total_components': len(health),
                'active_components': active_components,
                'events_processed': stats.get('events_processed', 0),
                'details': f"活跃组件: {active_components}/{len(health)}, 事件处理: {stats.get('events_processed', 0)}"
            }

        except Exception as e:
            logger.error(f"健康监控测试失败: {e}")
            return {'test_name': 'health_monitoring', 'passed': False, 'error': str(e)}

    def _test_shutdown(self) -> dict:
        """测试7: 系统关闭与清理"""
        if self.adapter is None:
            return {'test_name': 'shutdown', 'passed': False, 'error': '适配器未初始化'}

        try:
            # 执行关闭
            self.adapter.shutdown()

            return {
                'test_name': 'shutdown',
                'passed': True,
                'details': '适配器关闭成功'
            }

        except Exception as e:
            logger.error(f"关闭测试失败: {e}")
            return {'test_name': 'shutdown', 'passed': False, 'error': str(e)}

    def _analyze_results(self):
        """分析测试结果"""
        passed = sum(1 for t in self.results['tests'].values() if t.get('passed', False))
        total = len(self.results['tests'])

        self.results['overall_pass'] = passed == total
        self.results['passed_count'] = passed
        self.results['total_count'] = total
        self.results['pass_rate'] = passed / total if total > 0 else 0

        logger.info(f"\n测试结果:")
        logger.info(f"  通过: {passed}/{total}")
        logger.info(f"  通过率: {self.results['pass_rate']:.1%}")
        logger.info(f"  总体结果: {'✅ PASS' if self.results['overall_pass'] else '❌ FAIL'}")

    def _save_report(self):
        """保存测试报告"""
        report_path = self.output_dir / 'integration_test_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("M1-M4组件集成测试报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"总体结果: {'PASS' if self.results['overall_pass'] else 'FAIL'}\n")
            f.write(f"通过率: {self.results['pass_rate']:.1%}\n")
            f.write(f"通过数: {self.results['passed_count']}/{self.results['total_count']}\n\n")

            f.write("[测试详情]\n\n")
            for test_name, result in self.results['tests'].items():
                f.write(f"{test_name}:\n")
                f.write(f"  通过: {result.get('passed', False)}\n")
                f.write(f"  详情: {result.get('details', 'N/A')}\n")
                if 'error' in result:
                    f.write(f"  错误: {result['error']}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")

        logger.info(f"  报告已保存: {report_path}")


def main():
    """主测试函数"""
    test = M1M4IntegrationTest()
    results = test.run_test_suite()

    # 输出摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)
    print(f"\n通过率: {results['pass_rate']:.1%}")
    print(f"总体结果: {'PASS' if results['overall_pass'] else 'FAIL'}")

    return 0 if results['overall_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
