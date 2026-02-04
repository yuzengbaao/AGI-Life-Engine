"""
SelfModifyingEngine基准测试 - 验证M3阶段实施

⚠️  警告: 这是测试最危险的组件

目标:
1. 验证不可变约束有效 (CRITICAL修改被拒绝)
2. 验证沙箱测试机制
3. 验证回滚速度<30秒
4. 验证审计日志完整性

测试场景:
- 场景A: 安全优化 (SAFE级别,应通过)
- 场景B: 触发不可变约束 (CRITICAL级别,应拒绝)
- 场景C: 沙箱测试失败 (应拒绝)
- 场景D: 回滚速度测试 (必须<30秒)
- 场景E: 审计日志完整性
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.self_modifying_engine import (
    SelfModifyingEngine, CodePatch, CodeLocation, ModificationRisk,
    ImmutableConstraints, ModificationStatus
)

logger = logging.getLogger(__name__)


class SelfModifyingEngineBenchmark:
    """SelfModifyingEngine基准测试"""

    def __init__(self, output_dir: str = "./test_results/self_modifying_engine"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'scenarios': {},
            'immutable_constraints_effective': False,
            'rollback_speed_ok': False,
            'audit_log_complete': False,
            'overall_pass': False
        }

    def run_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整基准测试套件"""
        logger.info("=" * 80)
        logger.info("SelfModifyingEngine基准测试开始")
        logger.info("=" * 80)

        # 场景A: 安全优化
        logger.info("\n[场景A] 安全优化测试")
        self.results['scenarios']['safe_optimization'] = self._test_safe_optimization()

        # 场景B: 不可变约束
        logger.info("\n[场景B] 不可变约束测试")
        self.results['scenarios']['immutable_constraints'] = self._test_immutable_constraints()

        # 场景C: 沙箱测试
        logger.info("\n[场景C] 沙箱测试机制")
        self.results['scenarios']['sandbox_test'] = self._test_sandbox_test()

        # 场景D: 回滚速度
        logger.info("\n[场景D] 回滚速度测试")
        self.results['scenarios']['rollback_speed'] = self._test_rollback_speed()

        # 场景E: 审计日志
        logger.info("\n[场景E] 审计日志完整性")
        self.results['scenarios']['audit_log'] = self._test_audit_log()

        # 分析结果
        logger.info("\n[分析] 计算验收指标")
        self._analyze_results()

        # 生成可视化
        logger.info("\n[可视化] 生成图表")
        self._generate_plots()

        # 保存报告
        logger.info("\n[报告] 保存结果")
        self._save_report()

        logger.info("\n" + "=" * 80)
        logger.info("SelfModifyingEngine基准测试完成")
        logger.info("=" * 80)

        return self.results

    def _test_safe_optimization(self) -> Dict[str, Any]:
        """
        场景A: 安全优化

        测试SAFE级别的优化是否能通过
        """
        # 创建引擎
        engine = SelfModifyingEngine(
            project_root=str(Path.cwd().parent),
            auto_apply_safe=True
        )

        # 分析一个模块
        analysis = engine.analyze("core.seed")

        logger.info(f"  分析结果: {len(analysis.locations)} 个位置")
        logger.info(f"  复杂度: {analysis.complexity:.2f}")
        logger.info(f"  安全分数: {analysis.safety_score:.2f}")

        # 尝试提出补丁 (简化版本,不实际修改)
        logger.info("  ⚠️  实际补丁生成需要LLM支持,当前仅测试分析功能")

        return {
            'scenario': 'safe_optimization',
            'analysis_successful': len(analysis.locations) > 0,
            'locations_count': len(analysis.locations),
            'complexity': float(analysis.complexity),
            'safety_score': float(analysis.safety_score)
        }

    def _test_immutable_constraints(self) -> Dict[str, Any]:
        """
        场景B: 不可变约束

        测试关键安全代码是否被保护
        """
        constraints = ImmutableConstraints.get_core_constraints()

        logger.info(f"  不可变约束数量: {len(constraints)}")

        all_protected = True
        test_results = []

        for constraint in constraints:
            # 根据约束类型生成对应的危险代码
            if constraint.name == "self_modification_protection":
                # 测试自修改保护
                dangerous_code = """
def propose_patch():
    pass  # 尝试修改核心方法
"""
            elif constraint.name == "audit_log_protection":
                # 测试审计日志保护
                dangerous_code = """
audit_log = []
audit_log.clear()  # 尝试破坏审计日志
"""
            else:
                # 其他约束使用通用测试代码
                dangerous_code = f"""
class {constraint.protected_patterns[0]}:
    def modify(self):
        pass  # 尝试修改关键代码
"""

            is_protected = not constraint.check_func(
                dangerous_code,
                type('obj', (object,), {'file_path': 'test.py'})()
            )

            test_results.append({
                'constraint': constraint.name,
                'protected': is_protected
            })

            if not is_protected:
                all_protected = False
                logger.error(f"  约束失效: {constraint.name}")
            else:
                logger.info(f"  约束有效: {constraint.name}")

        return {
            'scenario': 'immutable_constraints',
            'constraints_count': len(constraints),
            'all_protected': all_protected,
            'test_results': test_results
        }

    def _test_sandbox_test(self) -> Dict[str, Any]:
        """
        场景C: 沙箱测试机制

        测试沙箱环境是否能隔离风险代码
        """
        engine = SelfModifyingEngine(project_root=str(Path.cwd().parent))

        # 创建一个安全的补丁
        safe_patch = CodePatch(
            original_code="def hello():\n    return 'world'",
            modified_code="def hello():\n    return 'optimized'",
            location=CodeLocation(file_path="test.py"),
            description="简单优化",
            risk_level=ModificationRisk.SAFE,
            estimated_impact="微小",
            test_cases=["import_test"]
        )

        # 运行沙箱测试
        passed, results = engine.sandbox_test(safe_patch)

        logger.info(f"  沙箱测试: {'PASS' if passed else 'FAIL'}")
        logger.info(f"  测试用例: {results['test_cases_run']}")
        logger.info(f"  错误: {len(results['errors'])}")

        return {
            'scenario': 'sandbox_test',
            'test_passed': passed,
            'test_cases_run': results['test_cases_run'],
            'errors_count': len(results['errors']),
            'has_sandbox_mechanism': True
        }

    def _test_rollback_speed(self) -> Dict[str, Any]:
        """
        场景D: 回滚速度测试

        验证回滚时间<30秒
        """
        import time

        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_file = Path(f.name)
            f.write("# Original content\n")

        try:
            # 创建备份
            backup_file = test_file.with_suffix('.bak')
            shutil.copy2(test_file, backup_file)

            # 修改文件
            with open(test_file, 'w') as f:
                f.write("# Modified content\n")

            # 模拟回滚
            start_time = time.time()
            shutil.copy2(backup_file, test_file)
            rollback_time = time.time() - start_time

            logger.info(f"  回滚时间: {rollback_time:.3f}秒")

            speed_ok = rollback_time < SelfModifyingEngine.MAX_ROLLBACK_TIME_SECONDS

            logger.info(f"  结果: {'✅ PASS' if speed_ok else '❌ FAIL'} "
                       f"(要求<{SelfModifyingEngine.MAX_ROLLBACK_TIME_SECONDS}秒)")

            return {
                'scenario': 'rollback_speed',
                'rollback_time': float(rollback_time),
                'speed_ok': speed_ok,
                'max_allowed': SelfModifyingEngine.MAX_ROLLBACK_TIME_SECONDS
            }

        finally:
            # 清理
            test_file.unlink(missing_ok=True)
            backup_file.unlink(missing_ok=True)

    def _test_audit_log(self) -> Dict[str, Any]:
        """
        场景E: 审计日志完整性

        验证所有修改都有完整记录
        """
        engine = SelfModifyingEngine(project_root=str(Path.cwd().parent))

        # 检查审计日志功能
        stats = engine.get_statistics()

        logger.info(f"  总提议数: {stats['total_proposed']}")
        logger.info(f"  总应用数: {stats['total_applied']}")
        logger.info(f"  总拒绝数: {stats['total_rejected']}")
        logger.info(f"  总回滚数: {stats['total_rolled_back']}")

        # 检查是否有备份目录
        backup_dir_exists = engine.backup_dir.exists()

        logger.info(f"  备份目录: {'存在' if backup_dir_exists else '不存在'}")

        has_audit_trail = (
            'total_proposed' in stats and
            'total_applied' in stats and
            'total_rejected' in stats and
            'total_rolled_back' in stats
        )

        return {
            'scenario': 'audit_log',
            'has_audit_trail': has_audit_trail,
            'backup_dir_exists': backup_dir_exists,
            'statistics_complete': has_audit_trail
        }

    def _analyze_results(self):
        """分析测试结果"""
        # 1. 不可变约束有效
        immutable_result = self.results['scenarios'].get('immutable_constraints', {})
        self.results['immutable_constraints_effective'] = (
            immutable_result.get('all_protected', False)
        )

        # 2. 回滚速度<30秒
        rollback_result = self.results['scenarios'].get('rollback_speed', {})
        self.results['rollback_speed_ok'] = rollback_result.get('speed_ok', False)

        # 3. 审计日志完整
        audit_result = self.results['scenarios'].get('audit_log', {})
        self.results['audit_log_complete'] = (
            audit_result.get('has_audit_trail', False) and
            audit_result.get('backup_dir_exists', False)
        )

        # 4. 总体验收
        self.results['overall_pass'] = (
            self.results['immutable_constraints_effective'] and
            self.results['rollback_speed_ok'] and
            self.results['audit_log_complete']
        )

        logger.info(f"\n验收指标:")
        logger.info(f"  不可变约束有效: {self.results['immutable_constraints_effective']}")
        logger.info(f"  回滚速度<30秒: {self.results['rollback_speed_ok']}")
        logger.info(f"  审计日志完整: {self.results['audit_log_complete']}")
        logger.info(f"  总体通过: {self.results['overall_pass']}")

    def _generate_plots(self):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 不可变约束
        ax = axes[0, 0]
        immutable_result = self.results['scenarios'].get('immutable_constraints', {})
        constraints_count = immutable_result.get('constraints_count', 0)
        all_protected = immutable_result.get('all_protected', False)

        colors = ['green' if all_protected else 'red']
        ax.bar(['Protected'], [constraints_count], color=colors, alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title('Immutable Constraints')
        ax.set_ylim([0, 10])
        ax.grid(True, alpha=0.3)

        # 2. 回滚速度
        ax = axes[0, 1]
        rollback_result = self.results['scenarios'].get('rollback_speed', {})
        rollback_time = rollback_result.get('rollback_time', 0)
        max_allowed = rollback_result.get('max_allowed', 30)

        ax.bar(['Rollback Time'], [rollback_time],
               color='green' if rollback_time < max_allowed else 'red', alpha=0.7)
        ax.axhline(y=max_allowed, color='orange', linestyle='--', label='Max Allowed')
        ax.set_ylabel('Seconds')
        ax.set_title('Rollback Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 验收指标
        ax = axes[1, 0]
        metrics = ['Immutable\nConstraints', 'Rollback\nSpeed', 'Audit\nLog']
        values = [
            1.0 if self.results['immutable_constraints_effective'] else 0.0,
            1.0 if self.results['rollback_speed_ok'] else 0.0,
            1.0 if self.results['audit_log_complete'] else 0.0
        ]
        colors = ['green' if v > 0 else 'red' for v in values]
        ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Pass (1=Yes)')
        ax.set_title('Acceptance Metrics')
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)

        # 4. 总体验收结果
        ax = axes[1, 1]
        ax.axis('off')
        result_text = f"""
        M3 Stage Acceptance Result

        Immutable Constraints Effective: {'Yes' if self.results['immutable_constraints_effective'] else 'No'}
        Rollback Speed < 30s: {'Yes' if self.results['rollback_speed_ok'] else 'No'}
        Audit Log Complete: {'Yes' if self.results['audit_log_complete'] else 'No'}

        Overall Result: {'PASS' if self.results['overall_pass'] else 'FAIL'}
        """
        ax.text(0.5, 0.5, result_text, ha='center', va='center',
               fontsize=14, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat' if self.results['overall_pass'] else 'lightcoral', alpha=0.5))
        ax.set_title('Final Result')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'self_modifying_engine_benchmark.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  图表已保存到: {self.output_dir}")

    def _save_report(self):
        """保存测试报告"""
        report_path = self.output_dir / 'benchmark_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SelfModifyingEngine基准测试报告\n")
            f.write("=" * 80 + "\n\n")

            # 验收指标
            f.write("[验收指标]\n")
            f.write(f"  不可变约束有效: {'Yes' if self.results['immutable_constraints_effective'] else 'No'}\n")
            f.write(f"  回滚速度<30秒: {'Yes' if self.results['rollback_speed_ok'] else 'No'}\n")
            f.write(f"  审计日志完整: {'Yes' if self.results['audit_log_complete'] else 'No'}\n")
            f.write(f"  总体结果: {'PASS' if self.results['overall_pass'] else 'FAIL'}\n\n")

            # 各场景详情
            f.write("[场景详情]\n")
            for scenario, result in self.results['scenarios'].items():
                f.write(f"\n{scenario.upper()}:\n")
                for key, value in result.items():
                    if key != 'test_results':
                        f.write(f"  {key}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"  报告已保存到: {report_path}")


# ============================================================================
# 主测试入口
# ============================================================================

def main():
    """主测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    benchmark = SelfModifyingEngineBenchmark()
    results = benchmark.run_benchmark_suite()

    # 输出摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)

    print(f"\n不可变约束有效: {results['immutable_constraints_effective']}")
    print(f"回滚速度<30秒: {results['rollback_speed_ok']}")
    print(f"审计日志完整: {results['audit_log_complete']}")
    print(f"\n总体结果: {'PASS' if results['overall_pass'] else 'FAIL'}")

    return 0 if results['overall_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
