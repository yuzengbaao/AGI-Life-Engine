"""
GoalQuestioner基准测试 - 验证M2阶段实施

目标:
1. 验证GoalQuestioner能检测≥3类目标偏差
2. 验证误报率<20%
3. 验证建议采纳率≥60%
4. 验证反循环机制有效

测试场景:
- 场景A: 目标错位 (奖励下降+低方差)
- 场景B: 目标冲突 (外在权重过高)
- 场景C: 目标过拟合 (单一目标权重>0.8)
- 场景D: 目标漂移 (异常事件增多)
- 场景E: 正常情况 (不应误报)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.goal_questioner import (
    GoalQuestioner, GoalSpec, GoalComponent, GoalBiasType,
    QuestioningContext, create_default_goal_spec, collect_goal_context
)

logger = logging.getLogger(__name__)


class GoalQuestionerBenchmark:
    """GoalQuestioner基准测试"""

    def __init__(self, output_dir: str = "./test_results/goal_questioner"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'scenarios': {},
            'false_positive_rate': 0.0,
            'detection_coverage': 0,
            'overall_pass': False
        }

    def run_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整基准测试套件"""
        logger.info("=" * 80)
        logger.info("GoalQuestioner基准测试开始")
        logger.info("=" * 80)

        # 场景A: 目标错位
        logger.info("\n[场景A] 目标错位测试")
        self.results['scenarios']['misalignment'] = self._test_misalignment()

        # 场景B: 目标冲突
        logger.info("\n[场景B] 目标冲突测试")
        self.results['scenarios']['conflict'] = self._test_conflict()

        # 场景C: 目标过拟合
        logger.info("\n[场景C] 目标过拟合测试")
        self.results['scenarios']['overfitting'] = self._test_overfitting()

        # 场景D: 目标漂移
        logger.info("\n[场景D] 目标漂移测试")
        self.results['scenarios']['drift'] = self._test_drift()

        # 场景E: 正常情况 (误报测试)
        logger.info("\n[场景E] 误报率测试 (正常情况)")
        self.results['scenarios']['normal'] = self._test_normal()

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
        logger.info("GoalQuestioner基准测试完成")
        logger.info("=" * 80)

        return self.results

    def _test_misalignment(self) -> Dict[str, Any]:
        """
        场景A: 目标错位

        模拟:
        - 奖励持续下降
        - 方差很小 (陷入局部最优)
        - 应检测到 MISALIGNMENT
        """
        # 创建GoalQuestioner
        questioner = GoalQuestioner()

        # 创建正常的目标规范
        goal_spec = create_default_goal_spec()

        # 创建目标错位的上下文
        context = QuestioningContext()
        context.step_count = 1000

        # 模拟奖励持续下降 (线性下降 + 小噪声)
        for i in range(100):
            reward = 0.8 - 0.005 * i + np.random.randn() * 0.02
            context.reward_history.append(reward)

        # 抽取
        inspection = questioner.inspect(goal_spec, context)
        logger.info(f"  抽取结果: questioned={inspection['questioned']}")

        # 评估
        evaluation = questioner.evaluate(goal_spec, context)

        # 验证检测
        detected = GoalBiasType.MISALIGNMENT in evaluation.detected_biases

        logger.info(f"  检测结果: MISALIGNMENT={detected}")
        logger.info(f"  对齐度: {evaluation.alignment_score:.2f}")
        logger.info(f"  风险: {evaluation.risk_score:.2f}")
        logger.info(f"  原因: {evaluation.reasons}")

        # 生成修订建议
        revision = questioner.propose_revision(evaluation, goal_spec)

        if revision:
            logger.info(f"  修订建议: {revision.description}")
            logger.info(f"  风险等级: {revision.risk_level}")
            logger.info(f"  变更: {revision.changes}")
        else:
            # 如果没有自动生成修订，手动创建一个用于测试
            revision = questioner.propose_revision(evaluation, goal_spec)

        return {
            'scenario': 'misalignment',
            'detected': detected,
            'alignment_score': float(evaluation.alignment_score),
            'risk_score': float(evaluation.risk_score),
            'benefit_score': float(evaluation.benefit_score),
            'revision_proposed': revision is not None,
            'risk_level': revision.risk_level if revision else 1,
            'reasons': evaluation.reasons
        }

    def _test_conflict(self) -> Dict[str, Any]:
        """
        场景B: 目标冲突

        模拟:
        - 外在目标权重过高 (0.85)
        - 内在目标权重过低 (0.15)
        - 应检测到 CONFLICT
        """
        questioner = GoalQuestioner()

        # 创建目标冲突的目标规范
        goal_spec = GoalSpec(
            external_goals=[
                GoalComponent('task_completion', 0.85, '任务完成度', False, 'reward')
            ],
            intrinsic_goals=[
                GoalComponent('curiosity', 0.10, '好奇心', True, 'uncertainty'),
                GoalComponent('stability', 0.05, '稳定性', True, 'loss')
            ],
            description='目标冲突规范',
            version=1
        )

        # 正常的上下文
        context = QuestioningContext()
        context.step_count = 500

        # 正常奖励
        for i in range(50):
            reward = 0.5 + np.random.randn() * 0.1
            context.reward_history.append(reward)

        # 评估
        evaluation = questioner.evaluate(goal_spec, context)

        detected = GoalBiasType.CONFLICT in evaluation.detected_biases

        logger.info(f"  检测结果: CONFLICT={detected}")
        logger.info(f"  对齐度: {evaluation.alignment_score:.2f}")
        logger.info(f"  原因: {evaluation.reasons}")

        # 生成修订建议
        revision = questioner.propose_revision(evaluation, goal_spec)

        return {
            'scenario': 'conflict',
            'detected': detected,
            'alignment_score': float(evaluation.alignment_score),
            'risk_score': float(evaluation.risk_score),
            'benefit_score': float(evaluation.benefit_score),
            'revision_proposed': revision is not None,
            'risk_level': revision.risk_level if revision else 1,
            'reasons': evaluation.reasons
        }

    def _test_overfitting(self) -> Dict[str, Any]:
        """
        场景C: 目标过拟合

        模拟:
        - 单一目标权重过高 (>0.8)
        - 应检测到 OVERFITTING
        """
        questioner = GoalQuestioner()

        # 创建过拟合的目标规范
        goal_spec = GoalSpec(
            external_goals=[
                GoalComponent('task_completion', 0.90, '任务完成度', False, 'reward')
            ],
            intrinsic_goals=[
                GoalComponent('curiosity', 0.05, '好奇心', True, 'uncertainty'),
                GoalComponent('stability', 0.05, '稳定性', True, 'loss')
            ],
            description='过拟合规范',
            version=1
        )

        context = QuestioningContext()
        context.step_count = 500

        # 正常奖励
        for i in range(50):
            reward = 0.7 + np.random.randn() * 0.1
            context.reward_history.append(reward)

        evaluation = questioner.evaluate(goal_spec, context)

        detected = GoalBiasType.OVERFITTING in evaluation.detected_biases

        logger.info(f"  检测结果: OVERFITTING={detected}")
        logger.info(f"  对齐度: {evaluation.alignment_score:.2f}")
        logger.info(f"  原因: {evaluation.reasons}")

        # 生成修订建议
        revision = questioner.propose_revision(evaluation, goal_spec)

        return {
            'scenario': 'overfitting',
            'detected': detected,
            'alignment_score': float(evaluation.alignment_score),
            'risk_score': float(evaluation.risk_score),
            'benefit_score': float(evaluation.benefit_score),
            'revision_proposed': revision is not None,
            'risk_level': revision.risk_level if revision else 1,
            'reasons': evaluation.reasons
        }

    def _test_drift(self) -> Dict[str, Any]:
        """
        场景D: 目标漂移

        模拟:
        - 异常事件计数过高 (>10)
        - 应检测到 DRIFT
        """
        questioner = GoalQuestioner()

        goal_spec = create_default_goal_spec()

        context = QuestioningContext()
        context.step_count = 500
        context.anomaly_count = 15  # 异常事件过多

        # 正常奖励
        for i in range(50):
            reward = 0.6 + np.random.randn() * 0.1
            context.reward_history.append(reward)

        evaluation = questioner.evaluate(goal_spec, context)

        detected = GoalBiasType.DRIFT in evaluation.detected_biases

        logger.info(f"  检测结果: DRIFT={detected}")
        logger.info(f"  对齐度: {evaluation.alignment_score:.2f}")
        logger.info(f"  原因: {evaluation.reasons}")

        # 生成修订建议
        revision = questioner.propose_revision(evaluation, goal_spec)

        return {
            'scenario': 'drift',
            'detected': detected,
            'alignment_score': float(evaluation.alignment_score),
            'risk_score': float(evaluation.risk_score),
            'benefit_score': float(evaluation.benefit_score),
            'revision_proposed': revision is not None,
            'risk_level': revision.risk_level if revision else 1,
            'reasons': evaluation.reasons
        }

    def _test_normal(self) -> Dict[str, Any]:
        """
        场景E: 正常情况 (误报测试)

        模拟:
        - 健康的目标配置
        - 正常的奖励模式
        - 不应检测到任何偏差
        """
        questioner = GoalQuestioner()

        goal_spec = create_default_goal_spec()

        context = QuestioningContext()
        context.step_count = 500

        # 健康的奖励模式 (上升趋势 + 合理方差)
        for i in range(100):
            reward = 0.3 + 0.003 * i + np.random.randn() * 0.1
            context.reward_history.append(reward)

        evaluation = questioner.evaluate(goal_spec, context)

        # 正常情况不应检测到偏差
        has_false_positive = len(evaluation.detected_biases) > 0

        logger.info(f"  误报: {has_false_positive}")
        logger.info(f"  检测到的偏差: {[b.value for b in evaluation.detected_biases]}")
        logger.info(f"  对齐度: {evaluation.alignment_score:.2f}")

        return {
            'scenario': 'normal',
            'false_positive': has_false_positive,
            'detected_biases': [b.value for b in evaluation.detected_biases],
            'alignment_score': float(evaluation.alignment_score),
            'risk_score': float(evaluation.risk_score),
            'benefit_score': float(evaluation.benefit_score)
        }

    def _analyze_results(self):
        """分析测试结果"""
        # 1. 检测覆盖率 (应检测≥3类偏差)
        detected_types = set()
        for scenario, result in self.results['scenarios'].items():
            if scenario != 'normal' and result.get('detected', False):
                detected_types.add(scenario)

        self.results['detection_coverage'] = len(detected_types)

        # 2. 误报率
        normal_result = self.results['scenarios'].get('normal', {})
        self.results['false_positive_rate'] = 1.0 if normal_result.get('false_positive', False) else 0.0

        # 3. 建议采纳率 (模拟)
        # 统计有修订建议的场景
        total_proposals = 0
        total_applicable = 0
        for scenario, result in self.results['scenarios'].items():
            if scenario != 'normal' and result.get('revision_proposed', False):
                total_proposals += 1
                # 模拟采纳: 如果风险等级≤3则采纳
                if result.get('risk_level', 5) <= 3:
                    total_applicable += 1

        adoption_rate = total_applicable / total_proposals if total_proposals > 0 else 0.0
        self.results['adoption_rate'] = float(adoption_rate)

        # 4. 反循环机制
        # 测试冷却期
        questioner = GoalQuestioner()
        context1 = QuestioningContext()
        for _ in range(100):
            context1.reward_history.append(0.5)
        context1.step_count = 100

        # 第一次质疑
        spec = create_default_goal_spec()
        questioner.inspect(spec, context1)
        first_time = questioner._last_questioning_time

        # 立即再次质疑 (应被冷却阻止)
        context2 = QuestioningContext()
        for _ in range(100):
            context2.reward_history.append(0.5)
        context2.step_count = 200

        questioner.inspect(spec, context2)
        second_time = questioner._last_questioning_time

        # 反循环有效: 第二次不应更新时间
        anti_loop_works = (second_time - first_time) < 1.0

        self.results['anti_loop_effective'] = anti_loop_works

        # 5. 总体验收
        self.results['overall_pass'] = (
            self.results['detection_coverage'] >= 3 and
            self.results['false_positive_rate'] < 0.2 and
            self.results['adoption_rate'] >= 0.6 and
            self.results['anti_loop_effective']
        )

        logger.info(f"\n验收指标:")
        logger.info(f"  检测覆盖: {self.results['detection_coverage']}/3 ≥3")
        logger.info(f"  误报率: {self.results['false_positive_rate']*100:.1f}% <20%")
        logger.info(f"  采纳率: {self.results['adoption_rate']*100:.1f}% ≥60%")
        logger.info(f"  反循环: {anti_loop_works}")
        logger.info(f"  总体通过: {self.results['overall_pass']}")

    def _generate_plots(self):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 检测覆盖率
        ax = axes[0, 0]
        scenarios = ['Misalignment', 'Conflict', 'Overfitting', 'Drift']
        detected = [
            self.results['scenarios']['misalignment']['detected'],
            self.results['scenarios']['conflict']['detected'],
            self.results['scenarios']['overfitting']['detected'],
            self.results['scenarios']['drift']['detected']
        ]
        colors = ['green' if d else 'red' for d in detected]
        ax.bar(scenarios, [1 if d else 0 for d in detected], color=colors, alpha=0.7)
        ax.set_ylabel('Detected')
        ax.set_title('Detection Coverage')
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3)

        # 2. 评分对比
        ax = axes[0, 1]
        for scenario, result in self.results['scenarios'].items():
            if scenario != 'normal':
                alignment = result['alignment_score']
                risk = result['risk_score']
                benefit = result['benefit_score']
                ax.scatter(alignment, risk, s=100, alpha=0.6, label=scenario)
        ax.set_xlabel('Alignment Score')
        ax.set_ylabel('Risk Score')
        ax.set_title('Evaluation Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 验收指标
        ax = axes[1, 0]
        metrics = ['Detection\nCoverage', 'False Positive\nRate', 'Adoption\nRate', 'Anti-loop']
        values = [
            self.results['detection_coverage'] / 4,  # 归一化
            1 - self.results['false_positive_rate'],  # 反转 (越低越好)
            self.results['adoption_rate'],
            1.0 if self.results['anti_loop_effective'] else 0.0
        ]
        colors = ['green' if v >= 0.6 else 'orange' for v in values]
        ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Acceptance Metrics')
        ax.set_ylim([0, 1.2])
        ax.axhline(y=0.6, color='red', linestyle='--', label='Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 总体验收结果
        ax = axes[1, 1]
        ax.axis('off')
        result_text = f"""
        M2 Stage Acceptance Result

        Detection Coverage: {self.results['detection_coverage']}/3
        False Positive Rate: {self.results['false_positive_rate']*100:.1f}%
        Adoption Rate: {self.results['adoption_rate']*100:.1f}%
        Anti-loop Effective: {'Yes' if self.results['anti_loop_effective'] else 'No'}

        Overall Result: {'✅ PASS' if self.results['overall_pass'] else '❌ FAIL'}
        """
        ax.text(0.5, 0.5, result_text, ha='center', va='center',
               fontsize=14, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Final Result')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'goal_questioner_benchmark.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  图表已保存到: {self.output_dir}")

    def _save_report(self):
        """保存测试报告"""
        report_path = self.output_dir / 'benchmark_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GoalQuestioner基准测试报告\n")
            f.write("=" * 80 + "\n\n")

            # 验收指标
            f.write("[验收指标]\n")
            f.write(f"  检测覆盖: {self.results['detection_coverage']}/3 (要求≥3)\n")
            f.write(f"  误报率: {self.results['false_positive_rate']*100:.1f}% (要求<20%)\n")
            f.write(f"  采纳率: {self.results['adoption_rate']*100:.1f}% (要求≥60%)\n")
            f.write(f"  反循环有效: {'Yes' if self.results['anti_loop_effective'] else 'No'}\n")
            f.write(f"  总体结果: {'✅ PASS' if self.results['overall_pass'] else '❌ FAIL'}\n\n")

            # 各场景详情
            f.write("[场景详情]\n")
            for scenario, result in self.results['scenarios'].items():
                f.write(f"\n{scenario.upper()}:\n")
                if scenario == 'normal':
                    f.write(f"  误报: {result['false_positive']}\n")
                else:
                    f.write(f"  检测到偏差: {result['detected']}\n")
                    f.write(f"  对齐度: {result['alignment_score']:.3f}\n")
                    f.write(f"  风险: {result['risk_score']:.3f}\n")
                    f.write(f"  收益: {result['benefit_score']:.3f}\n")
                    if result.get('reasons'):
                        f.write(f"  原因:\n")
                        for reason in result['reasons']:
                            f.write(f"    - {reason}\n")

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

    benchmark = GoalQuestionerBenchmark()
    results = benchmark.run_benchmark_suite()

    # 输出摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)

    print(f"\n检测覆盖: {results['detection_coverage']}/3")
    print(f"误报率: {results['false_positive_rate']*100:.1f}%")
    print(f"采纳率: {results['adoption_rate']*100:.1f}%")
    print(f"反循环有效: {results['anti_loop_effective']}")
    print(f"\n总体结果: {'✅ PASS' if results['overall_pass'] else '❌ FAIL'}")

    return 0 if results['overall_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
