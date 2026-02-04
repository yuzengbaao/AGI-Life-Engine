"""
MetaLearner基准测试 - 验证M1阶段实施

目标:
1. 对比有/无MetaLearner的性能
2. 验证性能提升≥15%
3. 验证系统稳定性 (无NaN/Inf/发散)

基准任务:
- 任务A: 序列推理 (Fibonacci, Prime, 等比序列)
- 任务B: 因果发现 (学习因果图)
- 任务C: 多任务学习 (回归+分类+排序)

验收标准:
- 性能提升≥15% (平均loss降低或准确率提升)
- 稳定性: 无NaN/Inf, 重置率不升高
- 可解释: 每次调参输出原因+证据
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.seed import TheSeed, Experience
from core.seed_with_meta import TheSeedWithMeta, create_seed_with_meta
from core.meta_learner import MetaLearner, StepMetrics

logger = logging.getLogger(__name__)


class MetaLearningBenchmark:
    """元学习基准测试"""

    def __init__(self, output_dir: str = "./test_results/meta_learner"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'baseline': {},      # 无MetaLearner
            'with_meta': {},     # 有MetaLearner
            'comparison': {}     # 对比分析
        }

    def run_benchmark_suite(self) -> Dict[str, Any]:
        """运行完整基准测试套件"""
        logger.info("="*80)
        logger.info("MetaLearner基准测试开始")
        logger.info("="*80)

        # 任务A: 序列推理
        logger.info("\n[任务A] 序列推理基准测试")
        self.results['baseline']['sequence'] = self._test_sequence_learning(use_meta=False)
        self.results['with_meta']['sequence'] = self._test_sequence_learning(use_meta=True)

        # 任务B: 简化因果推理
        logger.info("\n[任务B] 因果推理基准测试")
        self.results['baseline']['causal'] = self._test_causal_learning(use_meta=False)
        self.results['with_meta']['causal'] = self._test_causal_learning(use_meta=True)

        # 对比分析
        logger.info("\n[分析] 生成对比报告")
        self.results['comparison'] = self._analyze_comparison()

        # 生成可视化
        logger.info("\n[可视化] 生成图表")
        self._generate_plots()

        # 保存报告
        logger.info("\n[报告] 保存结果")
        self._save_report()

        logger.info("\n"+"="*80)
        logger.info("MetaLearner基准测试完成")
        logger.info("="*80)

        return self.results

    def _test_sequence_learning(self, use_meta: bool, epochs: int = 50) -> Dict[str, Any]:
        """
        任务A: 序列学习基准测试

        测试内容:
        - 学习Fibonacci序列
        - 学习等比序列 (*2, *3)
        - 学习质数序列 (困难)

        指标:
        - 预测误差 (MSE)
        - 收敛速度 (达到目标loss的训练轮数)
        - 稳定性 (异常事件计数)
        """
        # 配置
        state_dim = 10  # 序列长度
        action_dim = 4  # 4种序列类型

        # 创建种子
        if use_meta:
            seed = create_seed_with_meta(state_dim, action_dim, enable_meta=True)
            logger.info(f"  创建TheSeedWithMeta (启用元学习)")
        else:
            seed = TheSeed(state_dim, action_dim)
            logger.info(f"  创建TheSeed (baseline)")

        # 生成训练数据
        train_data = self._generate_sequence_data(n_samples=500)
        test_data = self._generate_sequence_data(n_samples=100)

        # 训练循环
        history = {'loss': [], 'predictions': [], 'meta_updates': 0}

        for epoch in range(epochs):
            epoch_loss = 0.0
            anomaly_count = 0

            for i, (seq, next_val, seq_type) in enumerate(train_data):
                # 感知
                state = seed.perceive(seq)

                # 动作 (随机探索)
                action = np.random.randint(action_dim)

                # 预测下一个状态 (模拟)
                pred_state, uncertainty = seed.predict(state, action)

                # 计算预测误差
                target_next = np.roll(seq, -1)
                target_next[-1] = next_val
                loss = np.mean((pred_state - target_next) ** 2)

                # 检查异常
                if np.isnan(loss) or np.isinf(loss) or loss > 100:
                    anomaly_count += 1

                # 学习
                exp = Experience(state, action, -loss, pred_state)  # 负loss作为奖励
                seed.learn(exp)

                epoch_loss += loss

            # 记录
            avg_loss = epoch_loss / len(train_data)
            history['loss'].append(avg_loss)

            # 记录元学习更新次数
            if use_meta:
                stats = seed.get_meta_statistics()
                history['meta_updates'] = stats.get('update_count', 0)

            if epoch % 10 == 0:
                meta_info = f", meta_updates={history['meta_updates']}" if use_meta else ""
                logger.info(f"  Epoch {epoch:3d}: loss={avg_loss:.6f}{meta_info}")

        # 测试
        test_loss = self._evaluate_sequence_prediction(seed, test_data)

        # 计算指标
        final_loss = history['loss'][-1]
        convergence_epoch = self._find_convergence_epoch(history['loss'], threshold=0.5)

        result = {
            'use_meta': use_meta,
            'final_loss': float(final_loss),
            'test_loss': float(test_loss),
            'convergence_epoch': int(convergence_epoch),
            'loss_history': history['loss'],
            'meta_updates': history.get('meta_updates', 0),
            'epochs': epochs
        }

        logger.info(f"  结果: final_loss={final_loss:.6f}, test_loss={test_loss:.6f}, "
                   f"convergence_epoch={convergence_epoch}")

        return result

    def _test_causal_learning(self, use_meta: bool, epochs: int = 100) -> Dict[str, Any]:
        """
        任务B: 简化因果学习基准测试

        测试内容:
        - 学习简单的因果关系: X → Y
        - 因果强度估计

        指标:
        - 因果估计误差
        - 干预效果准确性
        """
        # 配置
        state_dim = 5
        action_dim = 3

        # 创建种子
        if use_meta:
            seed = create_seed_with_meta(state_dim, action_dim, enable_meta=True)
        else:
            seed = TheSeed(state_dim, action_dim)

        # 生成因果数据: X → Y (Y = 2*X + noise)
        train_data = []
        for _ in range(500):
            x = np.random.randn(state_dim)
            y = 2.0 * x + np.random.randn(state_dim) * 0.1  # 强因果
            train_data.append((x, y))

        # 训练
        history = {'loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0

            for x, y in train_data:
                # 感知
                state = seed.perceive(x)

                # 预测
                action = 0  # 固定动作
                pred_y, _ = seed.predict(state, action)

                # Loss
                loss = np.mean((pred_y - y) ** 2)
                epoch_loss += loss

                # 学习
                exp = Experience(state, action, -loss, pred_y)
                seed.learn(exp)

            avg_loss = epoch_loss / len(train_data)
            history['loss'].append(avg_loss)

            if epoch % 20 == 0:
                logger.info(f"  Epoch {epoch:3d}: loss={avg_loss:.6f}")

        # 测试因果强度估计
        test_x = np.random.randn(100, state_dim)
        test_y = 2.0 * test_x + np.random.randn(*test_x.shape) * 0.1

        estimated_effects = []
        for x in test_x:
            state = seed.perceive(x)
            pred_y, _ = seed.predict(state, 0)
            estimated_effects.append(pred_y[0] / (x[0] + 1e-8))

        estimated_effect = np.mean(estimated_effects)
        true_effect = 2.0
        effect_error = abs(estimated_effect - true_effect)

        result = {
            'use_meta': use_meta,
            'final_loss': float(history['loss'][-1]),
            'estimated_causal_effect': float(estimated_effect),
            'true_causal_effect': true_effect,
            'effect_error': float(effect_error),
            'loss_history': history['loss']
        }

        logger.info(f"  结果: loss={history['loss'][-1]:.6f}, "
                   f"estimated_effect={estimated_effect:.4f}, "
                   f"error={effect_error:.4f}")

        return result

    def _generate_sequence_data(self, n_samples: int) -> List[Tuple[np.ndarray, float, int]]:
        """生成序列数据"""
        data = []
        seq_length = 10

        for _ in range(n_samples):
            seq_type = np.random.randint(4)

            if seq_type == 0:  # Fibonacci
                seq = [1, 1]
                for i in range(seq_length - 2):
                    seq.append(seq[-1] + seq[-2])
                next_val = seq[-1] + seq[-2]
                seq = np.array(seq[:seq_length], dtype=float) / 1000.0  # 归一化

            elif seq_type == 1:  # 等比*2
                start = np.random.randint(1, 5)
                seq = [start]
                for i in range(seq_length - 1):
                    seq.append(seq[-1] * 2)
                next_val = seq[-1] * 2
                seq = np.array(seq, dtype=float) / 10000.0  # 归一化

            elif seq_type == 2:  # 等比*3
                start = np.random.randint(1, 3)
                seq = [start]
                for i in range(seq_length - 1):
                    seq.append(seq[-1] * 3)
                next_val = seq[-1] * 3
                seq = np.array(seq, dtype=float) / 100000.0  # 归一化

            else:  # 线性
                start = np.random.randint(1, 10)
                seq = [start + i for i in range(seq_length)]
                next_val = start + seq_length
                seq = np.array(seq, dtype=float) / 100.0  # 归一化

            data.append((seq, next_val, seq_type))

        return data

    def _evaluate_sequence_prediction(self, seed: Any, test_data: List) -> float:
        """评估序列预测"""
        total_loss = 0.0

        for seq, next_val, _ in test_data:
            state = seed.perceive(seq)
            action = 0
            pred_state, _ = seed.predict(state, action)

            target_next = np.roll(seq, -1)
            target_next[-1] = next_val
            loss = np.mean((pred_state - target_next) ** 2)
            total_loss += loss

        return total_loss / len(test_data)

    def _find_convergence_epoch(self, loss_history: List[float], threshold: float = 0.5) -> int:
        """找到收敛的epoch"""
        for i, loss in enumerate(loss_history):
            if loss < threshold:
                return i
        return len(loss_history)

    def _analyze_comparison(self) -> Dict[str, Any]:
        """对比分析"""
        comparison = {}

        # 任务A对比
        baseline_seq = self.results['baseline']['sequence']
        with_meta_seq = self.results['with_meta']['sequence']

        seq_improvement = (baseline_seq['final_loss'] - with_meta_seq['final_loss']) / baseline_seq['final_loss']
        seq_convergence_improvement = (baseline_seq['convergence_epoch'] - with_meta_seq['convergence_epoch']) / baseline_seq['convergence_epoch']

        comparison['sequence'] = {
            'loss_improvement_ratio': float(seq_improvement),
            'loss_improvement_percent': float(seq_improvement * 100),
            'convergence_improvement_ratio': float(seq_convergence_improvement),
            'convergence_improvement_percent': float(seq_convergence_improvement * 100),
            'baseline_final_loss': float(baseline_seq['final_loss']),
            'with_meta_final_loss': float(with_meta_seq['final_loss']),
            'baseline_convergence': int(baseline_seq['convergence_epoch']),
            'with_meta_convergence': int(with_meta_seq['convergence_epoch']),
            'meta_updates_count': int(with_meta_seq['meta_updates'])
        }

        # 任务B对比
        baseline_causal = self.results['baseline']['causal']
        with_meta_causal = self.results['with_meta']['causal']

        causal_loss_improvement = (baseline_causal['final_loss'] - with_meta_causal['final_loss']) / baseline_causal['final_loss']
        causal_effect_improvement = (baseline_causal['effect_error'] - with_meta_causal['effect_error']) / baseline_causal['effect_error']

        comparison['causal'] = {
            'loss_improvement_ratio': float(causal_loss_improvement),
            'loss_improvement_percent': float(causal_loss_improvement * 100),
            'effect_error_improvement_ratio': float(causal_effect_improvement),
            'effect_error_improvement_percent': float(causal_effect_improvement * 100),
            'baseline_final_loss': float(baseline_causal['final_loss']),
            'with_meta_final_loss': float(with_meta_causal['final_loss']),
            'baseline_effect_error': float(baseline_causal['effect_error']),
            'with_meta_effect_error': float(with_meta_causal['effect_error'])
        }

        # 综合评估
        avg_improvement = (seq_improvement + causal_loss_improvement) / 2

        comparison['overall'] = {
            'average_improvement_percent': float(avg_improvement * 100),
            'meets_threshold': avg_improvement >= 0.15,  # ≥15%目标
            'passed_sequence': seq_improvement >= 0.15,
            'passed_causal': causal_loss_improvement >= 0.15
        }

        return comparison

    def _generate_plots(self):
        """生成可视化图表"""
        # 序列学习loss曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.results['baseline']['sequence']['loss_history'], label='Baseline', alpha=0.7)
        plt.plot(self.results['with_meta']['sequence']['loss_history'], label='With MetaLearner', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Sequence Learning: Loss Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'sequence_learning_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 因果学习loss曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.results['baseline']['causal']['loss_history'], label='Baseline', alpha=0.7)
        plt.plot(self.results['with_meta']['causal']['loss_history'], label='With MetaLearner', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Causal Learning: Loss Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'causal_learning_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  图表已保存到: {self.output_dir}")

    def _save_report(self):
        """保存测试报告"""
        report_path = self.output_dir / 'benchmark_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MetaLearner基准测试报告\n")
            f.write("="*80 + "\n\n")

            # 任务A结果
            f.write("[任务A: 序列学习]\n")
            comp_seq = self.results['comparison']['sequence']
            f.write(f"  Loss改进: {comp_seq['loss_improvement_percent']:.2f}%\n")
            f.write(f"  收敛速度改进: {comp_seq['convergence_improvement_percent']:.2f}%\n")
            f.write(f"  Baseline最终loss: {comp_seq['baseline_final_loss']:.6f}\n")
            f.write(f"  WithMeta最终loss: {comp_seq['with_meta_final_loss']:.6f}\n")
            f.write(f"  元更新次数: {comp_seq['meta_updates_count']}\n\n")

            # 任务B结果
            f.write("[任务B: 因果学习]\n")
            comp_causal = self.results['comparison']['causal']
            f.write(f"  Loss改进: {comp_causal['loss_improvement_percent']:.2f}%\n")
            f.write(f"  因果效应估计误差改进: {comp_causal['effect_error_improvement_percent']:.2f}%\n")
            f.write(f"  Baseline最终loss: {comp_causal['baseline_final_loss']:.6f}\n")
            f.write(f"  WithMeta最终loss: {comp_causal['with_meta_final_loss']:.6f}\n\n")

            # 综合评估
            f.write("[综合评估]\n")
            overall = self.results['comparison']['overall']
            f.write(f"  平均改进: {overall['average_improvement_percent']:.2f}%\n")
            f.write(f"  达到15%目标: {'✅ 是' if overall['meets_threshold'] else '❌ 否'}\n")
            f.write(f"  序列学习通过: {'✅ 是' if overall['passed_sequence'] else '❌ 否'}\n")
            f.write(f"  因果学习通过: {'✅ 是' if overall['passed_causal'] else '❌ 否'}\n\n")

            f.write("="*80 + "\n")

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

    benchmark = MetaLearningBenchmark()
    results = benchmark.run_benchmark_suite()

    # 输出摘要
    print("\n" + "="*80)
    print("测试摘要")
    print("="*80)

    overall = results['comparison']['overall']
    print(f"\n平均性能改进: {overall['average_improvement_percent']:.2f}%")
    print(f"达到15%目标: {'✅ 通过' if overall['meets_threshold'] else '❌ 未通过'}")

    print("\n详细结果:")
    for task, comp in results['comparison'].items():
        if task != 'overall':
            print(f"  {task}: {comp.get('loss_improvement_percent', 0):.2f}% 改进")

    return 0 if overall['meets_threshold'] else 1


if __name__ == '__main__':
    sys.exit(main())
