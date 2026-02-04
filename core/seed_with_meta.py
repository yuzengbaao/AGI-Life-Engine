"""
TheSeedWithMeta - 集成MetaLearner的TheSeed扩展

功能边界:
- 保持TheSeed原有功能不变
- 添加元学习能力 (自动调整超参数)
- 提供指标收集接口

拓扑连接:
- TheSeedWithMeta 继承/组合 TheSeed
- 添加 _meta_learner: MetaLearner实例
- 添加 _uncertainty_buffer: 收集不确定性历史
- 添加 _last_wm_loss, _last_vf_loss: 记录最近loss

设计原则:
1. 不修改TheSeed核心逻辑 (开闭原则)
2. 元学习是可选功能 (向后兼容)
3. 指标收集对性能影响最小
"""

import numpy as np
import logging
from typing import Optional, Any, List
import time

from core.seed import TheSeed, Experience
from core.meta_learner import MetaLearner, StepMetrics, collect_seed_metrics, apply_meta_parameters_to_seed

logger = logging.getLogger(__name__)


class TheSeedWithMeta(TheSeed):
    """
    集成元学习能力的TheSeed扩展

    新增功能:
    1. 自动收集训练指标
    2. 自动调用MetaLearner调整超参数
    3. 记录不确定性历史用于趋势分析

    使用方式:
    ```python
    seed = TheSeedWithMeta(state_dim=64, action_dim=4, enable_meta=True)
    # 正常使用seed
    state = seed.perceive(raw_input)
    action = seed.act(state)
    # 元学习自动在后台运行
    ```
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 enable_meta: bool = True,
                 meta_strategy: str = 'rule_based',
                 event_bus: Any = None):
        """
        初始化TheSeedWithMeta

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            enable_meta: 是否启用元学习
            meta_strategy: 元学习策略 ('rule_based', 'bandit', 'meta_gradient')
            event_bus: 事件总线 (可选)
        """
        # 调用父类初始化
        super().__init__(state_dim, action_dim)

        # 元学习配置
        self.enable_meta = enable_meta
        self._meta_learner: Optional[MetaLearner] = None

        # 指标收集
        self._uncertainty_buffer: List[float] = []
        self._uncertainty_buffer_size = 100
        self._last_wm_loss = 0.0
        self._last_vf_loss = 0.0
        self._last_update_time = 0.0
        self._step_count = 0

        # 启用元学习
        if enable_meta:
            from core.meta_learner import MetaStrategy
            strategy = MetaStrategy(meta_strategy) if isinstance(meta_strategy, str) else meta_strategy
            self._meta_learner = MetaLearner(
                event_bus=event_bus,
                initial_strategy=strategy
            )
            logger.info(f"🧠 TheSeedWithMeta initialized with meta-learning (strategy={meta_strategy})")

    def predict(self, state: np.ndarray, action: int) -> tuple:
        """
        重写predict方法以收集不确定性

        Returns:
            (predicted_next_state, uncertainty)
        """
        # 调用父类predict
        prediction, uncertainty = super().predict(state, action)

        # 收集不确定性历史
        if self.enable_meta:
            self._uncertainty_buffer.append(uncertainty)
            if len(self._uncertainty_buffer) > self._uncertainty_buffer_size:
                self._uncertainty_buffer.pop(0)

        return prediction, uncertainty

    def learn(self, experience: Experience) -> float:
        """
        重写learn方法以收集loss并触发元学习

        Returns:
            平均loss
        """
        # 调用父类learn (这会更新world_model和value_network)
        # 注意: 父类learn没有返回值,我们需要手动计算loss

        # 1. 存储经验
        self.memory.push(experience)

        # 2. Dream (从记忆中采样训练)
        experiences = self.memory.sample(self.batch_size)

        total_loss = 0.0
        loss_count = 0

        for exp in experiences:
            # World Model训练
            action_vec = np.zeros(self.action_dim)
            if 0 <= exp.action < self.action_dim:
                action_vec[exp.action] = 1.0

            input_vec = np.concatenate([exp.state, action_vec])
            target = exp.next_state

            # Forward
            prediction = self.world_model.forward(input_vec)
            wm_loss = self.world_model.backward(target)

            # Value Network训练
            vf_loss = 0.0
            if loss_count < len(experiences) // 2:  # 只在一半样本上训练value network
                vf_loss = self.value_network.backward(np.array([exp.reward]))

            total_loss += (wm_loss + vf_loss)
            loss_count += 1

        avg_loss = total_loss / max(loss_count, 1)

        # 记录loss
        if self.enable_meta:
            self._last_wm_loss = avg_loss  # 简化: 使用平均loss
            self._last_vf_loss = avg_loss * 0.5  # 估算

        # 触发元学习检查 (每10步)
        if self.enable_meta:
            self._step_count += 1
            if self._step_count % 10 == 0:
                self._check_and_apply_meta_learning()

        return avg_loss

    def _check_and_apply_meta_learning(self) -> None:
        """检查并应用元学习"""
        if not self.enable_meta or self._meta_learner is None:
            return

        # 收集当前指标
        metrics = collect_seed_metrics(self)

        # 观察指标
        self._meta_learner.observe(metrics)

        # 提出更新建议
        update = self._meta_learner.propose_update(mode='auto')

        # 应用更新
        if update is not None:
            result = self._meta_learner.apply(update, mode='auto')

            if result.get('applied'):
                # 将新参数应用到TheSeed
                apply_meta_parameters_to_seed(self, self._meta_learner.get_current_parameters())

    def get_meta_statistics(self) -> dict:
        """获取元学习统计信息"""
        if not self.enable_meta or self._meta_learner is None:
            return {
                'meta_enabled': False,
                'learning_rate': self.learning_rate,
                'curiosity_weight': self.curiosity_weight
            }

        stats = self._meta_learner.get_statistics()
        stats['meta_enabled'] = True
        stats['step_count'] = self._step_count

        return stats

    def force_meta_update(self) -> Optional[dict]:
        """
        强制触发一次元参数更新 (用于测试)

        Returns:
            更新结果字典或None
        """
        if not self.enable_meta or self._meta_learner is None:
            logger.warning("[TheSeedWithMeta] 元学习未启用")
            return None

        # 收集当前指标
        metrics = collect_seed_metrics(self)
        self._meta_learner.observe(metrics)

        # 提出并应用更新
        update = self._meta_learner.propose_update(mode='auto')
        if update is not None:
            result = self._meta_learner.apply(update, mode='auto')
            if result.get('applied'):
                apply_meta_parameters_to_seed(self, self._meta_learner.get_current_parameters())
            return result

        return None

    def set_meta_strategy(self, strategy: str) -> None:
        """设置元学习策略"""
        if not self.enable_meta or self._meta_learner is None:
            logger.warning("[TheSeedWithMeta] 元学习未启用")
            return

        from core.meta_learner import MetaStrategy
        self._meta_learner.set_strategy(MetaStrategy(strategy))
        logger.info(f"[TheSeedWithMeta] 切换元学习策略: {strategy}")


# ============================================================================
# 工厂函数: 便捷创建带元学习的TheSeed
# ============================================================================

def create_seed_with_meta(state_dim: int,
                          action_dim: int,
                          **kwargs) -> TheSeedWithMeta:
    """
    创建带元学习能力的TheSeed

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        **kwargs: 传递给TheSeedWithMeta的其他参数

    Returns:
        TheSeedWithMeta实例
    """
    return TheSeedWithMeta(
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs
    )


# ============================================================================
# 向后兼容: 如果需要,可以将普通TheSeed包装为TheSeedWithMeta
# ============================================================================

def wrap_seed_with_meta(seed: TheSeed,
                        enable_meta: bool = True,
                        event_bus: Any = None) -> TheSeedWithMeta:
    """
    将现有TheSeed包装为TheSeedWithMeta

    注意: 这会创建一个新实例,复制原seed的状态
    """
    wrapped = TheSeedWithMeta(
        state_dim=seed.state_dim,
        action_dim=seed.action_dim,
        enable_meta=enable_meta,
        event_bus=event_bus
    )

    # 复制关键状态
    wrapped.learning_rate = seed.learning_rate
    wrapped.curiosity_weight = seed.curiosity_weight

    # 复制网络权重 (如果结构相同)
    try:
        wrapped.world_model = seed.world_model
        wrapped.value_network = seed.value_network
        wrapped.memory = seed.memory
    except Exception as e:
        logger.warning(f"[wrap_seed_with_meta] 状态复制失败: {e}")

    return wrapped
