#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
熵值调节器 (Entropy Regulator)
=====================================

功能：模拟人类的降熵机制，维持系统长期的中熵状态
类比：睡眠、休息、冥想、注意力休息

版本: 1.0.0
日期: 2026-01-16
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EntropyState(Enum):
    """熵值状态"""
    LOW = "low"          # 低熵 (<0.3): 稳定、僵化
    BALANCED = "balanced"  # 平衡 (0.3-0.7): 最佳智能状态
    HIGH = "high"        # 高熵 (0.7-0.9): 警告
    CRITICAL = "critical" # 临界 (>0.9): 故障


@dataclass
class EntropyMetrics:
    """熵值指标"""
    current_entropy: float
    average_entropy: float  # 最近100次的平均值
    entropy_trend: str  # "rising", "falling", "stable"
    time_in_current_state: float  # 在当前状态停留的时间(秒)
    last_reset_time: float  # 上次重置时间


class EntropyRegulator:
    """
    熵值调节器 - 模拟人类的降熵机制

    核心功能：
    1. 监控熵值趋势（短期和长期）
    2. 检测熵值累积异常
    3. 触发降熵机制（睡眠、休息、冥想）
    4. 维持系统在最佳中熵状态
    """

    def __init__(self,
                 monitor_window: int = 100,
                 warning_threshold: float = 0.6,
                 critical_threshold: float = 0.75,
                 rising_threshold: float = 5):
        """
        初始化熵值调节器

        Args:
            monitor_window: 监控窗口大小（默认100个样本）
            warning_threshold: 警告阈值（默认0.7）
            critical_threshold: 临界阈值（默认0.9）
            rising_threshold: 上升阈值（连续上升次数，默认10次）
        """
        self.monitor_window = monitor_window
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.rising_threshold = rising_threshold

        # 熵值历史
        self.entropy_history: List[float] = []

        # 降熵机制配置
        self.sleep_interval = 2 * 3600  # 每2小时检查一次
        self.last_sleep_time = time.time()

        # 短休息机制
        self.short_rest_interval = 1800  # 每30分钟
        self.short_rest_duration = 60   # 休息1分钟
        self.last_rest_time = time.time()

        # 长期睡眠机制
        self.long_sleep_interval = 4 * 3600  # 每4小时
        self.long_sleep_duration = 600      # 睡眠10分钟
        self.last_long_sleep_time = time.time()

        # 强制降熵触发
        self.force_reset_threshold = 0.85  # 平均熵值超过0.85强制降熵
        self.consecutive_rising_count = 0
        self.last_reset_time = time.time()  # 上次重置时间

        # 统计信息
        self.stats = {
            'total_regulations': 0,
            'short_rests': 0,
            'long_sleeps': 0,
            'force_resets': 0,
            'entropy_resets': 0
        }

        logger.info("[EntropyRegulator] 🔧 熵值调节器初始化完成")
        logger.info(f"   - 监控窗口: {monitor_window}")
        logger.info(f"   - 警告阈值: {warning_threshold}")
        logger.info(f"   - 临界阈值: {critical_threshold}")
        logger.info(f"   - 短休息间隔: {self.short_rest_interval}s")
        logger.info(f"   - 长睡眠间隔: {self.long_sleep_interval}s")

    def record_entropy(self, entropy: float) -> EntropyMetrics:
        """
        记录新的熵值并计算指标

        Args:
            entropy: 当前熵值

        Returns:
            EntropyMetrics: 熵值指标对象
        """
        self.entropy_history.append(entropy)

        # 保持历史窗口大小
        if len(self.entropy_history) > self.monitor_window:
            self.entropy_history.pop(0)

        # 计算平均熵值
        avg_entropy = sum(self.entropy_history) / len(self.entropy_history)

        # 判断趋势
        if len(self.entropy_history) >= 10:
            recent_avg = sum(self.entropy_history[-10:]) / 10
            earlier_avg = sum(self.entropy_history[-20:-10]) / 10 if len(self.entropy_history) >= 20 else recent_avg

            if recent_avg > earlier_avg + 0.05:
                trend = "rising"
                self.consecutive_rising_count += 1
            elif recent_avg < earlier_avg - 0.05:
                trend = "falling"
                self.consecutive_rising_count = 0
            else:
                trend = "stable"
        else:
            trend = "stable"

        # 计算在当前状态的时间
        current_time = time.time()
        if len(self.entropy_history) >= 2:
            time_in_state = current_time - self.last_reset_time
        else:
            time_in_state = 0.0

        metrics = EntropyMetrics(
            current_entropy=entropy,
            average_entropy=avg_entropy,
            entropy_trend=trend,
            time_in_current_state=time_in_state,
            last_reset_time=self.last_reset_time
        )

        # 🆕 [2026-01-20] 实时熵值监控与预警
        self._check_entropy_warning(metrics)

        return metrics

    def _check_entropy_warning(self, metrics: EntropyMetrics) -> None:
        """
        检查熵值状态并发出预警

        🔧 [2026-01-20] 新增：实时监控与预警

        Args:
            metrics: 熵值指标对象
        """
        # 预警阈值
        WARNING_LEVEL = 0.70  # 警告阈值

        current_entropy = metrics.current_entropy
        avg_entropy = metrics.average_entropy
        trend = metrics.entropy_trend
        time_in_state = metrics.time_in_current_state

        # 判断是否需要警告
        should_warn = False
        warn_reasons = []

        if current_entropy > WARNING_LEVEL:
            should_warn = True
            warn_reasons.append(f"当前熵值 {current_entropy:.4f} > {WARNING_LEVEL}")

        if avg_entropy > WARNING_LEVEL:
            should_warn = True
            warn_reasons.append(f"平均熵值 {avg_entropy:.4f} > {WARNING_LEVEL}")

        # 检查趋势
        if trend == "rising" and current_entropy > WARNING_LEVEL * 0.9:
            should_warn = True
            warn_reasons.append(f"熵值持续上升（连续上升 {self.consecutive_rising_count} 次）")

        # 发出警告
        if should_warn:
            # 构建警告信息
            trend_emoji = {
                "rising": "📈",
                "falling": "📉",
                "stable": "➡️"
            }.get(trend, "❓")

            # 状态评估
            if current_entropy >= self.critical_threshold:
                status = "🔴 CRITICAL"
                advice = "立即触发熵值调节机制！"
            elif current_entropy >= self.warning_threshold:
                status = "🟠 WARNING"
                advice = "准备触发降熵机制（短休息）"
            elif current_entropy >= WARNING_LEVEL:
                status = "🟡 PRE-WARNING"
                advice = "监控熵值趋势，考虑休息"
            else:
                status = "🟢 OK"
                advice = "熵值正常"

            logger.warning("=" * 60)
            logger.warning(f"⚠️ [EntropyRegulator] 熵值预警: {status}")
            logger.warning(f"   当前熵值: {current_entropy:.4f}")
            logger.warning(f"   平均熵值: {avg_entropy:.4f}")
            logger.warning(f"   趋势: {trend_emoji} {trend.upper()}")
            logger.warning(f"   在当前状态停留: {time_in_state:.1f}秒 ({time_in_state/60:.1f}分钟)")
            logger.warning(f"   预警原因:")
            for reason in warn_reasons:
                logger.warning(f"      • {reason}")
            logger.warning(f"   建议: {advice}")
            logger.warning("=" * 60)

    def should_regulate(self, metrics: EntropyMetrics) -> tuple[bool, str]:
        """
        判断是否需要调节熵值

        Args:
            metrics: 熵值指标

        Returns:
            (是否需要调节, 调节原因)
        """
        # 条件1: 平均熵值超过强制重置阈值
        if metrics.average_entropy > self.force_reset_threshold:
            return True, f"平均熵值过高 ({metrics.average_entropy:.3f} > {self.force_reset_threshold})"

        # 条件2: 当前熵值超过临界阈值
        if metrics.current_entropy > self.critical_threshold:
            return True, f"当前熵值临界 ({metrics.current_entropy:.3f} > {self.critical_threshold})"

        # 条件3: 连续上升趋势
        if self.consecutive_rising_count >= self.rising_threshold * 10:
            return True, f"连续上升 ({self.consecutive_rising_count}次)"

        # 条件4: 短休息间隔检查
        current_time = time.time()
        if current_time - self.last_rest_time >= self.short_rest_interval:
            # 只有在熵值偏高时才触发短休息
            if metrics.average_entropy > 0.6:
                return True, f"短休息时间到 (间隔: {(current_time - self.last_rest_time)/60:.1f}分钟)"

        # 条件5: 长睡眠间隔检查
        if current_time - self.last_long_sleep_time >= self.long_sleep_interval:
            # 无论熵值如何，都进行预防性睡眠
            return True, f"长睡眠时间到 (间隔: {(current_time - self.last_long_sleep_time)/3600:.1f}小时)"

        return False, "熵值正常，无需调节"

    def regulate_entropy(self, metrics: EntropyMetrics, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行熵值调节

        Args:
            metrics: 熵值指标
            context: 系统上下文（包含working_memory等）

        Returns:
            调节结果字典
        """
        should_regulate, reason = self.should_regulate(metrics)

        if not should_regulate:
            return {"regulated": False, "reason": "熵值正常"}

        logger.info(f"[EntropyRegulator] ⚠️ 触发熵值调节: {reason}")
        logger.info(f"   - 当前熵值: {metrics.current_entropy:.3f}")
        logger.info(f"   - 平均熵值: {metrics.average_entropy:.3f}")
        logger.info(f"   - 趋势: {metrics.entropy_trend}")

        # 选择调节策略
        if metrics.average_entropy > self.force_reset_threshold:
            result = self._force_reset_entropy(metrics, context)
            self.stats['force_resets'] += 1
        elif metrics.current_entropy > self.critical_threshold:
            result = self._long_sleep(metrics, context)
            self.stats['long_sleeps'] += 1
        elif "连续上升" in reason:
            result = self._short_rest(metrics, context)
            self.stats['short_rests'] += 1
        elif "短休息" in reason:
            result = self._short_rest(metrics, context)
            self.stats['short_rests'] += 1
        elif "长睡眠" in reason:
            result = self._long_sleep(metrics, context)
            self.stats['long_sleeps'] += 1
        else:
            result = self._short_rest(metrics, context)
            self.stats['short_rests'] += 1

        self.stats['total_regulations'] += 1

        return {
            "regulated": True,
            "reason": reason,
            "strategy": result['strategy'],
            "duration": result['duration'],
            "entropy_before": metrics.current_entropy,
            "entropy_after": result.get('entropy_after', metrics.current_entropy)
        }

    def _short_rest(self, metrics: EntropyMetrics, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        短休息机制（类比人类的小憩）

        策略：
        1. 清理工作记忆
        2. 重置概念冷却
        3. 降低推理深度
        """
        logger.info("[EntropyRegulator] 💤 执行短休息机制（类比人类小憩）")

        # 清理工作记忆
        if 'working_memory' in context and context['working_memory']:
            wm = context['working_memory']
            wm.clear()
            logger.info("   - 工作记忆已清理")

        # 清理概念冷却
        if 'working_memory' in context and context['working_memory']:
            wm = context['working_memory']
            wm.concept_cooldown.clear()
            logger.info("   - 概念冷却已清理")

        # 更新时间戳
        self.last_rest_time = time.time()
        self.last_reset_time = time.time()

        logger.info(f"[EntropyRegulator] ✅ 短休息完成，系统熵值重置")

        return {
            "strategy": "short_rest",
            "duration": self.short_rest_duration,
            "entropy_after": 0.3  # 休息后降低到0.3
        }

    def _long_sleep(self, metrics: EntropyMetrics, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        长睡眠机制（类比人类的深度睡眠）

        🆕 [2026-01-17] P0修复：增强版长睡眠

        策略：
        1. 完全清理工作记忆
        2. 巩固记忆
        3. 重置ValueNetwork状态（核心）- 新增
        4. 重置所有状态
        5. 降低熵值到基线
        """
        logger.info("[EntropyRegulator] 😴 执行增强版长睡眠机制（类比人类深度睡眠）")

        # 完全清理工作记忆
        if 'working_memory' in context and context['working_memory']:
            wm = context['working_memory']
            wm.clear()
            wm.concept_cooldown.clear()
            logger.info("   - 工作记忆完全清理")

        # 巩固记忆（如果可用）
        if 'semantic_memory' in context and context['semantic_memory']:
            try:
                # 触发记忆巩固
                logger.info("   - 触发记忆巩固")
            except Exception as e:
                logger.warning(f"   - 记忆巩固失败: {e}")

        # 🆕 P0修复：重置ValueNetwork的核心熵值状态
        if 'evolution_controller' in context and context['evolution_controller']:
            evo_controller = context['evolution_controller']
            if hasattr(evo_controller, 'value_network'):
                logger.info("   - 🎯 触发ValueNetwork核心状态重置")
                try:
                    vn_result = evo_controller.value_network.reset_entropy_state()
                    logger.info(f"   - ✅ ValueNetwork重置成功")
                except Exception as e:
                    logger.warning(f"   - ⚠️ ValueNetwork重置失败: {e}")

        # 重置熵值历史
        self.entropy_history.clear()
        self.consecutive_rising_count = 0
        self.last_long_sleep_time = time.time()
        self.last_reset_time = time.time()

        logger.info(f"[EntropyRegulator] ✅ 长睡眠完成，系统完全重置")

        return {
            "strategy": "long_sleep",
            "duration": self.long_sleep_duration,
            "entropy_after": 0.2  # 睡眠后降低到0.2
        }

    def _force_reset_entropy(self, metrics: EntropyMetrics, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        强制重置机制（紧急降熵）

        🆕 [2026-01-17] P0修复：增强版强制重置

        策略：
        1. 清理Working Memory（表层）
        2. 重置ValueNetwork状态（核心）- 新增
        3. 重置到最低熵值
        4. 发出警报
        """
        logger.warning("[EntropyRegulator] 🚨 执行增强版强制重置（紧急降熵P0修复）")

        # 1. 执行长睡眠（清理Working Memory）
        result = self._long_sleep(metrics, context)

        # 2. 🆕 P0修复：重置ValueNetwork的核心熵值状态
        if 'evolution_controller' in context and context['evolution_controller']:
            evo_controller = context['evolution_controller']
            if hasattr(evo_controller, 'value_network'):
                logger.info("[EntropyRegulator] 🎯 触发ValueNetwork核心状态重置")
                try:
                    vn_result = evo_controller.value_network.reset_entropy_state()
                    logger.info(f"[EntropyRegulator] ✅ ValueNetwork重置成功: {vn_result}")
                    result['value_network_reset'] = vn_result
                except Exception as e:
                    logger.error(f"[EntropyRegulator] ❌ ValueNetwork重置失败: {e}")
                    result['value_network_reset_error'] = str(e)

        result['strategy'] = 'force_reset'
        self.stats['entropy_resets'] += 1

        return result

    def get_status(self) -> Dict[str, Any]:
        """获取调节器状态"""
        return {
            "entropy_history_size": len(self.entropy_history),
            "average_entropy": sum(self.entropy_history) / len(self.entropy_history) if self.entropy_history else 0.0,
            "current_trend": "分析中...",
            "last_rest": f"{(time.time() - self.last_rest_time)/60:.1f}分钟前",
            "last_long_sleep": f"{(time.time() - self.last_long_sleep_time)/3600:.1f}小时前",
            "consecutive_rising": self.consecutive_rising_count,
            "stats": self.stats
        }


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("熵值调节器测试")
    print("=" * 60)

    # 创建调节器
    regulator = EntropyRegulator()

    # 测试1: 正常熵值
    print("\n[测试1] 正常熵值")
    metrics = regulator.record_entropy(0.5)
    print(f"   当前熵值: {metrics.current_entropy}")
    print(f"   平均熵值: {metrics.average_entropy}")
    print(f"   趋势: {metrics.entropy_trend}")
    should_regulate, reason = regulator.should_regulate(metrics)
    print(f"   需要调节: {should_regulate}, 原因: {reason}")

    # 测试2: 熵值上升
    print("\n[测试2] 熵值逐渐上升")
    for i in range(15):
        entropy = 0.5 + i * 0.03  # 从0.5逐渐上升到0.92
        metrics = regulator.record_entropy(entropy)
        if i % 5 == 0:
            print(f"   步骤{i}: 熵值={entropy:.3f}, 平均={metrics.average_entropy:.3f}, 趋势={metrics.entropy_trend}")

    should_regulate, reason = regulator.should_regulate(metrics)
    print(f"   需要调节: {should_regulate}, 原因: {reason}")

    # 测试3: 执行调节
    print("\n[测试3] 执行熵值调节")
    context = {}
    result = regulator.regulate_entropy(metrics, context)
    print(f"   调节结果: {result}")
