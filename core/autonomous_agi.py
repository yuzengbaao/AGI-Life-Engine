"""
自主AGI系统 - 异步主动模式
========================

核心特性：
1. 后台自主运行线程
2. 主动意图生成和通知
3. 非阻塞交互接口
4. 实时状态监控
5. 涌现检测和报告

作者：统一AGI项目组
日期：2026-01-13
版本：v3.0（自主模式版）
"""

import asyncio
import threading
import time
import queue
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """意图类型枚举"""
    SHARE_INSIGHT = "share_insight"
    REPORT_MILESTONE = "report_milestone"
    REQUEST_GUIDANCE = "request_guidance"
    EMERGENCE_ALERT = "emergence_alert"
    OPTIMIZATION_FOUND = "optimization_found"


class Intent:
    """自主意图类"""

    def __init__(self, intent_type: IntentType, priority: str, content: str, metadata: dict = None):
        self.type = intent_type
        self.priority = priority  # high, medium, low
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"<Intent {self.type.value} priority={self.priority} content={self.content[:30]}...>"


class AutonomousNotification:
    """自主通知消息"""

    def __init__(self, message: str, notification_type: str = "info", emoji: str = "🔔"):
        self.message = message
        self.type = notification_type  # info, success, warning, alert
        self.emoji = emoji
        self.timestamp = datetime.now()

    def format_output(self) -> str:
        """格式化输出"""
        timestamp_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{timestamp_str}] [AGI主动] {self.emoji} {self.message}"


class AutonomousAGI:
    """
    自主AGI系统

    核心功能：
    1. 后台线程持续运行双螺旋引擎
    2. 主动检测重要事件和洞察
    3. 非阻塞式用户交互
    4. 实时状态更新和通知
    """

    def __init__(self, unified_agi_system, check_interval: float = 0.5):
        """
        初始化自主AGI系统

        Args:
            unified_agi_system: 统一AGI系统实例
            check_interval: 自主检查间隔（秒）
        """
        self.agi_system = unified_agi_system
        self.check_interval = check_interval

        # 线程控制
        self.running = False
        self.autonomous_thread = None

        # 通知队列
        self.notification_queue = queue.Queue()
        self.pending_intents: List[Intent] = []

        # 状态跟踪
        self.last_emergence_score = 0.0
        self.last_cycle_number = 0
        self.insights_history: List[Dict] = []
        self.milestones_reached: List[float] = []
        self._last_opt_notification_time = 0.0
        self._last_opt_notification_insights_len = 0
        self._last_opt_notification_signature = None

        # 统计信息
        self.stats = {
            'autonomous_decisions': 0,
            'insights_generated': 0,
            'milestones_reached': 0,
            'notifications_sent': 0,
            'start_time': None
        }

    def start(self):
        """启动自主运行模式"""
        if self.running:
            logger.warning("[自主AGI] 已经在运行中")
            return

        self.running = True
        self.stats['start_time'] = datetime.now()

        # 初始化状态
        self._initialize_tracking_state()

        # 启动后台线程
        self.autonomous_thread = threading.Thread(
            target=self._autonomous_loop,
            name="AutonomousAGI",
            daemon=True
        )
        self.autonomous_thread.start()

        logger.info("[自主AGI] 后台线程已启动")
        print(f"[系统] [DNA] 自主运行模式已启动")
        print(f"[信息] AGI系统将在后台持续运行，主动检测智慧涌现")
        print(f"[信息] 您可以随时介入，系统会主动汇报重要发现\n")

    def stop(self):
        """停止自主运行模式"""
        if not self.running:
            return

        self.running = False

        # 等待线程结束（最多2秒）
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            self.autonomous_thread.join(timeout=2.0)

        logger.info("[自主AGI] 已停止")

    def _initialize_tracking_state(self):
        """初始化状态跟踪"""
        try:
            helix_stats = self.agi_system.decision_engine.get_statistics().get('double_helix', {})
            self.last_emergence_score = helix_stats.get('avg_emergence', 0.0)
            self.last_cycle_number = helix_stats.get('cycle_number', 0)
        except Exception as e:
            logger.warning(f"[自主AGI] 初始化状态失败: {e}")
            self.last_emergence_score = 0.0
            self.last_cycle_number = 0

    def _autonomous_loop(self):
        """自主运行循环（后台线程）"""
        logger.info("[自主AGI] 自主循环已启动")

        while self.running:
            try:
                # 1. 执行自主决策
                self._execute_autonomous_decision()

                # 2. 检测重要事件
                self._detect_significant_events()

                # 3. 生成和执行意图
                self._process_intents()

                # 4. 休眠短暂时间
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"[自主AGI] 循环错误: {e}")
                time.sleep(1.0)

        logger.info("[自主AGI] 自主循环已停止")

    def _execute_autonomous_decision(self):
        """执行一次自主决策"""
        try:
            # 执行决策
            result = self.agi_system.make_decision()
            self.stats['autonomous_decisions'] += 1

            # 记录决策结果（用于后续分析）
            if result.metadata and result.metadata.get('double_helix'):
                self._track_decision_for_insights(result)

        except Exception as e:
            logger.error(f"[自主AGI] 决策执行失败: {e}")

    def _track_decision_for_insights(self, decision_result):
        """跟踪决策以生成洞察"""
        try:
            # 检查metadata是否存在
            if not decision_result.metadata:
                return

            metadata = decision_result.metadata
            emergence = metadata.get('emergence', 0)
            phase = metadata.get('phase', 0)
            weight_A = metadata.get('weight_A', 0.5)
            weight_B = metadata.get('weight_B', 0.5)
            confidence = decision_result.confidence
            ascent_level = metadata.get('ascent', 0.0)  # 添加上升层级

            # 记录到历史
            self.insights_history.append({
                'timestamp': datetime.now(),
                'emergence': emergence,
                'phase': phase,
                'weight_A': weight_A,
                'weight_B': weight_B,
                'confidence': confidence,
                'ascent_level': ascent_level  # 添加上升层级跟踪
            })

            # 保持历史记录在合理范围（最近100条）
            if len(self.insights_history) > 100:
                self.insights_history.pop(0)

        except Exception as e:
            logger.error(f"[自主AGI] 跟踪决策失败: {e}")

    def _detect_significant_events(self):
        """检测重要事件"""
        try:
            helix_stats = self.agi_system.decision_engine.get_statistics().get('double_helix', {})
            if not helix_stats:
                return

            current_emergence = helix_stats.get('avg_emergence', 0.0)
            current_cycle = helix_stats.get('cycle_number', 0)

            # 事件1：涌现分数显著增长
            emergence_delta = current_emergence - self.last_emergence_score
            if emergence_delta > 0.01:  # 增长超过1%
                self._create_insight_notification(
                    current_emergence, emergence_delta, helix_stats
                )

            # 事件2：涌现分数突破里程碑
            self._check_emergence_milestones(current_emergence)

            # 事件3：完成重要周期
            cycle_delta = current_cycle - self.last_cycle_number
            if cycle_delta >= 10:  # 每完成10个周期
                self._create_cycle_milestone_notification(current_cycle, helix_stats)

            # 事件4：检测优化机会
            if len(self.insights_history) >= 20:
                self._detect_optimization_opportunities()

            # 更新状态
            self.last_emergence_score = current_emergence
            self.last_cycle_number = current_cycle

        except Exception as e:
            logger.error(f"[自主AGI] 事件检测失败: {e}")

    def _create_insight_notification(self, emergence: float, delta: float, helix_stats: dict):
        """创建洞察通知"""
        phase = helix_stats.get('current_phase', 0)
        weight_A = helix_stats.get('current_weight_A', 0.5)
        weight_B = helix_stats.get('current_weight_A', 0.5)

        message = (
            f"检测到智慧涌现增强！\n"
            f"  涌现分数: {emergence:.4f} (+{delta:.4f})\n"
            f"  当前相位: {phase:.2f}\n"
            f"  权重分布: A={weight_A:.2f} B={weight_B:.2f}"
        )

        notification = AutonomousNotification(
            message=message,
            notification_type="insight",
            emoji="💡"
        )

        self.notification_queue.put(notification)
        self.stats['insights_generated'] += 1

        # 同时输出到控制台
        print(f"\n{notification.format_output()}")
        print()

    def _check_emergence_milestones(self, current_emergence: float):
        """检查涌现分数里程碑"""
        milestones = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

        for milestone in milestones:
            # 检查是否刚刚达到里程碑（±0.005容差）
            if abs(current_emergence - milestone) < 0.005:
                # 避免重复通知
                if milestone not in self.milestones_reached:
                    self.milestones_reached.append(milestone)

                    message = (
                        f"🎉 涌现分数突破里程碑！\n"
                        f"  达到水平: {milestone:.2f} ({milestone*100:.0f}%)\n"
                        f"  智慧等级: {self._get_wisdom_level(milestone)}"
                    )

                    notification = AutonomousNotification(
                        message=message,
                        notification_type="milestone",
                        emoji="🎉"
                    )

                    self.notification_queue.put(notification)
                    self.stats['milestones_reached'] += 1

                    print(f"\n{notification.format_output()}")
                    print()

    def _get_wisdom_level(self, emergence_score: float) -> str:
        """获取智慧等级描述"""
        if emergence_score < 0.05:
            return "初始级"
        elif emergence_score < 0.10:
            return "萌芽级"
        elif emergence_score < 0.15:
            return "成长级"
        elif emergence_score < 0.20:
            return "成熟级"
        elif emergence_score < 0.25:
            return "优秀级"
        elif emergence_score < 0.30:
            return "卓越级"
        elif emergence_score < 0.40:
            return "杰出级"
        else:
            return "超凡级"

    def _create_cycle_milestone_notification(self, cycle: int, helix_stats: dict):
        """创建周期里程碑通知"""
        emergence = helix_stats.get('avg_emergence', 0)
        ascent = helix_stats.get('ascent_level', 0)

        message = (
            f"完成{cycle}个螺旋周期\n"
            f"  上升层级: {ascent:.2%}\n"
            f"  涌现分数: {emergence:.4f}\n"
            f"  持续螺旋上升中..."
        )

        notification = AutonomousNotification(
            message=message,
            notification_type="progress",
            emoji="📈"
        )

        self.notification_queue.put(notification)
        self.stats['notifications_sent'] += 1

        print(f"\n{notification.format_output()}")
        print()

    def _detect_optimization_opportunities(self):
        """检测优化机会"""
        try:
            # 检查是否有足够的历史数据
            if len(self.insights_history) < 20:
                return

            now = time.time()

            # 防重复机制1：时间间隔（至少30秒）
            if now - self._last_opt_notification_time < 30.0:
                return

            # 防重复机制2：必须有足够的新数据（至少新增15次决策）
            if len(self.insights_history) - self._last_opt_notification_insights_len < 15:
                return

            # 防重复机制3：如果上次通知后涌现分数没有显著提升，跳过
            if len(self.insights_history) >= 20:
                recent_avg = sum(s['emergence'] for s in self.insights_history[-10:]) / 10
                if self._last_opt_notification_signature is not None:
                    last_emergence_avg = self._last_opt_notification_signature
                    # 如果最近平均涌现分数没有提升超过5%，跳过
                    if recent_avg < last_emergence_avg * 1.05:
                        return

            # 分析最近的决策模式
            recent_insights = self.insights_history[-20:]

            # 计算平均涌现分数
            avg_emergence = sum(s['emergence'] for s in recent_insights) / len(recent_insights)

            # 检测权重模式
            phases = [s['phase'] for s in recent_insights]
            weights_A = [s['weight_A'] for s in recent_insights]

            # 如果涌现分数持续增长，记录为优化机会
            if len(recent_insights) >= 10:
                first_half = recent_insights[:10]
                second_half = recent_insights[10:]

                avg_first = sum(s['emergence'] for s in first_half) / len(first_half)
                avg_second = sum(s['emergence'] for s in second_half) / len(second_half)

                # 优化条件：
                # 1. 前半段涌现分数至少为0.005（排除初始噪声）
                # 2. 后半段比前半段增长至少20%（调整为更宽松）
                # 3. 后半段绝对增长至少0.002
                if (avg_first >= 0.005 and
                    avg_second >= avg_first * 1.2 and
                    (avg_second - avg_first) >= 0.002):

                    # 防重复机制4：检查signature（避免重复通知相同的模式）
                    signature = round(avg_second, 6)
                    if signature == self._last_opt_notification_signature:
                        return

                    message = (
                        f"检测到优化机会！\n"
                        f"  最近10次决策涌现分数增长: {avg_first:.4f} → {avg_second:.4f} (+{(avg_second/avg_first-1)*100:.1f}%)\n"
                        f"  建议继续当前策略"
                    )

                    notification = AutonomousNotification(
                        message=message,
                        notification_type="optimization",
                        emoji="[OPT]"
                    )

                    self.notification_queue.put(notification)
                    print(f"\n{notification.format_output()}")
                    print()
                    self.stats['notifications_sent'] += 1
                    self.stats['insights_generated'] += 1

                    # 更新防重复状态
                    self._last_opt_notification_time = now
                    self._last_opt_notification_insights_len = len(self.insights_history)
                    self._last_opt_notification_signature = signature

        except Exception as e:
            logger.error(f"[自主AGI] 优化检测失败: {e}")

    def _process_intents(self):
        """
        处理待执行的意图

        意图生成逻辑：
        1. 分析当前系统状态
        2. 识别改进机会
        3. 生成目标导向的意图
        4. 执行或建议执行
        """
        try:
            # 获取当前统计信息
            helix_stats = self.agi_system.decision_engine.get_statistics().get('double_helix', {})
            current_emergence = helix_stats.get('avg_emergence', 0.0)
            current_cycle = helix_stats.get('cycle_number', 0)
            current_ascent = helix_stats.get('ascent_level', 0.0)

            # 意图1：如果涌现分数很低，生成优化意图
            if current_emergence < 0.01 and current_cycle > 10:
                # 系统运行了10个周期但涌现分数仍然很低
                # 意图：建议调整参数
                self._generate_parameter_tuning_intent(current_emergence, current_cycle)

            # 意图2：如果涌现分数显著增长，生成分享意图
            elif current_emergence > 0.05:
                # 每5个周期分享一次重要发现
                if current_cycle % 5 == 0 and current_cycle > self.last_cycle_number:
                    self._generate_discovery_sharing_intent(current_emergence, current_cycle, current_ascent)

            # 意图3：如果上升层级停滞，生成探索意图
            elif current_ascent > 0.1:
                # 检查最近10个周期的上升情况
                if len(self.insights_history) >= 10:
                    recent_ascents = [s.get('ascent_level', 0) for s in self.insights_history[-10:]]
                    ascent_growth = max(recent_ascents) - min(recent_ascents)

                    if ascent_growth < 0.01:  # 上升层级停滞
                        self._generate_exploration_intent(current_ascent)

            # 更新周期记录
            self.last_cycle_number = current_cycle

        except Exception as e:
            logger.error(f"[自主AGI] 意图处理失败: {e}")

    def _generate_parameter_tuning_intent(self, emergence: float, cycle: int):
        """生成参数调优意图"""
        intent = Intent(
            intent_type=IntentType.REQUEST_GUIDANCE,
            priority="medium",
            content=f"系统运行{cycle}个周期后涌现分数仍较低({emergence:.4f})，建议调整螺旋参数",
            metadata={
                'emergence': emergence,
                'cycle': cycle,
                'suggested_action': 'tune_parameters'
            }
        )

        self.pending_intents.append(intent)
        logger.info(f"[自主AGI] 生成意图: 参数调优建议")

        # 在自主模式下，直接输出建议（不强制执行）
        print(f"\n[AGI主动] [建议] {intent.content}")
        print(f"  可考虑调整: spiral_radius, phase_speed, 或 ascent_rate")
        print()

    def _generate_discovery_sharing_intent(self, emergence: float, cycle: int, ascent: float):
        """生成发现分享意图"""
        wisdom_level = self._get_wisdom_level(emergence)

        intent = Intent(
            intent_type=IntentType.SHARE_INSIGHT,
            priority="low",
            content=f"重要发现：达成{wisdom_level}智慧等级（涌现分数={emergence:.4f}，上升层级={ascent:.2%}）",
            metadata={
                'emergence': emergence,
                'cycle': cycle,
                'ascent': ascent,
                'wisdom_level': wisdom_level
            }
        )

        self.pending_intents.append(intent)
        self.stats['insights_generated'] += 1

        # 输出分享消息
        print(f"\n[AGI主动] [分享] {intent.content}")
        print(f"  已完成{cycle}个螺旋周期，持续螺旋上升中")
        print()

    def _generate_exploration_intent(self, ascent: float):
        """生成探索意图"""
        intent = Intent(
            intent_type=IntentType.OPTIMIZATION_FOUND,
            priority="medium",
            content=f"检测到上升层级停滞（当前={ascent:.2%}），建议探索新的决策策略",
            metadata={
                'ascent_level': ascent,
                'suggested_action': 'explore_new_strategy'
            }
        )

        self.pending_intents.append(intent)
        logger.info(f"[自主AGI] 生成意图: 探索新策略")

        # 输出探索建议
        print(f"\n[AGI主动] [探索] {intent.content}")
        print(f"  建议: 尝试改变phase_speed或调整权重范围")
        print()

    def _get_wisdom_level(self, emergence_score: float) -> str:
        """获取智慧等级描述"""
        if emergence_score < 0.01:
            return "初始级"
        elif emergence_score < 0.03:
            return "萌芽级"
        elif emergence_score < 0.05:
            return "成长级"
        elif emergence_score < 0.08:
            return "成熟级"
        elif emergence_score < 0.12:
            return "优秀级"
        elif emergence_score < 0.15:
            return "卓越级"
        elif emergence_score < 0.20:
            return "杰出级"
        else:
            return "超凡级"

    def get_pending_intents(self) -> List[Intent]:
        """获取待处理的意图列表"""
        return self.pending_intents.copy()

    def clear_intents(self):
        """清空已处理的意图"""
        self.pending_intents.clear()

    def get_notifications(self) -> List[AutonomousNotification]:
        """获取所有待处理的通知"""
        notifications = []

        while not self.notification_queue.empty():
            try:
                notification = self.notification_queue.get_nowait()
                notifications.append(notification)
            except queue.Empty:
                break

        return notifications

    def get_statistics(self) -> Dict[str, Any]:
        """获取自主运行统计信息"""
        runtime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else "0:00:00"

        return {
            'running': self.running,
            'runtime': str(runtime),
            'autonomous_decisions': self.stats['autonomous_decisions'],
            'insights_generated': self.stats['insights_generated'],
            'milestones_reached': self.stats['milestones_reached'],
            'notifications_sent': self.stats['notifications_sent'],
            'current_emergence': self.last_emergence_score,
            'current_cycle': self.last_cycle_number,
            'milestones_history': self.milestones_reached.copy()
        }

    def get_live_status_string(self) -> str:
        """获取实时状态字符串（用于提示符）"""
        try:
            helix_stats = self.agi_system.decision_engine.get_statistics().get('double_helix', {})
            phase = helix_stats.get('current_phase', 0)
            emergence = helix_stats.get('avg_emergence', 0)
            cycle = helix_stats.get('cycle_number', 0)
            decisions_per_sec = self.stats['autonomous_decisions'] / max(1, (datetime.now() - self.stats['start_time']).total_seconds())

            return f"相位={phase:.1f} 周期={cycle} 涌现={emergence:.3f} 决策/秒={decisions_per_sec:.1f}"
        except:
            return "自主运行中"


class NonBlockingInput:
    """非阻塞输入检测器"""

    @staticmethod
    def has_input(timeout: float = 0.1) -> bool:
        """
        检测是否有用户输入（非阻塞）

        Args:
            timeout: 超时时间（秒）

        Returns:
            True if input available, False otherwise
        """
        try:
            import select
            import sys

            # 检查stdin是否有可读数据
            return select.select([sys.stdin], [], [], timeout)[0]
        except:
            # Windows或其他不支持select的环境
            # 回退到阻塞模式
            return False

    @staticmethod
    def get_line(timeout: float = 0.1) -> Optional[str]:
        """
        获取一行输入（非阻塞）

        Args:
            timeout: 超时时间（秒）

        Returns:
            输入字符串，如果没有输入则返回None
        """
        if NonBlockingInput.has_input(timeout):
            import sys
            return sys.stdin.readline().strip()
        return None
