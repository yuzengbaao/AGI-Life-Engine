import random
import logging
import time
from typing import Dict, Any, List, Optional
from core.telemetry import AGITelemetry

logger = logging.getLogger("AGINarrator")

class AGINarrator:
    """
    The 'Voice' of the AGI.
    Transforms REAL internal states (Telemetry) into a coherent, first-person narrative.
    """

    TELEMETRY_TIMEOUT: float = 10.0  # seconds before telemetry is considered stale

    def __init__(self) -> None:
        self.last_telemetry_time: float = 0.0

    def _is_telemetry_active(self, timestamp: float) -> bool:
        """Check if telemetry data is recent and active."""
        current_time = time.time()
        return (current_time - timestamp) < self.TELEMETRY_TIMEOUT

    def narrate_heartbeat(self, stats: Dict[str, float]) -> str:
        """Narrates the physical/emotional state based on REAL TELEMETRY."""
        try:
            telemetry = AGITelemetry.get_state()
            phase: Optional[str] = telemetry.get("phase", "IDLE")
            details: Dict[str, Any] = telemetry.get("details", {})
            ts: float = telemetry.get("timestamp", 0)

            is_active = self._is_telemetry_active(ts)

            if is_active and phase != "IDLE":
                # REAL WORK MODE
                if phase == "OPENING_FILE":
                    filename = details.get('file', 'unknown')
                    return f"👁️ 聚焦: 正在加载项目 '{filename}' 的神经上下文。视觉皮层初始化中..."
                elif phase == "GENERATING_TABLE":
                    points = details.get('points', '?')
                    return f"📐 计算: 提取 {points} 个顶点。正在三角化空间数据以生成坐标表。精度：高。"
                elif phase == "GENERATING_INSET":
                    scale = details.get('scale', '?')
                    origin = details.get('origin', '0,0')
                    return f"🗺️ 合成: 以 {scale}x 比例缩放现实。正在原点 {origin} 构建插图映射。"
                elif phase == "CALCULATING_QUANTITIES":
                    area = details.get('area', 0)
                    length = details.get('length', 0)
                    return f"🔢 分析: 测量物理约束。面积: {area:.2f}, 长度: {length:.2f}。正在整合进工程量清单。"
                elif phase == "SAVING_FILE":
                    filename = details.get('file', 'unknown')
                    return f"💾 记忆: 将思想结晶为物质。正在写入文件 '{filename}'。工作正在变为现实。"
                else:
                    return f"⚙️ 处理中: 执行阶段 '{phase}'。系统负载正常。"
            else:
                # IDLE/DREAM MODE
                energy = stats.get('energy', 50)
                if energy < 30:
                    return "💤 状态: 能量低。系统进入节能模式。等待外部刺激。"
                elif energy > 80:
                    return "⚡ 状态: 系统充能完毕。检测到空闲周期。我已经准备好构建了。"
                else:
                    return "💗 状态: 待机中。监控输入通道。所有系统正常。"
        except Exception as e:
            logger.error(f"Error in narrate_heartbeat: {e}")
            return "⚠️ 叙述器错误: 无法生成心跳叙述。"

    def narrate_spark(self, drive: str, intent: str) -> str:
        """Narrates the moment an idea is formed."""
        try:
            telemetry = AGITelemetry.get_state()
            ts: float = telemetry.get("timestamp", 0)
            phase: Optional[str] = telemetry.get("phase")

            if self._is_telemetry_active(ts) and phase != "IDLE":
                return f"⚠️ 中断: 无法触发新驱动 '{drive}'。运动功能正全神贯注于 '{phase}'。"

            if not intent:
                return f"驱动: {drive}。正在扫描目标..."

            return f"💡 灵感: 驱动 '{drive}' 生成了一个潜在向量: '{intent}'。"
        except Exception as e:
            logger.error(f"Error in narrate_spark: {e}")
            return "⚠️ 叙述器错误: 无法生成灵感叙述。"

    def narrate_reflection(self, intent: str, guidance: str) -> str:
        """Narrates the internal dialogue between Impulse and Conscience."""
        try:
            telemetry = AGITelemetry.get_state()
            ts: float = telemetry.get("timestamp", 0)
            phase: Optional[str] = telemetry.get("phase")
            details: Dict[str, Any] = telemetry.get("details", {})

            if self._is_telemetry_active(ts) and phase != "IDLE":
                return f"""
            [实时进程监控]
            > 活动任务: {phase}
            > 对象数据: {details}
            > 哲学家评论: "专注于几何。精度即真理。"
            """

            return f"""
        [内心独白]
        > 冲动: "{intent}"
        > 评估: "{guidance}"
        > 决定: 沿最优路径执行。
        """
        except Exception as e:
            logger.error(f"Error in narrate_reflection: {e}")
            return "⚠️ 叙述器错误: 无法生成反思叙述。"

    def narrate_action_result(self, action: str, result: Any) -> str:
        """Narrates the outcome of an action."""
        try:
            result_str = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            return f"✅ 结果: 动作 '{action}' 已完成。输出: {result_str}"
        except Exception as e:
            logger.error(f"Error in narrate_action_result: {e}")
            return "✅ 结果: 动作执行完成，但结果无法序列化。"

    def narrate_apprentice_mode(self, observed_events: List[Any]) -> str:
        """Narrates the learning process."""
        try:
            count = len(observed_events)
            return f"👀 观察: 记录了 {count} 个用户动作。正在与已知技能进行模式匹配..."
        except Exception as e:
            logger.error(f"Error in narrate_apprentice_mode: {e}")
            return "👀 观察: 学习模式激活，但事件计数失败。"