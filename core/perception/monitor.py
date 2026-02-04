"""
Phase 3.2 Stage 4 - 感知系统监控扩展

扩展自我监控层,添加对real_time_perception系统的性能追踪

功能:
1. 摄像头捕获性能监控 (FPS、帧延迟)
2. 麦克风音频性能监控 (采样率、缓冲区状态)
3. 感知系统资源使用追踪
4. 感知事件统计

作者: GitHub Copilot (Claude Sonnet 4.5)
创建时间: 2025-11-22
版本: 1.0.0
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerceptionMetrics:
    """感知系统指标"""
    timestamp: float
    
    # 摄像头指标
    camera_fps: float = 0.0
    camera_frame_count: int = 0
    camera_dropped_frames: int = 0
    camera_avg_latency_ms: float = 0.0
    camera_status: str = "unknown"
    
    # 麦克风指标
    audio_sample_rate: int = 0
    audio_chunk_count: int = 0
    audio_buffer_usage: float = 0.0  # 0.0-1.0
    audio_status: str = "unknown"
    
    # 整体指标
    total_perception_events: int = 0
    perception_active: bool = False


class PerceptionMonitorExtension:
    """
    感知系统监控扩展
    
    扩展AGISelfMonitoringLayer,添加对PerceptionManager的监控
    """
    
    def __init__(self, monitoring_layer, perception_manager=None):
        """
        初始化感知监控扩展
        
        Args:
            monitoring_layer: AGISelfMonitoringLayer实例
            perception_manager: PerceptionManager实例(可选)
        """
        self.monitoring_layer = monitoring_layer
        self.perception_manager = perception_manager
        
        # 使用双端队列优化历史记录的插入和删除操作
        self.metrics_history: deque[PerceptionMetrics] = deque(maxlen=3600)  # maxlen自动管理大小
        
        # 统计（线程安全）
        self.stats_lock = threading.RLock()
        self.stats = {
            'total_samples': 0,
            'camera_total_frames': 0,
            'audio_total_chunks': 0,
            'perception_errors': 0
        }
        
        # 性能优化：缓存常用属性访问路径
        self._last_capture_time_cache: Optional[float] = None
        self._frame_count_cache: int = 0
        
        logger.info("✅ 感知系统监控扩展初始化完成")
    
    def set_perception_manager(self, perception_manager) -> None:
        """设置感知管理器引用"""
        with self.stats_lock:
            self.perception_manager = perception_manager
        logger.info("✅ 感知管理器引用已设置")
    
    @contextmanager
    def _safe_attribute_access(self, obj: Any, operation: str):
        """上下文管理器，用于安全地访问对象属性并处理异常"""
        try:
            yield
        except AttributeError:
            pass  # 属于正常情况，某些属性可能不存在
        except Exception as e:
            error_msg = f"访问感知组件属性失败 ({operation}): {e}"
            logger.debug(error_msg)  # 使用debug避免日志泛滥
            with self.stats_lock:
                self.stats['perception_errors'] += 1
            if self.monitoring_layer:
                self.monitoring_layer.capture_exception(
                    e,
                    context={'operation': f'attribute_access_{operation}'},
                    severity='warning',
                    component='perception_monitor'
                )
    
    def capture_perception_metrics(self) -> PerceptionMetrics:
        """
        捕获当前感知系统指标
        
        Returns:
            PerceptionMetrics对象
        """
        timestamp = time.time()
        metrics = PerceptionMetrics(timestamp=timestamp)
        
        if self.perception_manager is None:
            logger.debug("感知管理器未设置,返回空指标")
            return metrics
        
        try:
            # 获取摄像头指标
            with self._safe_attribute_access(self.perception_manager, 'camera'):
                camera = getattr(self.perception_manager, 'camera', None)
                if camera is not None:
                    metrics.camera_status = getattr(camera.status, 'value', 'unknown')
                    frame_count = getattr(camera, 'frame_count', 0)
                    metrics.camera_frame_count = frame_count
                    
                    # 计算FPS基于增量变化，减少重复计算
                    if frame_count > self._frame_count_cache:
                        last_time = getattr(camera, 'last_capture_time', None)
                        if last_time:
                            time_delta = timestamp - last_time
                            if 0 < time_delta < 5:  # 合理范围限制
                                metrics.camera_fps = 1.0 / time_delta
                            self._last_capture_time_cache = last_time
                        self._frame_count_cache = frame_count
            
            # 获取麦克风指标
            with self._safe_attribute_access(self.perception_manager, 'microphone'):
                mic = getattr(self.perception_manager, 'microphone', None)
                if mic is not None:
                    metrics.audio_status = getattr(mic.status, 'value', 'unknown')
                    metrics.audio_sample_rate = getattr(mic.config, 'sample_rate', 0)
                    
                    # 缓冲区使用率
                    buffer = getattr(mic, 'audio_buffer', None)
                    if buffer is not None:
                        try:
                            buffer_size = buffer.qsize()
                            max_size = getattr(buffer, 'maxsize', 0)
                            if max_size > 0:
                                usage = min(max(buffer_size / max_size, 0.0), 1.0)
                                metrics.audio_buffer_usage = usage
                        except (OSError, ValueError) as e:
                            logger.debug(f"无法读取音频缓冲区状态: {e}")
            
            # 整体状态
            with self._safe_attribute_access(self.perception_manager, 'is_running'):
                metrics.perception_active = getattr(self.perception_manager, 'is_running', False)
            
            # 线程安全更新历史与统计
            with self.stats_lock:
                self.metrics_history.append(metrics)
                self.stats['total_samples'] += 1
                self.stats['camera_total_frames'] += metrics.camera_frame_count
                
        except Exception as e:
            logger.error(f"捕获感知指标失败: {type(e).__name__}: {e}")
            with self.stats_lock:
                self.stats['perception_errors'] += 1
            
            if self.monitoring_layer:
                self.monitoring_layer.capture_exception(
                    e,
                    context={'operation': 'capture_perception_metrics'},
                    severity='warning',
                    component='perception_monitor'
                )
        
        return metrics
    
    def get_perception_statistics(self) -> Dict[str, Any]:
        """
        获取感知系统统计信息（高效聚合）
        
        Returns:
            统计信息字典
        """
        history_len = len(self.metrics_history)
        if history_len == 0:
            return {
                'status': 'no_data',
                'message': '暂无感知指标数据'
            }
        
        # 最新指标
        latest = self.metrics_history[-1]
        
        # 只取最近100个样本进行平均计算
        recent_count = min(100, history_len)
        recent_slice = list(self.metrics_history)[-recent_count:]  # deque切片转为列表
        
        total_camera_fps = 0.0
        total_audio_buffer = 0.0
        for m in recent_slice:
            total_camera_fps += m.camera_fps
            total_audio_buffer += m.audio_buffer_usage
        
        avg_camera_fps = total_camera_fps / recent_count
        avg_audio_buffer = total_audio_buffer / recent_count
        
        with self.stats_lock:
            total_samples = self.stats['total_samples']
            total_errors = self.stats['perception_errors']
        
        return {
            'current_state': {
                'camera_status': latest.camera_status,
                'camera_fps': round(latest.camera_fps, 2),
                'camera_frame_count': latest.camera_frame_count,
                'audio_status': latest.audio_status,
                'audio_sample_rate': latest.audio_sample_rate,
                'audio_buffer_usage': round(latest.audio_buffer_usage, 3),
                'perception_active': latest.perception_active
            },
            'statistics': {
                'avg_camera_fps_100samples': round(avg_camera_fps, 2),
                'avg_audio_buffer_usage': round(avg_audio_buffer, 3),
                'total_samples': total_samples,
                'total_errors': total_errors
            },
            'history_size': history_len,
            'timestamp': latest.timestamp
        }
    
    def log_perception_summary(self) -> None:
        """记录感知系统摘要到日志（避免不必要的格式化开销）"""
        stats = self.get_perception_statistics()
        
        if stats.get('status') == 'no_data':
            logger.info("📡 [感知系统监控] 暂无数据")
            return
        
        current = stats['current_state']
        stats_data = stats['statistics']
        
        # 使用f-string避免日志格式化错误（修复 %.0%%% 转义问题）
        logger.info(
            f"📡 [感知系统监控] "
            f"摄像头={current['camera_status']}({current['camera_fps']:.1f}fps) | "
            f"音频={current['audio_status']}(buffer={current['audio_buffer_usage'] * 100:.0f}%) | "
            f"采样={stats_data['total_samples']}, 错误={stats_data['total_errors']}"
        )


def extend_monitoring_with_perception(
    monitoring_layer: Any, 
    perception_manager: Optional[Any] = None
) -> PerceptionMonitorExtension:
    """
    为监控层添加感知系统监控能力
    
    Args:
        monitoring_layer: AGISelfMonitoringLayer实例
        perception_manager: PerceptionManager实例(可选)
    
    Returns:
        PerceptionMonitorExtension实例
    """
    extension = PerceptionMonitorExtension(
        monitoring_layer=monitoring_layer,
        perception_manager=perception_manager
    )
    
    # 将扩展附加到监控层
    setattr(monitoring_layer, 'perception_monitor', extension)
    
    logger.info("✅ 感知系统监控扩展已附加到监控层")
    return extension


if __name__ == '__main__':
    # 测试
    print("感知系统监控扩展模块")
    print("=" * 70)
    
    # 创建模拟监控层
    class MockMonitoringLayer:
        def capture_exception(self, *args, **kwargs):
            pass
    
    mock_layer = MockMonitoringLayer()
    extension = PerceptionMonitorExtension(monitoring_layer=mock_layer)
    
    # 捕获指标
    metrics = extension.capture_perception_metrics()
    print(f"\n捕获的指标: {metrics}")
    
    # 获取统计
    stats = extension.get_perception_statistics()
    print(f"\n统计信息: {stats}")
    
    print("\n✅ 模块测试完成")