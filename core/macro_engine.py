import time
import threading
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroActionType(Enum):
    """宏操作动作类型"""
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_SCROLL = "mouse_scroll"
    WAIT = "wait"
    CUSTOM_FUNCTION = "custom_function"

@dataclass
class MacroAction:
    """宏操作动作定义"""
    action_type: MacroActionType
    params: Dict[str, Any]
    timestamp: float = 0.0
    description: str = ""

class MacroPlaybackState(Enum):
    """宏播放状态"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    RECORDING = "recording"

class MacroEngine:
    """
    桌面宏操作引擎核心类
    实现宏的录制、回放、编辑和管理功能
    """

    def __init__(self):
        """初始化宏引擎"""
        self._actions: List[MacroAction] = []
        self._state: MacroPlaybackState = MacroPlaybackState.STOPPED
        self._playback_thread: Optional[threading.Thread] = None
        self._is_looping: bool = False
        self._loop_count: int = 1
        self._current_loop: int = 0
        self._speed_factor: float = 1.0
        self._recorded_callbacks: List[Callable[[MacroAction], None]] = []
        self._played_callbacks: List[Callable[[MacroAction], None]] = []
        self._start_time: float = 0.0
        self._pause_time: float = 0.0
        
        # 注册系统事件监听器（模拟）
        self._setup_system_listeners()

    def _setup_system_listeners(self):
        """设置系统事件监听器（模拟实现）"""
        logger.info("初始化系统事件监听器")
        # 这里应该注册真实的系统钩子
        # 由于平台限制，此处仅做模拟

    def record_start(self) -> bool:
        """
        开始录制宏操作
        
        Returns:
            bool: 是否成功开始录制
        """
        if self._state != MacroPlaybackState.STOPPED:
            logger.warning("无法开始录制，当前状态不允许")
            return False

        self._state = MacroPlaybackState.RECORDING
        self._actions.clear()
        self._start_time = time.time()
        logger.info("开始录制宏操作")
        return True

    def record_stop(self) -> bool:
        """
        停止录制宏操作
        
        Returns:
            bool: 是否成功停止录制
        """
        if self._state != MacroPlaybackState.RECORDING:
            logger.warning("无法停止录制，当前未在录制状态")
            return False

        self._state = MacroPlaybackState.STOPPED
        logger.info(f"录制结束，共记录 {len(self._actions)} 个操作")
        return True

    def record_action(self, action: MacroAction) -> bool:
        """
        记录单个动作（供内部或外部调用）
        
        Args:
            action: 要记录的动作
            
        Returns:
            bool: 是否成功记录
        """
        if self._state != MacroPlaybackState.RECORDING:
            return False

        # 计算相对时间戳
        current_time = time.time()
        action.timestamp = current_time - self._start_time

        self._actions.append(action)
        logger.debug(f"记录动作: {action.action_type.value}")
        
        # 触发回调
        for callback in self._recorded_callbacks:
            try:
                callback(action)
            except Exception as e:
                logger.error(f"执行记录回调时出错: {e}")

        return True

    def playback_start(self, loop: bool = False, count: int = 1, speed: float = 1.0) -> bool:
        """
        开始回放宏操作
        
        Args:
            loop: 是否循环播放
            count: 播放次数
            speed: 播放速度倍率
            
        Returns:
            bool: 是否成功开始回放
        """
        if self._state != MacroPlaybackState.STOPPED or not self._actions:
            logger.warning("无法开始回放，宏为空或状态不允许")
            return False

        self._is_looping = loop
        self._loop_count = count if not loop else 1
        self._current_loop = 0
        self._speed_factor = max(0.1, speed)  # 最小速度0.1倍
        self._state = MacroPlaybackState.PLAYING
        self._start_time = time.time()

        # 启动播放线程
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        logger.info(f"开始回放宏操作，循环: {loop}, 次数: {count}, 速度: {speed}x")

        return True

    def playback_pause(self) -> bool:
        """
        暂停回放
        
        Returns:
            bool: 是否成功暂停
        """
        if self._state != MacroPlaybackState.PLAYING:
            return False

        self._state = MacroPlaybackState.PAUSED
        self._pause_time = time.time()
        logger.info("宏回放已暂停")
        return True

    def playback_resume(self) -> bool:
        """
        恢复回放
        
        Returns:
            bool: 是否成功恢复
        """
        if self._state != MacroPlaybackState.PAUSED):
            return False

        # 调整起始时间，保持时间连续性
        pause_duration = time.time() - self._pause_time
        self._start_time += pause_duration
        self._state = MacroPlaybackState.PLAYING
        logger.info("宏回放已恢复")
        return True

    def playback_stop(self) -> bool:
        """
        停止回放
        
        Returns:
            bool: 是否成功停止
        """
        if self._state == MacroPlaybackState.STOPPED:
            return False

        self._state = MacroPlaybackState.STOPPED
        if self._playback_thread and self._playback_thread.is_alive():
            # 线程会自行退出，无需强制终止
            pass
        logger.info("宏回放已停止")
        return True

    def _playback_worker(self):
        """播放工作线程"""
        while self._state != MacroPlaybackState.STOPPED:
            if self._current_loop >= self._loop_count and not self._is_looping:
                break

            success = self._execute_single_playback()
            if not success:
                break

            self._current_loop += 1
            if self._is_looping and self._current_loop >= self._loop_count:
                self._current_loop = 0  # 重置循环计数

            # 如果不是循环播放，则跳出
            if not self._is_looping:
                break

        self._state = MacroPlaybackState.STOPPED

    def _execute_single_playback(self) -> bool:
        """执行单次完整回放"""
        base_timestamp = time.time()
        
        for action in self._actions:
            if self._state != MacroPlaybackState.PLAYING:
                return False

            # 计算等待时间
            target_time = base_timestamp + (action.timestamp / self._speed_factor)
            sleep_time = target_time - time.time()
            
            if sleep_time > 0:
                # 分段睡眠，以便能够响应暂停/停止
                while sleep_time > 0 and self._state == MacroPlaybackState.PLAYING:
                    sleep_chunk = min(sleep_time, 0.01)  # 最大睡眠0.01秒
                    time.sleep(sleep_chunk)
                    sleep_time -= sleep_chunk

            if self._state != MacroPlaybackState.PLAYING:
                return False

            # 执行动作
            self._execute_action(action)
            
            # 触发播放回调
            for callback in self._played_callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.error(f"执行播放回调时出错: {e}")

        return True

    def _execute_action(self, action: MacroAction):
        """执行单个动作（模拟实现）"""
        logger.debug(f"执行动作: {action.action_type.value}, 参数: {action.params}")
        
        # 这里应该调用实际的系统API来执行操作
        # 由于平台限制，此处仅做模拟
        action_map = {
            MacroActionType.WAIT: self._handle_wait,
            MacroActionType.KEY_PRESS: self._handle_key_press,
            MacroActionType.KEY_RELEASE: self._handle_key_release,
            MacroActionType.MOUSE_MOVE: self._handle_mouse_move,
            MacroActionType.MOUSE_CLICK: self._handle_mouse_click,
            MacroActionType.MOUSE_SCROLL: self._handle_mouse_scroll,
            MacroActionType.CUSTOM_FUNCTION: self._handle_custom_function,
        }
        
        handler = action_map.get(action.action_type, self._handle_unknown_action)
        handler(action)

    def _handle_wait(self, action: MacroAction):
        """处理等待动作"""
        duration = action.params.get('duration', 1.0)
        time.sleep(duration / self._speed_factor)

    def _handle_key_press(self, action: MacroAction):
        """处理按键按下"""
        key = action.params.get('key', 'unknown')
        logger.info(f"按键按下: {key}")

    def _handle_key_release(self, action: MacroAction):
        """处理按键释放"""
        key = action.params.get('key', 'unknown')
        logger.info(f"按键释放: {key}")

    def _handle_mouse_move(self, action: MacroAction):
        """处理鼠标移动"""
        x = action.params.get('x', 0)
        y = action.params.get('y', 0)
        logger.info(f"鼠标移动到: ({x}, {y})")

    def _handle_mouse_click(self, action: MacroAction):
        """处理鼠标点击"""
        button = action.params.get('button', 'left')
        clicks = action.params.get('clicks', 1)
        logger.info(f"鼠标{button}键点击 {clicks} 次")

    def _handle_mouse_scroll(self, action: MacroAction):
        """处理鼠标滚轮"""
        dx = action.params.get('delta_x', 0)
        dy = action.params.get('delta_y', 0)
        logger.info(f"鼠标滚轮滚动: ({dx}, {dy})")

    def _handle_custom_function(self, action: MacroAction):
        """处理自定义函数"""
        func_name = action.params.get('function', '')
        args = action.params.get('args', [])
        kwargs = action.params.get('kwargs', {})
        logger.info(f"执行自定义函数: {func_name}, 参数: {args}, {kwargs}")

    def _handle_unknown_action(self, action: MacroAction):
        """处理未知动作"""
        logger.warning(f"不支持的动作类型: {action.action_type}")

    def register_record_callback(self, callback: Callable[[MacroAction], None]):
        """
        注册录制回调函数
        
        Args:
            callback: 回调函数，接收MacroAction参数
        """
        self._recorded_callbacks.append(callback)

    def register_playback_callback(self, callback: Callable[[MacroAction], None]):
        """
        注册播放回调函数
        
        Args:
            callback: 回调函数，接收MacroAction参数
        """
        self._played_callbacks.append(callback)

    def clear_actions(self):
        """清空所有动作"""
        self._actions.clear()
        logger.info("已清空所有宏操作")

    def get_actions(self) -> List[MacroAction]:
        """
        获取所有动作副本
        
        Returns:
            动作列表的深拷贝
        """
        return self._actions.copy()

    def insert_action(self, index: int, action: MacroAction) -> bool:
        """
        在指定位置插入动作
        
        Args:
            index: 插入位置
            action: 要插入的动作
            
        Returns:
            是否成功插入
        """
        if 0 <= index <= len(self._actions):
            self._actions.insert(index, action)
            return True
        return False

    def remove_action(self, index: int) -> bool:
        """
        移除指定位置的动作
        
        Args:
            index: 动作位置
            
        Returns:
            是否成功移除
        """
        if 0 <= index < len(self._actions):
            self._actions.pop(index)
            return True
        return False

    def save_to_file(self, filepath: str) -> bool:
        """
        保存宏到文件（模拟实现）
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功保存
        """
        try:
            # 这里应该实现序列化逻辑
            logger.info(f"宏已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存宏失败: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """
        从文件加载宏（模拟实现）
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功加载
        """
        try:
            # 这里应该实现反序列化逻辑
            self._actions.clear()
            # 模拟加载一些动作
            logger.info(f"从文件加载宏: {filepath}")
            return True
        except Exception as e:
            logger.error(f"加载宏失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        获取引擎状态
        
        Returns:
            包含状态信息的字典
        """
        return {
            'state': self._state.value,
            'action_count': len(self._actions),
            'current_loop': self._current_loop,
            'loop_count': self._loop_count,
            'is_looping': self._is_looping,
            'speed_factor': self._speed_factor,
            'recording_duration': time.time() - self._start_time if self._state == MacroPlaybackState.RECORDING else 0,
            'playback_progress': self._calculate_playback_progress()
        }

    def _calculate_playback_progress(self) -> float:
        """计算播放进度"""
        if not self._actions or self._state == MacroPlaybackState.STOPPED:
            return 0.0

        if self._state in [MacroPlaybackState.PLAYING, MacroPlaybackState.PAUSED]:
            elapsed = time.time() - self._start_time
            if self._actions:
                total_duration = self._actions[-1].timestamp
                return min(elapsed / total_duration, 1.0) if total_duration > 0 else 0.0

        return 0.0

    def cleanup(self):
        """清理资源"""
        self.playback_stop()
        self.record_stop()
        self._recorded_callbacks.clear()
        self._played_callbacks.clear()
        logger.info("宏引擎已清理")