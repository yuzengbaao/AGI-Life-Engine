#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时感知系统 - Real-Time Perception System
为AGI系统提供实时摄像头和麦克风输入

作者: AGI System Team
日期: 2025-11-21
版本: 1.0.0
"""

import cv2
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import logging
try:
    import webrtcvad
except ImportError:
    import pip
    pip.main(['install', 'webrtcvad-wheels'])
    import webrtcvad

from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class CaptureStatus(Enum):
    """捕获状态枚举"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class PerceptionConfig:
    """感知配置"""
    # 摄像头配置
    camera_device_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    frame_buffer_size: int = 60  # 保留2秒的帧
    
    # 麦克风配置
    mic_device_id: Optional[int] = None  # None表示使用默认设备
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 0.03  # WebRTC VAD requires 10, 20, or 30ms frames (0.03 = 30ms)
    audio_buffer_size: int = 1000  # Increased buffer size for smaller chunks
    
    # VAD配置
    vad_aggressiveness: int = 3  # 0-3, 3 is most aggressive in filtering non-speech
    speech_padding_ms: int = 300  # Silence padding around speech
    min_speech_duration_ms: int = 200  # Minimum duration to be considered speech
    
    # 处理配置
    process_interval: int = 5  # 每N帧处理一次(节省CPU)
    motion_threshold: float = 0.1  # 运动检测阈值
    enable_motion_detection: bool = True


class CameraCapture:
    """摄像头捕获类"""
    
    def __init__(self, config: PerceptionConfig):
        """初始化摄像头捕获
        
        Args:
            config: 感知配置对象
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.status = CaptureStatus.STOPPED
        self.frame_buffer = queue.Queue(maxsize=config.frame_buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.last_frame: Optional[np.ndarray] = None
        
        logger.info(f"📷 摄像头捕获初始化: device={config.camera_device_id}, "
                   f"resolution={config.camera_width}x{config.camera_height}, "
                   f"fps={config.camera_fps}")
    
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """列出可用的摄像头设备
        
        Returns:
            设备列表,每个设备包含id和name
        """
        devices = []
        for i in range(10):  # 检查前10个设备
            # 使用DSHOW后端检测设备(Windows)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                devices.append({
                    'id': i,
                    'name': f'Camera {i}',
                    'backend': cap.getBackendName()
                })
                cap.release()
        return devices
    
    def start(self) -> bool:
        """启动摄像头捕获
        
        Returns:
            启动成功返回True,否则False
        """
        if self.status == CaptureStatus.RUNNING:
            logger.warning("⚠️ 摄像头已在运行中")
            return True
        
        try:
            # 打开摄像头 - 使用DirectShow后端(Windows)以确保稳定访问
            # 优先尝试DSHOW后端，失败则回退到默认后端
            self.cap = cv2.VideoCapture(self.config.camera_device_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.warning(f"⚠️ DSHOW后端失败，尝试默认后端...")
                self.cap = cv2.VideoCapture(self.config.camera_device_id)
            
            if not self.cap.isOpened():
                logger.error(f"❌ 无法打开摄像头设备 {self.config.camera_device_id}")
                self.status = CaptureStatus.ERROR
                return False
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # 获取实际设置的参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"✅ 摄像头已启动: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # 启动捕获线程
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.status = CaptureStatus.RUNNING
            return True
            
        except Exception as e:
            logger.error(f"❌ 摄像头启动失败: {e}")
            self.status = CaptureStatus.ERROR
            return False
    
    def stop(self):
        """停止摄像头捕获"""
        if self.status == CaptureStatus.STOPPED:
            return
        
        logger.info("🛑 停止摄像头捕获...")
        self.stop_event.set()
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status = CaptureStatus.STOPPED
        logger.info("✅ 摄像头已停止")
    
    def _capture_loop(self):
        """捕获循环(在单独线程中运行)"""
        logger.info("🎬 摄像头捕获循环已启动")
        
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("⚠️ 读取帧失败")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                self.last_frame = frame.copy()
                
                # 将帧加入缓冲区
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()  # 丢弃最旧的帧
                    except queue.Empty:
                        pass
                
                self.frame_buffer.put({
                    'frame': frame,
                    'timestamp': datetime.now(),
                    'frame_number': self.frame_count
                })
                
                # 控制帧率
                time.sleep(1.0 / self.config.camera_fps)
                
            except Exception as e:
                logger.error(f"❌ 捕获帧时出错: {e}")
                time.sleep(0.1)
        
        logger.info("🎬 摄像头捕获循环已结束")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取一帧
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            包含frame, timestamp, frame_number的字典,超时返回None
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的帧(不从缓冲区移除)
        
        Returns:
            最新的帧,如果没有则返回None
        """
        return self.last_frame.copy() if self.last_frame is not None else None
    
    def detect_motion(self, threshold: float = None) -> bool:
        """检测是否有运动
        
        Args:
            threshold: 运动检测阈值,None使用配置值
            
        Returns:
            检测到运动返回True
        """
        if threshold is None:
            threshold = self.config.motion_threshold
        
        # 简单的帧差法检测运动
        # 实际应用中可以使用更复杂的算法
        if self.last_frame is None or self.frame_buffer.qsize() < 2:
            return False
        
        try:
            # 获取前一帧
            prev_frame_data = self.frame_buffer.queue[-2]
            prev_frame = prev_frame_data['frame']
            
            # 计算帧差
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            
            # 计算差异比例
            motion_ratio = np.sum(diff > 30) / diff.size
            
            return motion_ratio > threshold
            
        except Exception as e:
            logger.error(f"❌ 运动检测失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取捕获状态
        
        Returns:
            状态字典
        """
        return {
            'status': self.status.value,
            'frame_count': self.frame_count,
            'buffer_size': self.frame_buffer.qsize(),
            'has_frame': self.last_frame is not None,
            'config': {
                'device_id': self.config.camera_device_id,
                'resolution': f"{self.config.camera_width}x{self.config.camera_height}",
                'fps': self.config.camera_fps
            }
        }


class MicrophoneCapture:
    """麦克风捕获类"""
    
    def __init__(self, config: PerceptionConfig):
        """初始化麦克风捕获
        
        Args:
            config: 感知配置对象
        """
        self.config = config
        self.status = CaptureStatus.STOPPED
        self.audio_buffer = queue.Queue(maxsize=config.audio_buffer_size)
        self.stream: Optional[sd.InputStream] = None
        self.chunk_size = int(config.sample_rate * config.chunk_duration)
        self.current_chunk = []
        self.chunk_count = 0
        
        # VAD Initialization
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_frames = []
        
        # Calculate frame counts for timing
        # chunk_duration is in seconds (e.g. 0.03), so *1000 gives ms
        frame_ms = config.chunk_duration * 1000
        self.min_speech_frames = int(config.min_speech_duration_ms / frame_ms)
        self.padding_frames = int(config.speech_padding_ms / frame_ms)
        
        logger.info(f"🎤 麦克风捕获初始化: sample_rate={config.sample_rate}Hz, "
                   f"channels={config.channels}, chunk={config.chunk_duration}s")
    
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        """列出可用的音频设备
        
        Returns:
            设备列表
        """
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
        except Exception as e:
            logger.error(f"❌ 查询音频设备失败: {e}")
        
        return devices
    
    def start(self) -> bool:
        """启动麦克风捕获
        
        Returns:
            启动成功返回True,否则False
        """
        if self.status == CaptureStatus.RUNNING:
            logger.warning("⚠️ 麦克风已在运行中")
            return True
        
        try:
            # 创建音频流
            # 增大blocksize减少回调频率，避免input overflow
            # blocksize=8192相当于512ms@16kHz，给系统更多处理时间
            self.stream = sd.InputStream(
                device=self.config.mic_device_id,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                callback=self._audio_callback,
                blocksize=8192,  # 从4096增加到8192，进一步减少overflow风险
                latency='high'   # 使用高延迟模式，优先稳定性
            )
            
            self.stream.start()
            self.status = CaptureStatus.RUNNING
            
            logger.info(f"✅ 麦克风已启动: {self.config.sample_rate}Hz, "
                       f"{self.config.channels}声道, blocksize=8192")
            return True
            
        except Exception as e:
            logger.error(f"❌ 麦克风启动失败: {e}")
            self.status = CaptureStatus.ERROR
            return False
    
    def stop(self):
        """停止麦克风捕获"""
        if self.status == CaptureStatus.STOPPED:
            return
        
        logger.info("🛑 停止麦克风捕获...")
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.status = CaptureStatus.STOPPED
        logger.info("✅ 麦克风已停止")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数(由sounddevice调用)
        
        Args:
            indata: 输入音频数据
            frames: 帧数
            time_info: 时间信息
            status: 状态标志位
        """
        if status:
            # logger.warning(f"⚠️ 音频状态: {status}")
            pass
            
        try:
            # 1. Convert to 16-bit PCM (required by WebRTC VAD)
            # indata is float32 [-1.0, 1.0], convert to int16 [-32768, 32767]
            audio_int16 = (indata * 32768).astype(np.int16)
            
            # 2. Check for speech using VAD
            is_speech = False
            try:
                # WebRTC VAD only supports 16000Hz (and 8k, 32k, 48k) mono 16-bit PCM
                # Ensure input is mono
                if self.config.channels > 1:
                    audio_mono = audio_int16.mean(axis=1).astype(np.int16)
                    raw_bytes = audio_mono.tobytes()
                else:
                    raw_bytes = audio_int16.tobytes()
                    
                is_speech = self.vad.is_speech(raw_bytes, self.config.sample_rate)
            except Exception as e:
                # Fallback if VAD fails
                pass
            
            # 3. Speech Logic with Padding
            if is_speech:
                self.is_speech_active = True
                self.silence_counter = 0
                self.speech_frames.append(indata.copy())
            elif self.is_speech_active:
                # Currently in speech mode, but silence detected
                self.silence_counter += 1
                self.speech_frames.append(indata.copy())
                
                # If silence exceeds padding, stop speech segment
                if self.silence_counter > self.padding_frames:
                    self.is_speech_active = False
                    
                    # Only process if duration is long enough
                    if len(self.speech_frames) >= self.min_speech_frames:
                        # Concatenate all speech frames
                        full_speech = np.concatenate(self.speech_frames, axis=0)
                        
                        # Add to buffer
                        if self.audio_buffer.full():
                            try:
                                self.audio_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.audio_buffer.put({
                            'audio': full_speech,
                            'timestamp': datetime.now(),
                            'chunk_number': self.chunk_count,
                            'sample_rate': self.config.sample_rate,
                            'channels': self.config.channels,
                            'is_speech': True
                        })
                        self.chunk_count += 1
                    
                    # Reset
                    self.speech_frames = []
                    self.silence_counter = 0
            
            # If not speech and not active, do nothing (filter out noise)
            
        except Exception as e:
            logger.error(f"音频回调错误: {e}")
    
    def get_audio_chunk(self, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """获取一块音频数据
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            包含audio, timestamp, chunk_number等信息的字典
        """
        try:
            return self.audio_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取捕获状态
        
        Returns:
            状态字典
        """
        return {
            'status': self.status.value,
            'chunk_count': self.chunk_count,
            'buffer_size': self.audio_buffer.qsize(),
            'config': {
                'device_id': self.config.mic_device_id,
                'sample_rate': self.config.sample_rate,
                'channels': self.config.channels,
                'chunk_duration': self.config.chunk_duration
            }
        }


class PerceptionManager:
    """感知管理器 - 统一管理摄像头和麦克风"""
    
    def __init__(self, config: Optional[PerceptionConfig] = None):
        """初始化感知管理器
        
        Args:
            config: 感知配置,None使用默认配置
        """
        self.config = config or PerceptionConfig()
        self.camera = CameraCapture(self.config)
        self.microphone = MicrophoneCapture(self.config)
        
        # 处理回调
        self.video_processor: Optional[Callable] = None
        self.audio_processor: Optional[Callable] = None
        
        # 处理线程
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = threading.Event()
        
        logger.info("🎯 感知管理器初始化完成")
    
    def set_video_processor(self, processor: Callable):
        """设置视频处理器回调
        
        Args:
            processor: 处理函数,接收frame参数
        """
        self.video_processor = processor
        logger.info("✅ 视频处理器已设置")
    
    def set_audio_processor(self, processor: Callable):
        """设置音频处理器回调
        
        Args:
            processor: 处理函数,接收audio_data参数
        """
        self.audio_processor = processor
        logger.info("✅ 音频处理器已设置")
    
    def start_camera(self) -> bool:
        """启动摄像头"""
        return self.camera.start()
    
    def stop_camera(self):
        """停止摄像头"""
        self.camera.stop()
    
    def start_microphone(self) -> bool:
        """启动麦克风"""
        return self.microphone.start()
    
    def stop_microphone(self):
        """停止麦克风"""
        self.microphone.stop()
    
    def start_all(self) -> Dict[str, bool]:
        """启动所有感知设备
        
        Returns:
            启动结果字典
        """
        results = {
            'camera': self.start_camera(),
            'microphone': self.start_microphone()
        }
        
        # 启动处理线程
        if any(results.values()):
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop, 
                daemon=True
            )
            self.processing_thread.start()
        
        return results
    
    def stop_all(self):
        """停止所有感知设备"""
        logger.info("🛑 停止所有感知设备...")
        
        # 停止处理线程
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.stop_camera()
        self.stop_microphone()
        
        logger.info("✅ 所有感知设备已停止")
    
    def _processing_loop(self):
        """处理循环(在单独线程中运行)"""
        logger.info("⚙️ 感知处理循环已启动")
        
        frame_counter = 0
        
        while not self.stop_processing.is_set():
            try:
                # 处理视频帧
                if (self.camera.status == CaptureStatus.RUNNING and 
                    self.video_processor):
                    
                    frame_counter += 1
                    
                    # 按间隔处理(节省CPU)
                    if frame_counter % self.config.process_interval == 0:
                        frame_data = self.camera.get_frame(timeout=0.1)
                        if frame_data:
                            try:
                                self.video_processor(frame_data)
                            except Exception as e:
                                logger.error(f"❌ 视频处理失败: {e}")
                
                # 处理音频块
                if (self.microphone.status == CaptureStatus.RUNNING and 
                    self.audio_processor):
                    
                    audio_data = self.microphone.get_audio_chunk(timeout=0.1)
                    if audio_data:
                        try:
                            self.audio_processor(audio_data)
                        except Exception as e:
                            logger.error(f"❌ 音频处理失败: {e}")
                
                time.sleep(0.01)  # 避免CPU占用过高
                
            except Exception as e:
                logger.error(f"❌ 处理循环出错: {e}")
                time.sleep(0.1)
        
        logger.info("⚙️ 感知处理循环已结束")
    
    def is_camera_running(self) -> bool:
        """检查摄像头是否运行中
        
        Returns:
            True表示运行中，False表示停止
        """
        return self.camera.status == CaptureStatus.RUNNING
    
    def is_microphone_running(self) -> bool:
        """检查麦克风是否运行中
        
        Returns:
            True表示运行中，False表示停止
        """
        return self.microphone.status == CaptureStatus.RUNNING
    
    def get_status(self) -> Dict[str, Any]:
        """获取整体状态
        
        Returns:
            状态字典
        """
        return {
            'camera': self.camera.get_status(),
            'microphone': self.microphone.get_status(),
            'processing_active': not self.stop_processing.is_set()
        }
    
    @staticmethod
    def list_devices() -> Dict[str, List[Dict[str, Any]]]:
        """列出所有可用设备
        
        Returns:
            设备列表字典
        """
        return {
            'cameras': CameraCapture.list_devices(),
            'microphones': MicrophoneCapture.list_devices()
        }


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("实时感知系统测试")
    print("=" * 60)
    
    # 列出设备
    print("\n📋 可用设备:")
    devices = PerceptionManager.list_devices()
    
    print("\n📷 摄像头设备:")
    for cam in devices['cameras']:
        print(f"  - ID {cam['id']}: {cam['name']} ({cam['backend']})")
    
    print("\n🎤 麦克风设备:")
    for mic in devices['microphones']:
        print(f"  - ID {mic['id']}: {mic['name']} "
              f"({mic['channels']}ch @ {mic['sample_rate']}Hz)")
    
    # 创建管理器
    print("\n" + "=" * 60)
    print("创建感知管理器...")
    manager = PerceptionManager()
    
    # 设置简单的处理器
    def simple_video_processor(frame_data):
        print(f"📹 处理帧 #{frame_data['frame_number']}, "
              f"时间: {frame_data['timestamp']}")
    
    def simple_audio_processor(audio_data):
        print(f"🎵 处理音频块 #{audio_data['chunk_number']}, "
              f"时间: {audio_data['timestamp']}, "
              f"大小: {audio_data['audio'].shape}")
    
    manager.set_video_processor(simple_video_processor)
    manager.set_audio_processor(simple_audio_processor)
    
    # 启动设备
    print("\n启动感知设备...")
    results = manager.start_all()
    print(f"启动结果: {results}")
    
    # 运行10秒
    print("\n运行10秒...")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\n用户中断")
    
    # 获取状态
    print("\n📊 当前状态:")
    status = manager.get_status()
    print(f"摄像头: {status['camera']}")
    print(f"麦克风: {status['microphone']}")
    
    # 停止
    print("\n停止所有设备...")
    manager.stop_all()
    
    print("\n✅ 测试完成")
