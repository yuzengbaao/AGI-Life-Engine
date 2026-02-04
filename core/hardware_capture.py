#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件采集模块
Hardware Capture Module

功能：
1. 摄像头实时采集
2. 麦克风实时采集
3. 多线程异步采集
4. 资源管理和错误处理

Author: AGI System Development Team
Date: 2026-01-26
Version: 1.0.0
"""

import cv2
import sounddevice as sd
import numpy as np
import threading
import queue
import logging
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum


class CaptureDevice(Enum):
    """采集设备类型"""
    CAMERA = "camera"
    MICROPHONE = "microphone"


@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class MicrophoneConfig:
    """麦克风配置"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024


class CameraCapture:
    """摄像头采集器"""
    
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self.cap = None
        self.is_running = False
        self.logger = logging.getLogger("CameraCapture")
        
    def start(self):
        """启动摄像头"""
        if self.is_running:
            self.logger.warning("摄像头已经在运行中")
            return False
            
        try:
            self.cap = cv2.VideoCapture(self.config.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开摄像头 {self.config.camera_id}")
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            self.is_running = True
            self.logger.info(f"摄像头启动成功: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"摄像头启动失败: {e}")
            return False
    
    def capture_frame(self):
        """捕获一帧图像"""
        if not self.is_running or self.cap is None:
            self.logger.warning("摄像头未启动")
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("无法读取摄像头帧")
            return None
            
        return frame
    
    def stop(self):
        """停止摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_running = False
            self.logger.info("摄像头已停止")


class MicrophoneCapture:
    """麦克风采集器"""
    
    def __init__(self, config: MicrophoneConfig = None):
        self.config = config or MicrophoneConfig()
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.logger = logging.getLogger("MicrophoneCapture")
        
    def start(self):
        """启动麦克风"""
        if self.is_running:
            self.logger.warning("麦克风已经在运行中")
            return False
            
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
            self.logger.info(f"麦克风启动成功: {self.config.sample_rate}Hz, {self.config.channels}ch")
            return True
            
        except Exception as e:
            self.logger.error(f"麦克风启动失败: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            self.logger.warning(f"音频流状态: {status}")
        self.audio_queue.put(indata.copy())
    
    def capture_audio(self, duration: float = 1.0):
        """捕获指定时长的音频"""
        if not self.is_running:
            self.logger.warning("麦克风未启动")
            return None
            
        frames_needed = int(duration * self.config.sample_rate)
        audio_data = []
        total_frames = 0
        
        while total_frames < frames_needed:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                audio_data.append(chunk)
                total_frames += len(chunk)
            except queue.Empty:
                self.logger.warning("音频队列超时")
                break
                
        if audio_data:
            audio = np.concatenate(audio_data, axis=0)
            return audio[:frames_needed]
        
        return None
    
    def stop(self):
        """停止麦克风"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_running = False
            self.logger.info("麦克风已停止")


class HardwareCaptureManager:
    """硬件采集管理器"""
    
    def __init__(
        self,
        camera_config: CameraConfig = None,
        microphone_config: MicrophoneConfig = None,
        enable_camera: bool = True,
        enable_microphone: bool = True
    ):
        self.camera_config = camera_config or CameraConfig()
        self.microphone_config = microphone_config or MicrophoneConfig()
        
        self.camera = CameraCapture(self.camera_config) if enable_camera else None
        self.microphone = MicrophoneCapture(self.microphone_config) if enable_microphone else None
        
        self.logger = logging.getLogger("HardwareCaptureManager")
    
    def start_all(self):
        """启动所有设备"""
        success = True
        
        if self.camera:
            if not self.camera.start():
                success = False
                
        if self.microphone:
            if not self.microphone.start():
                success = False
                
        if success:
            self.logger.info("所有硬件采集设备启动成功")
        else:
            self.logger.warning("部分硬件采集设备启动失败")
            
        return success
    
    def stop_all(self):
        """停止所有设备"""
        if self.camera:
            self.camera.stop()
            
        if self.microphone:
            self.microphone.stop()
            
        self.logger.info("所有硬件采集设备已停止")
    
    def capture_frame(self):
        """捕获一帧图像"""
        if self.camera:
            return self.camera.capture_frame()
        return None
    
    def capture_audio(self, duration: float = 1.0):
        """捕获音频"""
        if self.microphone:
            return self.microphone.capture_audio(duration)
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        status = {
            "camera": {
                "enabled": self.camera is not None,
                "running": self.camera.is_running if self.camera else False,
                "config": {
                    "camera_id": self.camera_config.camera_id,
                    "width": self.camera_config.width,
                    "height": self.camera_config.height,
                    "fps": self.camera_config.fps
                } if self.camera else None
            },
            "microphone": {
                "enabled": self.microphone is not None,
                "running": self.microphone.is_running if self.microphone else False,
                "config": {
                    "sample_rate": self.microphone_config.sample_rate,
                    "channels": self.microphone_config.channels,
                    "chunk_size": self.microphone_config.chunk_size
                } if self.microphone else None
            }
        }
        return status


def test_hardware_capture():
    """测试硬件采集功能"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("测试硬件采集功能")
    print("=" * 70)
    
    manager = HardwareCaptureManager(
        camera_config=CameraConfig(camera_id=0, width=640, height=480, fps=30),
        microphone_config=MicrophoneConfig(sample_rate=16000, channels=1),
        enable_camera=True,
        enable_microphone=True
    )
    
    if not manager.start_all():
        print("❌ 硬件采集设备启动失败")
        return
    
    print("\n设备状态:")
    status = manager.get_status()
    print(f"摄像头: {'✅ 运行中' if status['camera']['running'] else '❌ 未运行'}")
    print(f"麦克风: {'✅ 运行中' if status['microphone']['running'] else '❌ 未运行'}")
    
    print("\n测试摄像头采集...")
    frame = manager.capture_frame()
    if frame is not None:
        print(f"✅ 成功捕获图像: {frame.shape}")
    else:
        print("❌ 摄像头采集失败")
    
    print("\n测试麦克风采集...")
    audio = manager.capture_audio(duration=2.0)
    if audio is not None:
        print(f"✅ 成功捕获音频: {audio.shape}")
    else:
        print("❌ 麦克风采集失败")
    
    print("\n等待3秒后关闭...")
    time.sleep(3)
    
    manager.stop_all()
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_hardware_capture()
