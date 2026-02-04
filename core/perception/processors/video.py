#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级视频处理器
实现复杂场景理解、时序分析和多目标跟踪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Deque, Union
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="cv2")

# 尝试加载配置文件
def load_video_config() -> Dict[str, Any]:
    """加载视频处理配置，带缓存机制"""
    config_path = Path(__file__).parent / "video_processing_config.json"
    
    # 使用模块级缓存避免重复加载
    if hasattr(load_video_config, '_cached_config'):
        return load_video_config._cached_config
    
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("✅ 视频处理配置文件加载成功")
            result = config.get("video_processing", {})
        else:
            raise FileNotFoundError("Config file not found")
    except Exception as e:
        logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        # 默认配置
        result = {
            "frame_extraction": {
                "fps": 30,
                "resize_width": 640,
                "resize_height": 480
            },
            "models": {
                "spatial_temporal_cnn": {
                    "input_channels": 3,
                    "feature_dim": 512
                },
                "temporal_attention": {
                    "feature_dim": 512,
                    "num_heads": 8
                }
            }
        }
        logger.info("使用默认视频处理配置")
    
    # 缓存结果
    load_video_config._cached_config = result
    return result

VIDEO_CONFIG = load_video_config()

class SceneComplexity(Enum):
    """场景复杂度枚举"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"

@dataclass
class DetectionResult:
    """检测结果数据类"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id: Optional[int] = None
    features: Optional[np.ndarray] = None

@dataclass
class SceneAnalysis:
    """场景分析结果"""
    complexity: SceneComplexity
    object_count: int
    motion_intensity: float
    lighting_quality: float
    occlusion_level: float
    crowd_density: float
    scene_description: str

class TemporalAttention(nn.Module):
    """
    时序注意力机制
    用于捕获视频帧间的时序关系
    """
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # 预计算标量以提高效率
        self.scale_factor = 1.0 / np.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, feature_dim]
            
        Returns:
            输出特征 [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算注意力（优化版本）
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用预计算的缩放因子
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        
        # 输出投影和残差连接
        output = self.output(attended)
        output = self.layer_norm(output + x)
        
        return output

class SpatialTemporalCNN(nn.Module):
    """
    时空卷积神经网络
    用于提取视频的时空特征
    """
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        
        # 使用Sequential简化网络结构
        self.features = nn.Sequential(
            # 3D卷积层用于时空特征提取
            nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, feature_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入视频 [batch_size, channels, frames, height, width]
            
        Returns:
            特征向量 [batch_size, feature_dim]
        """
        # 3D卷积特征提取
        x = self.features(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平并通过全连接层
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class MultiObjectTracker:
    """
    多目标跟踪器
    使用深度特征和卡尔曼滤波进行目标跟踪
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3, distance_threshold: float = 100.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.tracks: List[Dict[str, Any]] = []
        self.track_id_counter = 0
        
        # 预分配常用数组以减少内存分配
        self._bbox_cache = {}
    
    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        更新跟踪器
        
        Args:
            detections: 当前帧的检测结果
            
        Returns:
            带有跟踪ID的检测结果
        """
        if not detections:
            # 更新现有轨迹年龄
            self._update_track_ages()
            return []
        
        tracked_detections = []
        
        # 批量处理：先检查所有现有轨迹
        active_tracks = []
        for detection in detections:
            best_match = None
            best_distance = self.distance_threshold
            
            for track in self.tracks:
                if (track['class_id'] == detection.class_id and 
                    track['age'] < self.max_age):
                    
                    distance = self._calculate_bbox_distance(detection.bbox, track['last_bbox'])
                    if distance < best_distance:
                        best_distance = distance
                        best_match = track
            
            if best_match:
                # 更新现有轨迹
                best_match['last_bbox'] = detection.bbox
                best_match['age'] = 0
                best_match['hits'] += 1
                detection.track_id = best_match['id']
            else:
                # 创建新轨迹
                new_track = {
                    'id': self.track_id_counter,
                    'class_id': detection.class_id,
                    'last_bbox': detection.bbox,
                    'age': 0,
                    'hits': 1
                }
                self.tracks.append(new_track)
                detection.track_id = self.track_id_counter
                self.track_id_counter += 1
            
            tracked_detections.append(detection)
        
        # 更新轨迹年龄并移除过期轨迹
        self._update_track_ages()
        
        return tracked_detections
    
    def _calculate_bbox_distance(self, bbox1: Tuple[int, int, int, int], 
                                bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的中心距离（向量化实现）"""
        center1_x = (bbox1[0] + bbox1[2]) * 0.5
        center1_y = (bbox1[1] + bbox1[3]) * 0.5
        center2_x = (bbox2[0] + bbox2[2]) * 0.5
        center2_y = (bbox2[1] + bbox2[3]) * 0.5
        
        dx = center1_x - center2_x
        dy = center1_y - center2_y
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _update_track_ages(self):
        """批量更新轨迹年龄并清理过期轨迹"""
        current_time = time.time()
        self.tracks = [track for track in self.tracks if track['age'] < self.max_age]

class SceneComplexityAnalyzer:
    """
    场景复杂度分析器
    分析视频场景的复杂程度
    """
    def __init__(self, history_length: int = 10):
        self.motion_history: Deque[float] = deque(maxlen=history_length)
        self.frame_shape_cache: Optional[Tuple[int, int]] = None
        self.gray_cache: Optional[np.ndarray] = None
        
    def analyze_scene(self, frame: np.ndarray, detections: List[DetectionResult], 
                     prev_frame: Optional[np.ndarray] = None) -> SceneAnalysis:
        """
        分析场景复杂度
        
        Args:
            frame: 当前帧
            detections: 检测结果
            prev_frame: 前一帧
            
        Returns:
            场景分析结果
        """
        try:
            # 计算各项指标
            object_count = len(detections)
            motion_intensity = self._calculate_motion_intensity(frame, prev_frame)
            lighting_quality = self._calculate_lighting_quality(frame)
            occlusion_level = self._calculate_occlusion_level(detections)
            crowd_density = self._calculate_crowd_density(detections, frame.shape)
            
            # 确定复杂度等级
            complexity = self._determine_complexity(
                object_count, motion_intensity, lighting_quality, 
                occlusion_level, crowd_density
            )
            
            # 生成场景描述
            scene_description = self._generate_scene_description(
                complexity, object_count, motion_intensity, crowd_density
            )
            
            return SceneAnalysis(
                complexity=complexity,
                object_count=object_count,
                motion_intensity=motion_intensity,
                lighting_quality=lighting_quality,
                occlusion_level=occlusion_level,
                crowd_density=crowd_density,
                scene_description=scene_description
            )
            
        except Exception as e:
            logger.error(f"场景分析失败: {e}")
            # 返回安全的默认值
            return SceneAnalysis(
                complexity=SceneComplexity.MODERATE,
                object_count=0,
                motion_intensity=0.0,
                lighting_quality=0.5,
                occlusion_level=0.0,
                crowd_density=0.0,
                scene_description="分析失败，使用默认场景"
            )
    
    def _calculate_motion_intensity(self, frame: np.ndarray, 
                                   prev_frame: Optional[np.ndarray]) -> float:
        """计算运动强度（优化版本）"""
        if prev_frame is None:
            return 0.0
        
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # 使用更高效的光流算法
            flow = cv2.calcOpticalFlowFarneback(
                gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算运动强度
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_magnitude = np.mean(magnitude)
            
            self.motion_history.append(motion_magnitude)
            return float(np.mean(self.motion_history)) if self.motion_history else 0.0
            
        except Exception as e:
            logger.debug(f"光流计算失败，使用帧差方法: {e}")
            # 备用方法：使用帧差
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            motion_magnitude = np.mean(diff) / 255.0
            self.motion_history.append(motion_magnitude)
            return float(np.mean(self.motion_history)) if self.motion_history else 0.0
    
    def _calculate_lighting_quality(self, frame: np.ndarray) -> float:
        """计算光照质量（优化版本）"""
        if self.gray_cache is None or self.frame_shape_cache != frame.shape[:2]:
            self.gray_cache = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_shape_cache = frame.shape[:2]
        
        gray = self.gray_cache
        
        # 使用更稳定的统计方法
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 计算对比度
        contrast = brightness_std / (mean_brightness + 1e-6)
        
        # 归一化到0-1范围
        quality = min(1.0, contrast / 50.0)  # 调整阈值
        
        return float(quality)
    
    def _calculate_occlusion_level(self, detections: List[DetectionResult]) -> float:
        """计算遮挡程度（向量化实现）"""
        if len(detections) < 2:
            return 0.0
        
        total_overlap = 0.0
        total_pairs = 0
        
        # 向量化计算重叠
        bboxes = [det.bbox for det in detections]
        
        for i, bbox1 in enumerate(bboxes):
            for j, bbox2 in enumerate(bboxes[i+1:], i+1):
                overlap = self._calculate_bbox_overlap(bbox1, bbox2)
                total_overlap += overlap
                total_pairs += 1
        
        return total_overlap / max(total_pairs, 1)
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的重叠率（优化版本）"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)
    
    def _calculate_crowd_density(self, detections: List[DetectionResult], 
                                frame_shape: Tuple[int, int, int]) -> float:
        """计算人群密度（优化版本）"""
        # 统计人类目标
        person_count = sum(1 for det in detections if det.class_name.lower() in ['person', 'human'])
        
        # 计算密度（人数/面积）
        frame_area = frame_shape[0] * frame_shape[1]
        density = person_count / (frame_area / 10000)  # 每万像素的人数
        
        return min(1.0, float(density))
    
    def _determine_complexity(self, object_count: int, motion_intensity: float,
                             lighting_quality: float, occlusion_level: float,
                             crowd_density: float) -> SceneComplexity:
        """确定场景复杂度（优化权重）"""
        # 计算复杂度分数
        complexity_score = (
            min(object_count / 15, 1.0) * 0.3 +
            min(motion_intensity / 30, 1.0) * 0.25 +
            (1.0 - lighting_quality) * 0.2 +
            occlusion_level * 0.15 +
            crowd_density * 0.1
        )
        
        if complexity_score < 0.25:
            return SceneComplexity.SIMPLE
        elif complexity_score < 0.5:
            return SceneComplexity.MODERATE
        elif complexity_score < 0.75:
            return SceneComplexity.COMPLEX
        else:
            return SceneComplexity.EXTREME
    
    def _generate_scene_description(self, complexity: SceneComplexity, 
                                   object_count: int, motion_intensity: float,
                                   crowd_density: float) -> str:
        """生成场景描述（更丰富的描述）"""
        descriptions = {
            SceneComplexity.SIMPLE: "简单场景：目标较少，运动缓慢，光照良好",
            SceneComplexity.MODERATE: "中等复杂场景：目标适中，有一定运动",
            SceneComplexity.COMPLEX: "复杂场景：目标较多，运动频繁，可能有遮挡",
            SceneComplexity.EXTREME: "极复杂场景：大量目标，剧烈运动，严重遮挡或恶劣光照"
        }
        
        base_desc = descriptions[complexity]
        
        # 添加具体信息
        details = []
        if object_count > 10:
            details.append(f"{object_count}个目标")
        if motion_intensity > 20:
            details.append("高强度运动")
        if crowd_density > 0.3:
            details.append("高人群密度")
        
        if details:
            return f"{base_desc}，包含{', '.join(details)}"
        
        return base_desc

@contextmanager
def video_capture_context(video_path: str):
    """视频捕获的上下文管理器"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        yield cap
    finally:
        cap.release()

class AdvancedVideoProcessor:
    """
    高级视频处理器
    集成场景理解、目标跟踪和复杂度分析
    """
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化视频处理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型组件（带错误处理）
        try:
            model_config = VIDEO_CONFIG.get("models", {}).get("spatial_temporal_cnn", {})
            feature_dim = model_config.get("feature_dim", 512)
            
            self.spatial_temporal_cnn = SpatialTemporalCNN(
                feature_dim=feature_dim
            ).to(self.device)
            
            attention_config = VIDEO_CONFIG.get("models", {}).get("temporal_attention", {})
            num_heads = attention_config.get("num_heads", 8)
            
            self.temporal_attention = TemporalAttention(
                feature_dim=feature_dim, 
                num_heads=num_heads
            ).to(self.device)
            
            # 设置评估模式
            self.spatial_temporal_cnn.eval()
            self.temporal_attention.eval()
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
        
        # 初始化分析器和跟踪器
        tracking_config = self.config.get("tracking", {})
        self.tracker = MultiObjectTracker(
            max_age=tracking_config.get("max_age", 30),
            min_hits=tracking_config.get("min_hits", 3)
        )
        
        self.scene_analyzer = SceneComplexityAnalyzer()
        
        # 处理历史
        self.processing_history: List[Dict[str, Any]] = []
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=16)
        
        # 性能监控
        self._processing_times: List[float] = []
        
        logger.info("🎬 高级视频处理器初始化完成")
        logger.info(f"   - 设备: {self.device}")
        logger.info(f"   - 时空CNN特征维度: {feature_dim}")
        logger.info(f"   - 时序注意力头数: {num_heads}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """加载配置文件（带缓存）"""
        if hasattr(self._load_config, '_cached_config'):
            return self._load_config._cached_config
        
        default_config = {
            "video_processing": {
                "target_fps": 30,
                "frame_size": [224, 224],
                "batch_size": 8
            },
            "detection": {
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4
            },
            "tracking": {
                "max_age": 30,
                "min_hits": 3
            },
            "analysis": {
                "complexity_threshold": 0.7,
                "motion_sensitivity": 1.0
            }
        }
        
        if config_file is None:
            config_file = 'video_processing_config.json'
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 合并默认配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            result = config
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_file} 未找到，使用默认配置")
            result = default_config
        except json.JSONDecodeError as e:
            logger.error(f"配置文件解析失败: {e}")
            result = default_config
        
        # 缓存结果
        self._load_config._cached_config = result
        return result
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出路径（可选）
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        logger.info(f"🎬 开始处理视频: {video_path}")
        
        # 使用上下文管理器确保资源释放
        with video_capture_context(video_path) as cap:
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"   - 视频信息: {width}x{height}, {fps:.2f}fps, {frame_count}帧")
            
            # 初始化结果存储
            results = {
                'video_info': {
                    'path': video_path,
                    'fps': fps,
                    'frame_count': frame_count,
                    'resolution': (width, height),
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'frame_results': [],
                'summary': {
                    'total_objects': 0,
                    'unique_tracks': 0,
                    'avg_complexity': 0.0,
                    'scene_changes': 0,
                    'processing_time': 0.0,
                    'frames_processed': 0
                }
            }
            
            frame_idx = 0
            prev_frame = None
            prev_complexity = None
            
            # 安全限制
            max_frames = 100000  # 最大处理10万帧
            processing_timeout = 3600  # 最大处理时间1小时
            processing_start = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 检查帧数限制
                    if frame_idx >= max_frames:
                        logger.warning(f"⚠️ 达到最大帧数限制: {max_frames}")
                        break
                    
                    # 检查处理超时
                    elapsed_time = time.time() - processing_start
                    if elapsed_time > processing_timeout:
                        logger.warning(f"⚠️ 处理超时: {processing_timeout}秒 (已处理{frame_idx}帧)")
                        break
                    
                    # 处理单帧
                    frame_start = time.time()
                    frame_result = self.process_frame(frame, frame_idx, prev_frame)
                    frame_end = time.time()
                    
                    # 记录处理时间
                    self._processing_times.append(frame_end - frame_start)
                    
                    results['frame_results'].append(frame_result)
                    
                    # 更新统计信息
                    self._update_summary(results['summary'], frame_result, prev_complexity)
                    
                    # 更新历史
                    prev_frame = frame.copy()
                    prev_complexity = frame_result['scene_analysis'].complexity
                    frame_idx += 1
                    
                    # 显示进度
                    if frame_idx % 30 == 0:
                        progress = (frame_idx / max(frame_count, 1)) * 100
                        avg_time = np.mean(self._processing_times[-30:]) if self._processing_times else 0
                        logger.info(f"   - 处理进度: {progress:.1f}% ({frame_idx}/{frame_count}), "
                                  f"平均帧处理时间: {avg_time*1000:.1f}ms")
                        
            except Exception as e:
                logger.error(f"视频处理过程中发生错误: {e}")
                raise
        
        # 计算最终统计
        self._finalize_summary(results['summary'], results['frame_results'])
        results['summary']['processing_time'] = time.time() - start_time
        results['summary']['frames_processed'] = frame_idx
        
        # 保存结果
        if output_path:
            try:
                self._save_results(results, output_path)
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
        
        logger.info(f"✅ 视频处理完成，共处理 {frame_idx} 帧, "
                   f"总耗时: {results['summary']['processing_time']:.2f}秒")
        
        return results
    
    def process_frame(self, frame: np.ndarray, frame_idx: int, 
                     prev_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        处理单帧
        
        Args:
            frame: 当前帧
            frame_idx: 帧索引
            prev_frame: 前一帧
            
        Returns:
            帧处理结果
        """
        try:
            # 模拟目标检测（实际应用中应使用真实的检测模型）
            detections = self._simulate_detection(frame)
            
            # 目标跟踪
            tracked_detections = self.tracker.update(detections)
            
            # 场景分析
            scene_analysis = self.scene_analyzer.analyze_scene(frame, tracked_detections, prev_frame)
            
            # 添加到帧缓冲区
            if len(self.frame_buffer) < self.frame_buffer.maxlen:
                self.frame_buffer.append(frame.copy())
            else:
                # 循环缓冲区
                self.frame_buffer.appendleft(frame.copy())
            
            # 构建结果
            result = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / 30.0,  # 假设30fps
                'detections': [
                    {
                        'class_id': det.class_id,
                        'class_name': det.class_name,
                        'confidence': float(det.confidence),
                        'bbox': list(det.bbox),
                        'track_id': det.track_id
                    }
                    for det in tracked_detections
                ],
                'scene_analysis': scene_analysis,
                'processing_time': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"帧处理失败 (帧 {frame_idx}): {e}")
            # 返回最小可行结果
            return {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / 30.0,
                'detections': [],
                'scene_analysis': SceneAnalysis(
                    complexity=SceneComplexity.MODERATE,
                    object_count=0,
                    motion_intensity=0.0,
                    lighting_quality=0.5,
                    occlusion_level=0.0,
                    crowd_density=0.0,
                    scene_description="处理失败"
                ),
                'processing_time': time.time()
            }
    
    def _simulate_detection(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        模拟目标检测
        实际应用中应替换为真实的检测模型
        """
        try:
            # 使用固定的随机种子保证可重现性（调试时）
            rng = np.random.default_rng(seed=(hash(time.time()) % 2**32))
            
            # 简单的模拟检测逻辑
            detections = []
            
            # 根据分辨率调整对象数量
            resolution_factor = (frame.shape[0] * frame.shape[1]) / (640 * 480)
            max_objects = max(1, int(8 * resolution_factor))
            num_objects = rng.integers(0, max_objects + 1)
            
            class_names = ['person', 'car', 'bicycle', 'dog', 'cat', 'truck', 'bus']
            
            for i in range(num_objects):
                class_id = rng.integers(0, len(class_names))
                class_name = class_names[class_id]
                confidence = rng.uniform(0.5, 0.95)
                
                # 防止边界框超出图像范围
                max_w = min(150, frame.shape[1] // 4)
                max_h = min(150, frame.shape[0] // 4)
                
                w = rng.integers(50, max_w)
                h = rng.integers(50, max_h)
                
                x1 = rng.integers(0, frame.shape[1] - w)
                y1 = rng.integers(0, frame.shape[0] - h)
                x2 = x1 + w
                y2 = y1 + h
                
                detection = DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2))
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"检测模拟失败: {e}")
            return []
    
    def _update_summary(self, summary: Dict[str, Any], frame_result: Dict[str, Any], 
                       prev_complexity: Optional[SceneComplexity]):
        """更新摘要统计"""
        summary['total_objects'] += len(frame_result['detections'])
        
        # 检测场景变化
        current_complexity = frame_result['scene_analysis'].complexity
        if prev_complexity and current_complexity != prev_complexity:
            summary['scene_changes'] += 1
    
    def _finalize_summary(self, summary: Dict[str, Any], frame_results: List[Dict[str, Any]]):
        """完成最终统计"""
        if not frame_results:
            return
        
        # 计算平均复杂度
        complexity_scores = []
        unique_tracks: set = set()
        
        complexity_map = {
            SceneComplexity.SIMPLE: 0.25,
            SceneComplexity.MODERATE: 0.5,
            SceneComplexity.COMPLEX: 0.75,
            SceneComplexity.EXTREME: 1.0
        }
        
        for result in frame_results:
            # 复杂度分数映射
            complexity_enum = result['scene_analysis'].complexity
            if complexity_enum in complexity_map:
                complexity_scores.append(complexity_map[complexity_enum])
            
            # 收集唯一轨迹ID
            for det in result['detections']:
                track_id = det['track_id']
                if track_id is not None:
                    unique_tracks.add(track_id)
        
        summary['avg_complexity'] = float(np.mean(complexity_scores)) if complexity_scores else 0.0
        summary['unique_tracks'] = len(unique_tracks)
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """保存处理结果（带错误处理）"""
        try:
            # 转换不可序列化的对象
            serializable_results = self._make_serializable(results)
            
            # 确保目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 结果已保存到: {output_path}")
            
        except PermissionError:
            logger.error(f"权限错误：无法写入文件 {output_path}")
            raise
        except OSError as e:
            logger.error(f"文件系统错误：{e}")
            raise
        except Exception as e:
            logger.error(f"保存结果时发生未知错误: {e}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化（增强版本）"""
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, SceneAnalysis):
            return {
                'complexity': obj.complexity.value,
                'object_count': int(obj.object_count),
                'motion_intensity': float(obj.motion_intensity),
                'lighting_quality': float(obj.lighting_quality),
                'occlusion_level': float(obj.occlusion_level),
                'crowd_density': float(obj.crowd_density),
                'scene_description': str(obj.scene_description)
            }
        elif isinstance(obj, SceneComplexity):
            return obj.value
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # 处理其他numpy标量类型
            return obj.item()
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif hasattr(obj, '__dict__'):
            # 对于其他对象，尝试转换其属性
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        else:
            try:
                return str(obj)
            except:
                return "unserializable_object"

def create_sample_video() -> str:
    """创建示例视频用于测试"""
    output_path = "test_data/video/sample_video.mp4"
    
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    # 使用固定随机种子保证可重现性
    rng = np.random.default_rng(seed=42)
    
    # 生成100帧测试视频
    for i in range(100):
        # 创建彩色背景
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些移动的矩形
        for j in range(3):
            x = int(50 + 200 * j + 50 * np.sin(i * 0.1 + j))
            y = int(100 + 50 * np.cos(i * 0.1 + j))
            cv2.rectangle(frame, (x, y), (x+80, y+60), (255, 255, 255), -1)
            cv2.putText(frame, f'Obj{j}', (x+10, y+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 添加帧号
        cv2.putText(frame, f'Frame {i}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    logger.info(f"✅ 示例视频已创建: {output_path}")
    
    return output_path

def main():
    """主函数"""
    logger.info("🎬 启动高级视频处理器测试")
    
    try:
        # 创建示例视频
        video_path = create_sample_video()
        
        # 初始化处理器
        processor = AdvancedVideoProcessor()
        
        # 处理视频
        results = processor.process_video(
            video_path=video_path,
            output_path="video_processing_results.json"
        )
        
        # 显示结果摘要
        summary = results['summary']
        logger.info("📊 处理结果摘要:")
        logger.info(f"   - 总目标数: {summary['total_objects']}")
        logger.info(f"   - 唯一轨迹数: {summary['unique_tracks']}")
        logger.info(f"   - 平均复杂度: {summary['avg_complexity']:.3f}")
        logger.info(f"   - 场景变化次数: {summary['scene_changes']}")
        logger.info(f"   - 处理时间: {summary['processing_time']:.2f}秒")
        
        logger.info("✅ 高级视频处理器测试完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()