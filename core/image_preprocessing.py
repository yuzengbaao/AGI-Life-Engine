#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理管道
Image Preprocessing Pipeline

功能：
1. 图像缩放和调整大小
2. 归一化处理
3. 颜色空间转换
4. 边缘检测和特征提取
5. 噪声过滤

Author: AGI System Development Team
Date: 2026-01-26
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import logging


class ColorSpace(Enum):
    """颜色空间"""
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"
    HSV = "HSV"
    LAB = "LAB"


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.logger = logging.getLogger("ImagePreprocessor")
    
    def resize(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None, 
              keep_aspect_ratio: bool = True, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            size: 目标大小 (width, height)，默认使用初始化时的target_size
            keep_aspect_ratio: 是否保持宽高比
            interpolation: 插值方法
            
        Returns:
            调整大小后的图像
        """
        if size is None:
            size = self.target_size
        
        if keep_aspect_ratio:
            # 保持宽高比
            h, w = image.shape[:2]
            target_w, target_h = size
            
            # 计算缩放比例
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 调整大小
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            # 填充到目标大小
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return padded
        else:
            # 直接调整大小
            return cv2.resize(image, size, interpolation=interpolation)
    
    def normalize(self, image: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 输入图像
            method: 归一化方法
                - "standard": 标准化到 [0, 1]
                - "mean": 均值归一化
                - "zscore": Z-score 标准化
                
        Returns:
            归一化后的图像
        """
        image_float = image.astype(np.float32)
        
        if method == "standard":
            # 标准化到 [0, 1]
            if image_float.max() > 1.0:
                normalized = image_float / 255.0
            else:
                normalized = image_float
        elif method == "mean":
            # 均值归一化
            mean = image_float.mean()
            std = image_float.std()
            if std > 0:
                normalized = (image_float - mean) / std
            else:
                normalized = image_float - mean
        elif method == "zscore":
            # Z-score 标准化
            mean = image_float.mean(axis=(0, 1), keepdims=True)
            std = image_float.std(axis=(0, 1), keepdims=True)
            normalized = (image_float - mean) / (std + 1e-8)
        else:
            normalized = image_float
        
        return normalized
    
    def convert_color_space(self, image: np.ndarray, from_space: ColorSpace, 
                          to_space: ColorSpace) -> np.ndarray:
        """
        转换颜色空间
        
        Args:
            image: 输入图像
            from_space: 源颜色空间
            to_space: 目标颜色空间
            
        Returns:
            转换后的图像
        """
        if from_space == to_space:
            return image
        
        # 先转换到BGR（OpenCV默认格式）
        if from_space == ColorSpace.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif from_space == ColorSpace.GRAY:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif from_space == ColorSpace.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif from_space == ColorSpace.LAB:
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        
        # 从BGR转换到目标颜色空间
        if to_space == ColorSpace.RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif to_space == ColorSpace.GRAY:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif to_space == ColorSpace.HSV:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif to_space == ColorSpace.LAB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            return image
    
    def detect_edges(self, image: np.ndarray, method: str = "canny", 
                    threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
        """
        边缘检测
        
        Args:
            image: 输入图像（灰度图）
            method: 边缘检测方法
                - "canny": Canny边缘检测
                - "sobel": Sobel边缘检测
                - "laplacian": Laplacian边缘检测
            threshold1: 第一个阈值（Canny）
            threshold2: 第二个阈值（Canny）
            
        Returns:
            边缘图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 如果是float类型，转换为uint8
        if gray.dtype == np.float32 or gray.dtype == np.float64:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        if method == "canny":
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            edges = gray
        
        return edges
    
    def denoise(self, image: np.ndarray, method: str = "gaussian", 
               kernel_size: int = 5) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入图像
            method: 去噪方法
                - "gaussian": 高斯模糊
                - "median": 中值滤波
                - "bilateral": 双边滤波
            kernel_size: 核大小
            
        Returns:
            去噪后的图像
        """
        if method == "gaussian":
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == "median":
            denoised = cv2.medianBlur(image, kernel_size)
        elif method == "bilateral":
            denoised = cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            denoised = image
        
        return denoised
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        提取图像特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本统计特征
        features['mean'] = image.mean(axis=(0, 1)).tolist()
        features['std'] = image.std(axis=(0, 1)).tolist()
        features['min'] = image.min(axis=(0, 1)).tolist()
        features['max'] = image.max(axis=(0, 1)).tolist()
        
        # 边缘特征
        edges = self.detect_edges(image, method="canny")
        features['edge_density'] = (edges > 0).sum() / edges.size
        
        # 亮度特征
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        features['brightness'] = gray.mean()
        features['contrast'] = gray.std()
        
        return features
    
    def preprocess(self, image: np.ndarray, resize: bool = True, 
                   normalize: bool = True, denoise: bool = False,
                   color_space: Optional[ColorSpace] = None) -> Dict[str, Any]:
        """
        完整的预处理流程
        
        Args:
            image: 输入图像
            resize: 是否调整大小
            normalize: 是否归一化
            denoise: 是否去噪
            color_space: 目标颜色空间
            
        Returns:
            预处理结果字典
        """
        result = {
            'original': image.copy(),
            'processed': image.copy(),
            'features': {}
        }
        
        processed = image.copy()
        
        # 去噪
        if denoise:
            processed = self.denoise(processed, method="gaussian", kernel_size=5)
        
        # 调整大小
        if resize:
            processed = self.resize(processed, keep_aspect_ratio=False)
        
        # 颜色空间转换
        if color_space is not None:
            processed = self.convert_color_space(processed, ColorSpace.BGR, color_space)
        
        # 归一化
        if normalize:
            processed = self.normalize(processed, method="standard")
        
        result['processed'] = processed
        result['features'] = self.extract_features(processed)
        
        return result


def test_image_preprocessing():
    """测试图像预处理"""
    import cv2
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("测试图像预处理管道")
    print("=" * 70)
    
    # 创建预处理器
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # 生成测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"\n原始图像形状: {test_image.shape}")
    
    # 测试调整大小
    resized = preprocessor.resize(test_image, size=(224, 224), keep_aspect_ratio=False)
    print(f"调整大小后: {resized.shape}")
    
    # 测试归一化
    normalized = preprocessor.normalize(resized, method="standard")
    print(f"归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # 测试边缘检测
    edges = preprocessor.detect_edges(resized, method="canny")
    print(f"边缘检测: {edges.shape}")
    
    # 测试去噪
    denoised = preprocessor.denoise(resized, method="gaussian", kernel_size=5)
    print(f"去噪后: {denoised.shape}")
    
    # 测试特征提取
    features = preprocessor.extract_features(resized)
    print(f"\n提取的特征:")
    for key, value in features.items():
        if isinstance(value, list):
            print(f"  {key}: {[f'{v:.2f}' for v in value]}")
        else:
            print(f"  {key}: {value:.3f}")
    
    # 测试完整预处理流程
    print("\n测试完整预处理流程...")
    result = preprocessor.preprocess(
        test_image,
        resize=True,
        normalize=True,
        denoise=True,
        color_space=ColorSpace.RGB
    )
    
    print(f"预处理后形状: {result['processed'].shape}")
    print(f"预处理后范围: [{result['processed'].min():.3f}, {result['processed'].max():.3f}]")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_image_preprocessing()
