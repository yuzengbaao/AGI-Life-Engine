#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构感知层 (Architecture Awareness Layer)
===========================================

AGI系统的架构感知层实现 - 让系统能够"理解自己的架构"

核心功能：
1. 组件依赖图谱映射 - 理解组件间的依赖关系
2. 性能瓶颈分析 - 识别系统性能热点
3. 架构健康度监控 - 持续评估架构状态

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

from .component_dependency_mapper import (
    ComponentDependencyMapper,
    DependencyNode,
    DependencyEdge,
    DependencyAnalysis,
    DependencyType,
    ComponentType,
    CircularDependency,
    CriticalPath
)

from .performance_bottleneck_analyzer import (
    PerformanceBottleneckAnalyzer,
    PerformanceMonitor,
    PerformanceSample,
    PerformanceAnalysis,
    PerformanceMetric,
    BottleneckSeverity,
    Bottleneck,
    PerformanceTrend,
    OptimizationSuggestion
)

from .architecture_health_monitor import (
    ArchitectureHealthMonitor,
    HealthMetric,
    HealthRisk,
    ComponentHealth,
    ArchitectureHealthReport,
    HealthStatus,
    RiskType
)

from .architecture_awareness_layer import (
    ArchitectureAwarenessLayer,
    ArchitectureAwarenessReport
)

__all__ = [
    # Core Components
    'ComponentDependencyMapper',
    'PerformanceBottleneckAnalyzer',
    'ArchitectureHealthMonitor',

    # Integration Layer
    'ArchitectureAwarenessLayer',

    # Dependency Mapping Classes
    'DependencyNode',
    'DependencyEdge',
    'DependencyAnalysis',
    'DependencyType',
    'ComponentType',
    'CircularDependency',
    'CriticalPath',

    # Performance Analysis Classes
    'PerformanceMonitor',
    'PerformanceSample',
    'PerformanceAnalysis',
    'PerformanceMetric',
    'BottleneckSeverity',
    'Bottleneck',
    'PerformanceTrend',
    'OptimizationSuggestion',

    # Health Monitoring Classes
    'HealthMetric',
    'HealthRisk',
    'ComponentHealth',
    'ArchitectureHealthReport',
    'HealthStatus',
    'RiskType',

    # Integration
    'ArchitectureAwarenessReport',
]

__version__ = '1.0.0'
