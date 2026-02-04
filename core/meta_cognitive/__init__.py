#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元认知层 (Meta-Cognitive Layer)
================================

AGI系统的元认知层实现 - 让系统能够"思考自己的思考"

核心功能：
1. 任务理解深度评估 - "我真的理解了吗？"
2. 能力匹配分析 - "我能解决这个问题吗？我的局限在哪里？"
3. 失败原因归因 - "为什么失败？是架构问题还是数据问题？"

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

from .task_understanding_evaluator import (
    TaskUnderstandingEvaluator,
    TaskAnalysis,
    UnderstandingLevel
)

from .capability_matcher import (
    CapabilityMatcher,
    MatchResult,
    CapabilityProfile,
    MatchLevel
)

from .failure_attribution_engine import (
    FailureAttributionEngine,
    FailureAnalysis,
    FailureType,
    RootCause
)

from .meta_cognitive_layer import (
    MetaCognitiveLayer,
    MetaCognitiveReport
)

__all__ = [
    # Core Components
    'TaskUnderstandingEvaluator',
    'CapabilityMatcher',
    'FailureAttributionEngine',

    # Integration Layer
    'MetaCognitiveLayer',
    'MetaCognitiveReport',

    # Data Classes
    'TaskAnalysis',
    'MatchResult',
    'FailureAnalysis',
    'CapabilityProfile',

    # Enums
    'UnderstandingLevel',
    'MatchLevel',
    'FailureType',
    'RootCause',
]

__version__ = '1.0.0'
