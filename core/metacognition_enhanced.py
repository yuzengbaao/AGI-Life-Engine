#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型元认知层 (Enhanced MetaCognition)
===========================================

功能: 集成超深度推理引擎，支持99,999步递归推理
版本: 2.0.0 (2026-01-19)

主要改进:
1. 集成UltraDeepReasoningEngine
2. 动态推理深度选择 (100-99,999步)
3. 分层递归推理支持
4. 语义压缩与快照机制
"""

import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入原始元认知模块
try:
    from core.metacognition import MetaCognition as BaseMetaCognition
    from core.metacognition import ThoughtFrame, Intention, MetaInsight
except ImportError:
    # 如果导入失败，定义基本类
    BaseMetaCognition = object
    ThoughtFrame = None
    Intention = None
    MetaInsight = None

# 导入新的深度推理引擎
try:
    from core.deep_reasoning_engine import (
        UltraDeepReasoningEngine,
        LayerType,
        ReasoningState,
        HierarchicalReasoningConfig
    )
except ImportError:
    # 独立运行时的备用定义
    UltraDeepReasoningEngine = None
    LayerType = None
    ReasoningState = None
    HierarchicalReasoningConfig = None

logger = logging.getLogger(__name__)


class EnhancedMetaCognition(BaseMetaCognition):
    """
    增强型元认知层

    核心改进:
    1. 集成99,999步深度推理能力
    2. 分层递归架构
    3. 自动选择推理深度
    4. 语义压缩优化
    """

    # 🔧 增强的推理深度配置 (v2.0)
    SHALLOW_HORIZON = 1000      # 简单任务 (原100，提升10倍)
    NORMAL_HORIZON = 10000      # 常规任务 (原500，提升20倍)
    DEEP_HORIZON = 50000        # 复杂任务 (原1000，提升50倍)
    ULTRA_DEEP_HORIZON = 99999  # 极端复杂任务 (原2000，提升50倍)

    MIN_HORIZON = 100           # 最小推理步数（快速响应）
    MAX_HORIZON = 99999         # 最大推理步数（超深度思考）
    DEFAULT_HORIZON = NORMAL_HORIZON  # 默认使用常规深度

    # 层级推理阈值
    LAYER_THRESHOLDS = {
        'meta': 99,              # 元层: 1-99步
        'strategic': 999,        # 战略层: 100-999步
        'tactical': 9999,        # 战术层: 1000-9999步
        'operational': 99999     # 操作层: 10000-99999步
    }

    def __init__(self, seed_ref=None, memory_ref=None, enable_deep_reasoning=True):
        """
        初始化增强型元认知层

        Args:
            seed_ref: TheSeed实例的引用
            memory_ref: TopologicalMemory实例的引用
            enable_deep_reasoning: 是否启用超深度推理
        """
        # 调用父类初始化
        if BaseMetaCognition != object:
            super().__init__(seed_ref, memory_ref)

        # 初始化超深度推理引擎
        self.enable_deep_reasoning = enable_deep_reasoning
        self.deep_reasoning_engine = None

        if enable_deep_reasoning:
            self.deep_reasoning_engine = UltraDeepReasoningEngine(
                max_depth=self.MAX_HORIZON
            )

            # 注册推理步骤回调
            self.deep_reasoning_engine.register_callback(
                self._on_reasoning_step
            )

        # 推理统计
        self.reasoning_stats = {
            'total_reasoning_steps': 0,
            'deep_reasoning_used': 0,
            'compression_saves': 0,
            'layer_usage': {layer: 0 for layer in LayerType}
        }

        logger.info(f"🧠 Enhanced MetaCognition initialized (max_depth={self.MAX_HORIZON})")
        logger.info(f"   - 推理深度档位: {self.SHALLOW_HORIZON}/{self.NORMAL_HORIZON}/{self.DEEP_HORIZON}/{self.ULTRA_DEEP_HORIZON}")

    def _on_reasoning_step(self, state: ReasoningState):
        """
        推理步骤回调

        Args:
            state: 当前推理状态
        """
        self.reasoning_stats['total_reasoning_steps'] += 1
        self.reasoning_stats['layer_usage'][state.layer] += 1

        # 记录关键推理点
        if state.confidence < 0.5:
            logger.debug(f"[DeepReasoning] Low confidence at step {state.step_number}: {state.confidence:.2f}")

        # 记录层级转换
        if self.deep_reasoning_engine and self.deep_reasoning_engine.trace.reasoning_path:
            last_state = self.deep_reasoning_engine.trace.reasoning_path[-1]
            if last_state.layer != state.layer and last_state.step_number > 0:
                logger.info(f"[DeepReasoning] Layer transition: {last_state.layer} -> {state.layer} at step {state.step_number}")

    def adjust_horizon_adaptive(self,
                                 task_complexity: float,
                                 uncertainty: float,
                                 available_time: Optional[float] = None) -> int:
        """
        自适应调整推理深度（增强版）

        决策逻辑:
        1. 任务复杂度 > 0.8 → 超深度推理 (50,000-99,999步)
        2. 任务复杂度 > 0.6 → 深度推理 (10,000-50,000步)
        3. 任务复杂度 > 0.3 → 常规推理 (1,000-10,000步)
        4. 否则 → 浅层推理 (100-1,000步)

        Args:
            task_complexity: 任务复杂度 (0-1)
            uncertainty: 不确定性 (0-1)
            available_time: 可用时间（秒），None表示无限制

        Returns:
            int: 推荐的推理深度
        """
        # 基础深度选择
        if task_complexity > 0.8:
            base_horizon = self.ULTRA_DEEP_HORIZON
            tier = 'ultra_deep'
        elif task_complexity > 0.6:
            base_horizon = self.DEEP_HORIZON
            tier = 'deep'
        elif task_complexity > 0.3:
            base_horizon = self.NORMAL_HORIZON
            tier = 'normal'
        else:
            base_horizon = self.SHALLOW_HORIZON
            tier = 'shallow'

        # 不确定性调整
        if uncertainty > 0.7:
            # 高不确定性需要更深度推理
            base_horizon = min(base_horizon * 1.5, self.MAX_HORIZON)
        elif uncertainty < 0.3:
            # 低确定性可减少推理深度
            base_horizon = max(base_horizon * 0.7, self.MIN_HORIZON)

        # 时间约束调整
        if available_time is not None:
            # 假设每步需要0.01秒（保守估计）
            max_steps_by_time = int(available_time / 0.01)
            base_horizon = min(base_horizon, max_steps_by_time)

        final_horizon = int(base_horizon)

        logger.info(f"  [MetaCog] 推理深度调整: 复杂度={task_complexity:.2f}, "
                   f"不确定性={uncertainty:.2f} → {tier} ({final_horizon}步)")

        return final_horizon

    def perform_deep_reasoning(self,
                              initial_context: Dict[str, Any],
                              max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        执行超深度推理

        Args:
            initial_context: 初始推理上下文
            max_steps: 最大推理步数（None表示使用引擎默认值）

        Returns:
            推理结果摘要
        """
        if not self.enable_deep_reasoning or not self.deep_reasoning_engine:
            logger.warning("Deep reasoning engine not enabled")
            return {'error': 'Deep reasoning not enabled'}

        # 设置推理深度
        if max_steps:
            original_max = self.deep_reasoning_engine.max_depth
            self.deep_reasoning_engine.max_depth = min(max_steps, self.MAX_HORIZON)

        # 执行推理循环
        try:
            # 模拟推理步骤（实际应用中应由具体任务驱动）
            step_count = 0
            target_steps = max_steps or self.NORMAL_HORIZON

            for step in range(target_steps):
                if step >= self.deep_reasoning_engine.max_depth:
                    break

                # 创建推理上下文
                context = {
                    **initial_context,
                    'step': step + 1,
                    'query': f"推理步骤 {step + 1}",
                    'concept': f"concept_{step % 100}"
                }

                # 计算置信度（模拟）
                confidence = 0.5 + 0.3 * np.sin(step / 100)  # 周期性变化

                # 执行推理步骤
                state = self.deep_reasoning_engine.reasoning_step(
                    context=context,
                    confidence=confidence
                )

                step_count += 1

                # 每1000步报告一次进度
                if step_count % 1000 == 0:
                    progress = step_count / self.deep_reasoning_engine.max_depth
                    logger.info(f"  [DeepReasoning] Progress: {step_count}/{self.deep_reasoning_engine.max_depth} "
                               f"({progress:.2%}), Current Layer: {state.layer}")

            # 获取推理轨迹摘要
            trace_summary = self.deep_reasoning_engine.get_trace_summary()

            result = {
                'success': True,
                'total_steps': step_count,
                'trace_summary': trace_summary,
                'reasoning_stats': self.reasoning_stats,
                'compression_ratio': trace_summary['compression_ratio'],
                'time_saved': trace_summary.get('estimated_time_saved', 'N/A')
            }

            logger.info(f"  [DeepReasoning] Completed: {step_count} steps, "
                       f"compression={trace_summary['compression_ratio']:.1f}:1")

            return result

        except StopIteration as e:
            logger.warning(f"  [DeepReasoning] Stopped: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_steps': step_count
            }
        except Exception as e:
            logger.error(f"  [DeepReasoning] Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_steps': step_count
            }
        finally:
            # 恢复原始最大深度
            if max_steps and 'original_max' in locals():
                self.deep_reasoning_engine.max_depth = original_max

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """获取增强型统计信息"""
        base_stats = {}

        # 获取父类统计（如果可用）
        if hasattr(super(), 'get_statistics'):
            base_stats = super().get_statistics()

        # 添加深度推理统计
        enhanced_stats = {
            **base_stats,
            'deep_reasoning_enabled': self.enable_deep_reasoning,
            'max_reasoning_depth': self.MAX_HORIZON,
            'reasoning_stats': self.reasoning_stats,
            'layer_distribution': self.reasoning_stats['layer_usage'] if self.reasoning_stats else {}
        }

        # 如果有深度推理引擎，添加其统计
        if self.deep_reasoning_engine:
            trace_summary = self.deep_reasoning_engine.get_trace_summary()
            enhanced_stats['deep_reasoning_trace'] = trace_summary

        return enhanced_stats


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("增强型元认知层测试")
    print("=" * 70)

    # 创建增强型元认知
    meta_cog = EnhancedMetaCognition(enable_deep_reasoning=True)

    print("\n测试1: 自适应深度选择")
    print("-" * 70)

    # 测试不同复杂度的任务
    test_cases = [
        (0.2, 0.3, "简单对话"),
        (0.5, 0.5, "常规任务"),
        (0.7, 0.7, "复杂推理"),
        (0.9, 0.8, "超深度分析")
    ]

    for complexity, uncertainty, desc in test_cases:
        horizon = meta_cog.adjust_horizon_adaptive(complexity, uncertainty)
        print(f"{desc}: 复杂度={complexity:.1f}, 不确定性={uncertainty:.1f} → {horizon}步")

    print("\n测试2: 超深度推理执行")
    print("-" * 70)

    # 执行100步超深度推理
    result = meta_cog.perform_deep_reasoning(
        initial_context={'query': '分析系统拓扑结构', 'domain': 'AGI'},
        max_steps=100
    )

    if result.get('success'):
        print(f"✅ 推理成功")
        print(f"   总步数: {result['total_steps']}")
        print(f"   压缩比: {result['compression_ratio']:.1f}:1")
        print(f"   时间节省: {result['time_saved']}")
    else:
        print(f"❌ 推理失败: {result.get('error')}")

    print("\n" + "=" * 70)
    print("测试完成")
