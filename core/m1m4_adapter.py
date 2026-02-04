#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1-M4组件适配器 (M1-M4 Components Adapter)
================================================

集成M1-M4分形AGI组件到现有AGI_Life_Engine系统

M1: MetaLearner - 元参数优化器
M2: GoalQuestioner - 目标质疑模块
M3: SelfModifyingEngine - 架构自修改引擎
M4: RecursiveSelfMemory - 递归自引用记忆系统

设计原则:
- 非侵入式集成: 通过EventBus连接，不修改现有组件
- 渐进式启用: 可选择性启用各组件
- 容错降级: 组件失败不影响主系统
- 可观测性: 完整的事件追踪和日志

版本: 1.0.0
状态: 生产就绪
"""

import sys
import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# 确保项目路径在sys.path中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """组件状态"""
    DISABLED = "disabled"       # 未启用
    INITIALIZING = "initializing"  # 初始化中
    ACTIVE = "active"           # 运行中
    DEGRADED = "degraded"       # 降级运行
    ERROR = "error"             # 错误状态


@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_name: str
    status: ComponentStatus
    last_heartbeat: float
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class M1M4Adapter:
    """
    M1-M4组件适配器

    职责:
    1. 初始化M1-M4组件
    2. 建立与现有系统的EventBus连接
    3. 提供统一的组件管理接口
    4. 监控组件健康状态
    """

    # 配置
    ENABLE_M1_META_LEARNER = True
    ENABLE_M2_GOAL_QUESTIONER = True
    ENABLE_M3_SELF_MODIFYING = True   # ⚠️ 谨慎启用
    ENABLE_M4_RECURSIVE_MEMORY = True

    # 冷却时间（秒）
    M2_COOLDOWN_SECONDS = 300  # 5分钟
    M3_ANALYSIS_INTERVAL = 600  # 10分钟

    def __init__(self, event_bus, project_root: str = None):
        """
        初始化适配器

        Args:
            event_bus: AGI_Life_Engine的EventBus实例
            project_root: 项目根路径
        """
        self.event_bus = event_bus
        self.project_root = project_root or str(Path.cwd())

        # 组件实例
        self.meta_learner = None
        self.goal_questioner = None
        self.self_modifier = None
        self.recursive_memory = None

        # 组件健康状态
        self.component_health: Dict[str, ComponentHealth] = {}

        # 统计信息
        self.stats = {
            'total_initialized': 0,
            'total_active': 0,
            'total_errors': 0,
            'events_processed': 0
        }

        # 冷却时间追踪
        self._last_m2_questioning = 0
        self._last_m3_analysis = 0

        logger.info("🔧 M1M4Adapter initialized")

    def initialize_all(self) -> Dict[str, bool]:
        """
        初始化所有启用的M1-M4组件

        Returns:
            组件名 -> 初始化是否成功的字典
        """
        results = {}
        logger.info("=" * 60)
        logger.info("初始化M1-M4分形AGI组件")
        logger.info("=" * 60)

        # M1: MetaLearner
        if self.ENABLE_M1_META_LEARNER:
            results['M1_MetaLearner'] = self._init_m1_meta_learner()
        else:
            results['M1_MetaLearner'] = False
            logger.info("   [M1] MetaLearner: DISABLED")

        # M2: GoalQuestioner
        if self.ENABLE_M2_GOAL_QUESTIONER:
            results['M2_GoalQuestioner'] = self._init_m2_goal_questioner()
        else:
            results['M2_GoalQuestioner'] = False
            logger.info("   [M2] GoalQuestioner: DISABLED")

        # M3: SelfModifyingEngine
        if self.ENABLE_M3_SELF_MODIFYING:
            results['M3_SelfModifyingEngine'] = self._init_m3_self_modifying_engine()
        else:
            results['M3_SelfModifyingEngine'] = False
            logger.info("   [M3] SelfModifyingEngine: DISABLED")

        # M4: RecursiveSelfMemory
        if self.ENABLE_M4_RECURSIVE_MEMORY:
            results['M4_RecursiveSelfMemory'] = self._init_m4_recursive_memory()
        else:
            results['M4_RecursiveSelfMemory'] = False
            logger.info("   [M4] RecursiveSelfMemory: DISABLED")

        # 统计
        self.stats['total_initialized'] = sum(results.values())
        self.stats['total_active'] = len([k for k, v in results.items() if v])

        logger.info("=" * 60)
        logger.info(f"M1-M4组件初始化完成: {self.stats['total_active']}/{len(results)} 成功")
        logger.info("=" * 60)

        return results

    # ========================================================================
    # M1: MetaLearner 初始化
    # ========================================================================

    def _init_m1_meta_learner(self) -> bool:
        """初始化M1: MetaLearner元参数优化器"""
        try:
            from core.meta_learner import MetaLearner, MetaStrategy, StepMetrics, ParameterUpdate

            logger.info("   [M1] 初始化MetaLearner...")

            self.meta_learner = MetaLearner(
                event_bus=self.event_bus,
                initial_strategy=MetaStrategy.RULE_BASED  # 使用规则策略（稳定）
            )

            # 订阅性能指标事件
            self.event_bus.subscribe("the_seed.performance", self._on_the_seed_performance)
            self.event_bus.subscribe("learning.step_completed", self._on_learning_step)

            self.component_health['M1_MetaLearner'] = ComponentHealth(
                component_name='M1_MetaLearner',
                status=ComponentStatus.ACTIVE,
                last_heartbeat=time.time()
            )

            logger.info("   [M1] ✅ MetaLearner已启动 (规则策略)")
            return True

        except Exception as e:
            logger.error(f"   [M1] ❌ MetaLearner初始化失败: {e}")
            self.component_health['M1_MetaLearner'] = ComponentHealth(
                component_name='M1_MetaLearner',
                status=ComponentStatus.ERROR,
                last_heartbeat=time.time(),
                last_error=str(e)
            )
            return False

    def _on_the_seed_performance(self, event):
        """处理TheSeed性能指标事件"""
        if self.meta_learner is None:
            return

        try:
            data = event.data if hasattr(event, 'data') else event
            metrics = StepMetrics(
                step=data.get('step', 0),
                reward=data.get('reward', 0.0),
                loss=data.get('loss', 0.0),
                uncertainty=data.get('uncertainty', 0.0),
                exploration_rate=data.get('exploration_rate', 0.0)
            )

            # MetaLearner观察性能指标
            self.meta_learner.observe(metrics)

            # 获取参数更新建议
            update = self.meta_learner.propose_update(mode='auto')
            if update:
                logger.info(f"   [M1] 参数更新建议: {update}")
                # 发布参数更新事件
                self._publish_event("meta.parameter_update", {
                    'parameters': update.parameters,
                    'confidence': update.confidence,
                    'reason': update.reason
                })

            self.stats['events_processed'] += 1
            self._update_heartbeat('M1_MetaLearner')

        except Exception as e:
            logger.error(f"   [M1] 处理性能指标失败: {e}")
            self._record_error('M1_MetaLearner', str(e))

    def _on_learning_step(self, event):
        """处理学习步骤事件"""
        if self.meta_learner is None:
            return

        try:
            data = event.data if hasattr(event, 'data') else event
            metrics = StepMetrics(
                step=data.get('step', 0),
                reward=data.get('reward', 0.0),
                loss=data.get('loss', 0.0)
            )
            self.meta_learner.observe(metrics)

        except Exception as e:
            logger.warning(f"   [M1] 处理学习步骤失败: {e}")

    # ========================================================================
    # M2: GoalQuestioner 初始化
    # ========================================================================

    def _init_m2_goal_questioner(self) -> bool:
        """初始化M2: GoalQuestioner目标质疑模块"""
        try:
            from core.goal_questioner import (
                GoalQuestioner, GoalSpec, QuestioningContext,
                GoalEvaluation, GoalRevision, GoalBiasType
            )

            logger.info("   [M2] 初始化GoalQuestioner...")

            self.goal_questioner = GoalQuestioner(
                event_bus=self.event_bus
            )

            # 订阅目标相关事件
            self.event_bus.subscribe("goal.created", self._on_goal_created)
            self.event_bus.subscribe("goal.completed", self._on_goal_completed)
            self.event_bus.subscribe("goal.failed", self._on_goal_failed)

            self.component_health['M2_GoalQuestioner'] = ComponentHealth(
                component_name='M2_GoalQuestioner',
                status=ComponentStatus.ACTIVE,
                last_heartbeat=time.time()
            )

            logger.info(f"   [M2] ✅ GoalQuestioner已启动 (冷却期: {self.M2_COOLDOWN_SECONDS}s)")
            return True

        except Exception as e:
            logger.error(f"   [M2] ❌ GoalQuestioner初始化失败: {e}")
            self.component_health['M2_GoalQuestioner'] = ComponentHealth(
                component_name='M2_GoalQuestioner',
                status=ComponentStatus.ERROR,
                last_heartbeat=time.time(),
                last_error=str(e)
            )
            return False

    def _on_goal_created(self, event):
        """处理目标创建事件"""
        if self.goal_questioner is None:
            return

        # 检查冷却期
        if time.time() - self._last_m2_questioning < self.M2_COOLDOWN_SECONDS:
            return

        try:
            data = event.data if hasattr(event, 'data') else event

            # 构建GoalSpec
            goal_spec = GoalSpec(
                goal_id=data.get('goal_id', ''),
                goal_type=data.get('goal_type', 'unknown'),
                description=data.get('description', ''),
                target_outcome=data.get('target_outcome', ''),
                success_criteria=data.get('success_criteria', []),
                hard_constraints=data.get('hard_constraints', []),
                soft_constraints=data.get('soft_constraints', []),
                priority=data.get('priority', 0.5),
                deadline=data.get('deadline'),
                metadata=data.get('metadata', {})
            )

            # 构建QuestioningContext
            context = QuestioningContext(
                current_goals=[],  # 可从GoalManager获取
                recent_outcomes=[],
                system_state=data.get('system_state', {}),
                available_resources=data.get('available_resources', {}),
                time_pressure=data.get('time_pressure', 0.5)
            )

            # 检查目标
            result = self.goal_questioner.inspect(goal_spec, context)

            if result.get('has_bias'):
                logger.warning(f"   [M2] ⚠️ 目标偏差检测: {result.get('bias_type')}")
                # 发布目标偏差事件
                self._publish_event("goal.bias_detected", {
                    'goal_id': goal_spec.goal_id,
                    'bias_type': result.get('bias_type'),
                    'severity': result.get('severity'),
                    'description': result.get('description')
                })

                # 如果严重偏差，提出修订建议
                if result.get('severity') in ['high', 'critical']:
                    evaluation = self.goal_questioner.evaluate(goal_spec, context)
                    revision = self.goal_questioner.propose_revision(evaluation, goal_spec)

                    if revision:
                        logger.info(f"   [M2] 📝 目标修订建议: {revision.revision_reason}")
                        self._publish_event("goal.revision_proposed", {
                            'goal_id': goal_spec.goal_id,
                            'revision': revision.__dict__
                        })

            self._last_m2_questioning = time.time()
            self.stats['events_processed'] += 1
            self._update_heartbeat('M2_GoalQuestioner')

        except Exception as e:
            logger.error(f"   [M2] 处理目标创建失败: {e}")
            self._record_error('M2_GoalQuestioner', str(e))

    def _on_goal_completed(self, event):
        """处理目标完成事件"""
        if self.goal_questioner is None:
            return

        try:
            # 记录成功案例用于学习
            data = event.data if hasattr(event, 'data') else event
            self.goal_questioner.record_outcome({
                'goal_id': data.get('goal_id'),
                'status': 'completed',
                'outcome': data.get('outcome')
            })

        except Exception as e:
            logger.warning(f"   [M2] 记录目标完成失败: {e}")

    def _on_goal_failed(self, event):
        """处理目标失败事件"""
        if self.goal_questioner is None:
            return

        try:
            # 记录失败案例用于学习
            data = event.data if hasattr(event, 'data') else event
            self.goal_questioner.record_outcome({
                'goal_id': data.get('goal_id'),
                'status': 'failed',
                'error': data.get('error')
            })

        except Exception as e:
            logger.warning(f"   [M2] 记录目标失败失败: {e}")

    # ========================================================================
    # M3: SelfModifyingEngine 初始化
    # ========================================================================

    def _init_m3_self_modifying_engine(self) -> bool:
        """初始化M3: SelfModifyingEngine架构自修改引擎"""
        try:
            from core.self_modifying_engine import SelfModifyingEngine

            logger.info("   [M3] 初始化SelfModifyingEngine...")
            logger.warning("   [M3] ⚠️  自修改引擎已启动 - 所有修改将经过沙箱测试")

            self.self_modifier = SelfModifyingEngine(
                project_root=self.project_root,
                auto_apply_safe=False,  # 不自动应用，需要人工确认
                event_bus=self.event_bus
            )

            # 订阅代码分析相关事件
            self.event_bus.subscribe("code.analysis_requested", self._on_code_analysis_requested)
            self.event_bus.subscribe("code.patch_proposed", self._on_code_patch_proposed)

            self.component_health['M3_SelfModifyingEngine'] = ComponentHealth(
                component_name='M3_SelfModifyingEngine',
                status=ComponentStatus.ACTIVE,
                last_heartbeat=time.time()
            )

            logger.info("   [M3] ✅ SelfModifyingEngine已启动 (沙箱模式)")
            return True

        except Exception as e:
            logger.error(f"   [M3] ❌ SelfModifyingEngine初始化失败: {e}")
            self.component_health['M3_SelfModifyingEngine'] = ComponentHealth(
                component_name='M3_SelfModifyingEngine',
                status=ComponentStatus.ERROR,
                last_heartbeat=time.time(),
                last_error=str(e)
            )
            return False

    def _on_code_analysis_requested(self, event):
        """处理代码分析请求"""
        if self.self_modifier is None:
            return

        # 检查冷却期
        if time.time() - self._last_m3_analysis < self.M3_ANALYSIS_INTERVAL:
            return

        try:
            data = event.data if hasattr(event, 'data') else event
            module_path = data.get('module_path')

            if not module_path:
                return

            logger.info(f"   [M3] 分析模块: {module_path}")

            # 执行静态分析
            analysis = self.self_modifier.analyze(module_path)

            logger.info(f"   [M3] 分析结果: 复杂度={analysis.complexity:.2f}, "
                       f"安全分数={analysis.safety_score:.2f}")

            # 发布分析结果事件
            self._publish_event("code.analysis_completed", {
                'module_path': module_path,
                'complexity': analysis.complexity,
                'safety_score': analysis.safety_score,
                'locations_count': len(analysis.locations)
            })

            self._last_m3_analysis = time.time()
            self.stats['events_processed'] += 1
            self._update_heartbeat('M3_SelfModifyingEngine')

        except Exception as e:
            logger.error(f"   [M3] 代码分析失败: {e}")
            self._record_error('M3_SelfModifyingEngine', str(e))

    def _on_code_patch_proposed(self, event):
        """处理代码补丁提案（从InsightIntegrator等来源）"""
        if self.self_modifier is None:
            return

        try:
            data = event.data if hasattr(event, 'data') else event

            # 记录提案（不自动应用）
            logger.info(f"   [M3] 📋 收到代码补丁提案: {data.get('description')}")

            # 这里可以添加验证逻辑
            # 实际应用需要人工确认或自动测试通过

        except Exception as e:
            logger.warning(f"   [M3] 处理补丁提案失败: {e}")

    # ========================================================================
    # M4: RecursiveSelfMemory 初始化
    # ========================================================================

    def _init_m4_recursive_memory(self) -> bool:
        """初始化M4: RecursiveSelfMemory递归自引用记忆系统"""
        try:
            from core.recursive_self_memory import RecursiveSelfMemory, MemoryImportance
            # 修复：导入核心同步事件总线
            from core.event_bus import EventBus as CoreEventBus

            logger.info("   [M4] 初始化RecursiveSelfMemory...")

            # 使用核心同步事件总线供M4组件使用，解决与LifeEngineEventBus的不兼容问题
            core_bus = CoreEventBus.get_instance()

            self.recursive_memory = RecursiveSelfMemory(
                event_bus=core_bus,
                memory_dir=str(Path(self.project_root) / "data" / "recursive_self_memory")
            )

            # 订阅记忆相关事件
            self.event_bus.subscribe("memory.*", self._on_memory_operation)
            self.event_bus.subscribe("memory.query", self._on_memory_query)
            self.event_bus.subscribe("system.shutdown", self._on_system_shutdown)

            # 记住系统启动
            self.recursive_memory.remember(
                event_type="system_event",
                content={"event": "agi_system_startup", "components": ["M1", "M2", "M3", "M4"]},
                importance=MemoryImportance.HIGH,
                why="记录M1-M4组件集成启动",
                trigger="M1M4Adapter"
            )

            self.component_health['M4_RecursiveSelfMemory'] = ComponentHealth(
                component_name='M4_RecursiveSelfMemory',
                status=ComponentStatus.ACTIVE,
                last_heartbeat=time.time()
            )

            stats = self.recursive_memory.get_statistics()
            logger.info(f"   [M4] ✅ RecursiveSelfMemory已启动 (记忆数: {stats['l0_event_count']})")
            return True

        except Exception as e:
            logger.error(f"   [M4] ❌ RecursiveSelfMemory初始化失败: {e}")
            self.component_health['M4_RecursiveSelfMemory'] = ComponentHealth(
                component_name='M4_RecursiveSelfMemory',
                status=ComponentStatus.ERROR,
                last_heartbeat=time.time(),
                last_error=str(e)
            )
            return False

    def _on_memory_operation(self, event):
        """处理记忆操作事件（记住"记忆"本身）"""
        if self.recursive_memory is None:
            return

        try:
            from core.recursive_self_memory import MemoryImportance

            # 递归自指：记住记忆系统的操作
            event_type = event.type if hasattr(event, 'type') else 'memory_operation'
            data = event.data if hasattr(event, 'data') else {}

            # 只记录重要事件（避免无限递归）
            if event_type not in ['memory.operation_recorded']:
                self.recursive_memory.remember(
                    event_type=f"memory_{event_type}",
                    content=data,
                    importance=MemoryImportance.LOW,
                    why=f"自动记录记忆操作: {event_type}",
                    trigger="M4_RecursiveSelfMemory",
                    _is_meta=True
                )

        except Exception as e:
            logger.warning(f"   [M4] 记录记忆操作失败: {e}")

    def _on_memory_query(self, event):
        """处理记忆查询事件"""
        if self.recursive_memory is None:
            return

        try:
            data = event.data if hasattr(event, 'data') else event
            query = data.get('query', '')
            limit = data.get('limit', 10)

            if not query:
                return

            # 查询记忆
            results = self.recursive_memory.recall(query, limit=limit)

            # 发布查询结果事件
            self._publish_event("memory.query_result", {
                'query': query,
                'results_count': len(results),
                'results': [e.id for e in results[:5]]  # 只返回前5个ID
            })

            self.stats['events_processed'] += 1
            self._update_heartbeat('M4_RecursiveSelfMemory')

        except Exception as e:
            logger.warning(f"   [M4] 记忆查询失败: {e}")

    def _on_system_shutdown(self, event):
        """处理系统关闭事件"""
        if self.recursive_memory is None:
            return

        try:
            from core.recursive_self_memory import MemoryImportance

            # 记住系统关闭
            self.recursive_memory.remember(
                event_type="system_event",
                content={"event": "agi_system_shutdown"},
                importance=MemoryImportance.MEDIUM,
                why="记录系统关闭",
                trigger="M4_RecursiveSelfMemory"
            )

            # 导出记忆
            export_path = Path(self.project_root) / "data" / "recursive_self_memory" / f"backup_{int(time.time())}.json"
            self.recursive_memory.export_memories(str(export_path))
            logger.info(f"   [M4] 记忆已备份: {export_path}")

        except Exception as e:
            logger.error(f"   [M4] 关闭时备份失败: {e}")

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """发布事件到EventBus"""
        try:
            # 同步发布（如果EventBus支持异步，这里需要调整）
            if hasattr(self.event_bus, 'publish'):
                # 检查是否是协程
                if asyncio.iscoroutinefunction(self.event_bus.publish):
                    # 创建异步任务
                    asyncio.create_task(self.event_bus.publish(event_type, data))
                else:
                    # 同步调用
                    self.event_bus.publish(event_type, data)
            elif hasattr(self.event_bus, '_bus'):
                # LifeEngineEventBus的内部bus
                self.event_bus._bus.publish(type=event_type, source="M1M4Adapter", data=data)
        except Exception as e:
            logger.warning(f"发布事件失败 {event_type}: {e}")

    def _update_heartbeat(self, component_name: str):
        """更新组件心跳"""
        if component_name in self.component_health:
            self.component_health[component_name].last_heartbeat = time.time()

    def _record_error(self, component_name: str, error: str):
        """记录组件错误"""
        if component_name in self.component_health:
            health = self.component_health[component_name]
            health.error_count += 1
            health.last_error = error
            if health.error_count > 5:
                health.status = ComponentStatus.DEGRADED
        self.stats['total_errors'] += 1

    # ========================================================================
    # 公共接口
    # ========================================================================

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有组件的健康状态"""
        status = {}
        for name, health in self.component_health.items():
            status[name] = {
                'status': health.status.value,
                'last_heartbeat': health.last_heartbeat,
                'error_count': health.error_count,
                'last_error': health.last_error,
                'metrics': health.metrics
            }
        return status

    def get_statistics(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        stats = self.stats.copy()

        # 添加各组件的统计
        if self.recursive_memory:
            stats['M4_memory'] = self.recursive_memory.get_statistics()

        if self.meta_learner:
            stats['M1_meta'] = self.meta_learner.get_statistics()

        return stats

    def shutdown(self):
        """关闭适配器"""
        logger.info("🔧 M1M4Adapter shutting down...")

        # 导出M4记忆
        if self.recursive_memory:
            try:
                export_path = Path(self.project_root) / "data" / "recursive_self_memory" / f"shutdown_{int(time.time())}.json"
                self.recursive_memory.export_memories(str(export_path))
                logger.info(f"   [M4] 记忆已备份: {export_path}")
            except Exception as e:
                logger.error(f"   [M4] 备份失败: {e}")

        logger.info("🔧 M1M4Adapter shutdown complete")


# ============================================================================
# 便捷函数
# ============================================================================

def create_m1m4_adapter(event_bus, project_root: str = None) -> M1M4Adapter:
    """
    创建并初始化M1-M4适配器

    Args:
        event_bus: AGI_Life_Engine的EventBus实例
        project_root: 项目根路径

    Returns:
        初始化完成的M1M4Adapter实例
    """
    adapter = M1M4Adapter(event_bus, project_root)
    adapter.initialize_all()
    return adapter
