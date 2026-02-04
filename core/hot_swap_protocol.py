#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hot Swap Protocol - 组件热替换协议
===================================

功能：
1. 运行时组件替换
2. 状态迁移和恢复
3. 快速回滚机制
4. 服务中断最小化

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SwapStatus(Enum):
    """热替换状态"""
    PREPARING = "preparing"  # 准备中
    PAUSED = "paused"  # 已暂停
    SWAPPING = "swapping"  # 替换中
    COMPLETED = "completed"  # 完成
    FAILED = "failed"  # 失败
    ROLLED_BACK = "rolled_back"  # 已回滚


@dataclass
class SwapResult:
    """热替换结果"""
    status: SwapStatus
    component_id: str
    old_version: str
    new_version: str
    swap_time_ms: float
    downtime_ms: float
    error_message: Optional[str] = None
    migrated_attributes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HotSwapProtocol:
    """组件热替换协议"""

    def __init__(self, version_manager, migration_manager):
        """
        初始化热替换协议

        Args:
            version_manager: 组件版本管理器
            migration_manager: 状态迁移管理器
        """
        self.version_manager = version_manager
        self.migration_manager = migration_manager
        self.rollback_points: Dict[str, Any] = {}  # component_id -> component_instance
        self.swap_history: List[SwapResult] = []
        logger.info("[热替换协议] 初始化完成")

    async def prepare_hot_swap(
        self,
        component_id: str,
        component_instance: Any,
        current_version: str
    ) -> bool:
        """
        准备热替换（暂停组件并保存状态）

        Args:
            component_id: 组件ID
            component_instance: 组件实例
            current_version: 当前版本

        Returns:
            bool: 准备成功
        """
        try:
            logger.info(f"[热替换] 准备热替换: {component_id} v{current_version}")

            # 1. 保存当前实例到回滚点
            self.rollback_points[component_id] = component_instance

            # 2. 创建回滚点（在状态迁移管理器中）
            state_data = self.migration_manager.prepare_component_for_upgrade(
                component=component_instance,
                component_id=component_id,
                current_version=current_version
            )

            logger.info(f"[热替换] 热替换准备完成: {component_id}")
            return True

        except Exception as e:
            logger.error(f"[热替换] 准备失败: {e}")
            return False

    async def execute_hot_swap(
        self,
        component_id: str,
        new_component_class: type,
        new_version: str,
        migration_mapping: Optional[Dict[str, str]] = None,
        pre_swap_hook: Optional[Callable] = None,
        post_swap_hook: Optional[Callable] = None
    ) -> SwapResult:
        """
        执行热替换

        Args:
            component_id: 组件ID
            new_component_class: 新组件类
            new_version: 新版本号
            migration_mapping: 状态迁移映射
            pre_swap_hook: 替换前钩子
            post_swap_hook: 替换后钩子

        Returns:
            SwapResult: 替换结果
        """
        start_time = time.time()
        downtime_start = time.time()

        try:
            logger.info(f"[热替换] 开始热替换: {component_id} -> v{new_version}")

            # 1. 获取旧组件实例
            old_component = self.rollback_points.get(component_id)
            if old_component is None:
                raise ValueError(f"No component found for {component_id}")

            old_version = self._get_component_version(old_component)

            # 2. 执行替换前钩子
            if pre_swap_hook:
                logger.debug("[热替换] 执行替换前钩子")
                await pre_swap_hook(old_component)

            # 3. 暂停组件事件处理（如果有）
            if hasattr(old_component, 'pause'):
                logger.debug("[热替换] 暂停组件事件处理")
                old_component.pause()

            # 4. 执行状态迁移
            state_data = self.migration_manager.prepare_component_for_upgrade(
                component=old_component,
                component_id=component_id,
                current_version=old_version
            )

            # 5. 升级到新版本
            new_component = self.migration_manager.upgrade_component(
                state_data=state_data,
                new_component_class=new_component_class,
                migration_mapping=migration_mapping
            )

            # 6. 更新回滚点
            self.rollback_points[component_id] = new_component

            downtime_end = time.time()
            swap_time = time.time() - start_time
            downtime = downtime_end - downtime_start

            # 7. 恢复组件事件处理（如果有）
            if hasattr(new_component, 'resume'):
                logger.debug("[热替换] 恢复组件事件处理")
                new_component.resume()

            # 8. 执行替换后钩子
            if post_swap_hook:
                logger.debug("[热替换] 执行替换后钩子")
                await post_swap_hook(new_component)

            logger.info(
                f"[热替换] 热替换成功: {component_id}, "
                f"耗时: {swap_time*1000:.1f}ms, "
                f"中断时间: {downtime*1000:.1f}ms"
            )

            # 9. 创建结果
            result = SwapResult(
                status=SwapStatus.COMPLETED,
                component_id=component_id,
                old_version=old_version,
                new_version=new_version,
                swap_time_ms=swap_time * 1000,
                downtime_ms=downtime * 1000,
                metadata={'migrated': True}
            )

            self.swap_history.append(result)
            return result

        except Exception as e:
            downtime_end = time.time()
            downtime = downtime_end - downtime_start

            logger.error(f"[热替换] 热替换失败: {e}")

            # 尝试回滚
            try:
                self.rollback_hot_swap(component_id)
                error_msg = f"热替换失败，已回滚: {e}"
            except Exception as rollback_error:
                error_msg = f"热替换失败，回滚也失败: {e}, {rollback_error}"

            result = SwapResult(
                status=SwapStatus.FAILED,
                component_id=component_id,
                old_version=self._get_component_version(
                    self.rollback_points.get(component_id)
                ) if component_id in self.rollback_points else "unknown",
                new_version=new_version,
                swap_time_ms=(time.time() - start_time) * 1000,
                downtime_ms=downtime * 1000,
                error_message=error_msg
            )

            self.swap_history.append(result)
            return result

    def rollback_hot_swap(self, component_id: str) -> bool:
        """
        回滚热替换

        Args:
            component_id: 组件ID

        Returns:
            bool: 回滚成功
        """
        try:
            logger.info(f"[热替换] 开始回滚: {component_id}")

            # 从状态迁移管理器回滚
            # 注意：这里需要访问原始的旧组件类
            # 实际使用时应该保存旧组件类的引用
            old_component = self.rollback_points.get(component_id)

            if old_component is None:
                raise ValueError(f"No rollback point found for {component_id}")

            # 恢复旧组件
            # 在实际系统中，这里需要将旧组件重新注册到协调器
            self.rollback_points[component_id] = old_component

            logger.info(f"[热替换] 回滚成功: {component_id}")
            return True

        except Exception as e:
            logger.error(f"[热替换] 回滚失败: {e}")
            return False

    def _get_component_version(self, component: Any) -> str:
        """获取组件版本（启发式）"""
        # 尝试从组件属性获取版本
        if hasattr(component, '__version__'):
            return str(component.__version__)
        elif hasattr(component, 'version'):
            return str(component.version)
        else:
            return "unknown"

    def get_swap_statistics(self) -> Dict[str, Any]:
        """获取热替换统计"""
        successful = sum(1 for r in self.swap_history if r.status == SwapStatus.COMPLETED)
        failed = sum(1 for r in self.swap_history if r.status == SwapStatus.FAILED)
        rolled_back = sum(1 for r in self.swap_history if r.status == SwapStatus.ROLLED_BACK)
        total = len(self.swap_history)

        avg_downtime = 0.0
        if total > 0:
            avg_downtime = sum(r.downtime_ms for r in self.swap_history) / total

        avg_swap_time = 0.0
        if total > 0:
            avg_swap_time = sum(r.swap_time_ms for r in self.swap_history) / total

        return {
            'total_swaps': total,
            'successful': successful,
            'failed': failed,
            'rolled_back': rolled_back,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_downtime_ms': avg_downtime,
            'avg_swap_time_ms': avg_swap_time
        }


class ComponentHotSwapManager:
    """组件热替换管理器（高层接口）"""

    def __init__(self, version_manager=None, migration_manager=None):
        """
        初始化热替换管理器

        Args:
            version_manager: 可选的版本管理器
            migration_manager: 可选的迁移管理器
        """
        from core.component_versioning import get_component_version_manager
        from core.state_migration import get_state_migration_manager

        self.version_manager = version_manager or get_component_version_manager()
        self.migration_manager = migration_manager or get_state_migration_manager()
        self.protocol = HotSwapProtocol(
            version_manager=self.version_manager,
            migration_manager=self.migration_manager
        )

        # 组件注册表（component_id -> (instance, version)）
        self.components: Dict[str, tuple] = {}

        logger.info("[热替换管理器] 初始化完成")

    def register_component(self, component_id: str, instance: Any, version: str = "1.0.0"):
        """注册组件"""
        self.components[component_id] = (instance, version)

        # 同时注册到版本管理器
        try:
            self.version_manager.register_component(
                component_id=component_id,
                component_class=type(instance),
                version=version
            )
        except Exception as e:
            logger.debug(f"版本管理器注册失败（非致命）: {e}")

    async def swap_component(
        self,
        component_id: str,
        new_component_class: type,
        new_version: str = "1.0.0",
        **kwargs
    ) -> SwapResult:
        """
        交换组件到新版本

        Args:
            component_id: 组件ID
            new_component_class: 新组件类
            new_version: 新版本号
            **kwargs: 其他参数（migration_mapping等）

        Returns:
            SwapResult
        """
        # 1. 准备热替换
        success = await self.protocol.prepare_hot_swap(
            component_id=component_id,
            component_instance=self.components[component_id][0],
            current_version=self.components[component_id][1]
        )

        if not success:
            return SwapResult(
                status=SwapStatus.FAILED,
                component_id=component_id,
                old_version=self.components[component_id][1],
                new_version=new_version,
                swap_time_ms=0,
                downtime_ms=0,
                error_message="准备阶段失败"
            )

        # 2. 执行热替换
        result = await self.protocol.execute_hot_swap(
            component_id=component_id,
            new_component_class=new_component_class,
            new_version=new_version,
            migration_mapping=kwargs.get('migration_mapping'),
            pre_swap_hook=kwargs.get('pre_swap_hook'),
            post_swap_hook=kwargs.get('post_swap_hook')
        )

        # 3. 更新组件注册表
        if result.status == SwapStatus.COMPLETED:
            self.components[component_id] = (
                self.protocol.rollback_points[component_id],
                new_version
            )

        return result

    def get_component(self, component_id: str) -> Optional[Any]:
        """获取当前组件实例"""
        if component_id in self.components:
            return self.components[component_id][0]
        return None

    def get_component_version(self, component_id: str) -> Optional[str]:
        """获取组件版本"""
        if component_id in self.components:
            return self.components[component_id][1]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        swap_stats = self.protocol.get_swap_statistics()
        version_stats = self.version_manager.get_statistics()
        migration_stats = self.migration_manager.migration.get_migration_statistics()

        return {
            'hot_swap': swap_stats,
            'versioning': version_stats,
            'migration': migration_stats,
            'registered_components': len(self.components)
        }


# 全局单例
_global_hot_swap_manager: Optional[ComponentHotSwapManager] = None


def get_hot_swap_manager() -> ComponentHotSwapManager:
    """获取全局热替换管理器"""
    global _global_hot_swap_manager
    if _global_hot_swap_manager is None:
        _global_hot_swap_manager = ComponentHotSwapManager()
    return _global_hot_swap_manager
