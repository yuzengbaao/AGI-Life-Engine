#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Component Coordinator - 增强型组件协调器
==============================================

功能：
1. 集成组件版本管理
2. 集成状态迁移
3. 集成热替换协议
4. 提供快速回滚

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import logging
from typing import Dict, Any, Optional, List
from agi_component_coordinator import ComponentCoordinator as BaseCoordinator

logger = logging.getLogger(__name__)


class EnhancedComponentCoordinator(BaseCoordinator):
    """增强型组件协调器（支持热替换）"""

    def __init__(self, agi_system: Any, enable_hot_swap: bool = True):
        """
        初始化增强型组件协调器

        Args:
            agi_system: AGI系统实例
            enable_hot_swap: 是否启用热替换功能
        """
        # 初始化基类
        super().__init__(agi_system)

        # 热替换功能
        self.enable_hot_swap = enable_hot_swap

        if enable_hot_swap:
            from core.hot_swap_protocol import get_hot_swap_manager

            self.hot_swap_manager = get_hot_swap_manager()
            self.component_versions: Dict[str, str] = {}  # component_id -> version

            logger.info("[增强协调器] 热替换功能已启用")

            # 注册现有组件到热替换管理器
            self._register_components_for_hot_swap()

        logger.info("[增强协调器] 初始化完成（热替换增强）")

    def _register_components_for_hot_swap(self):
        """注册现有组件到热替换管理器"""
        for component_id in self.registry.keys():
            try:
                instance = self.registry[component_id]

                # 尝试获取版本
                version = "1.0.0"  # 默认版本
                if hasattr(instance, '__version__'):
                    version = str(instance.__version__)
                elif hasattr(instance, 'version'):
                    version = str(instance.version)

                # 注册到热替换管理器
                self.hot_swap_manager.register_component(
                    component_id=component_id,
                    instance=instance,
                    version=version
                )

                # 记录版本
                self.component_versions[component_id] = version

                logger.debug(
                    f"[增强协调器] 组件已注册热替换: "
                    f"{component_id} v{version}"
                )

            except Exception as e:
                logger.debug(
                    f"[增强协调器] 组件注册热替换失败（非致命）: "
                    f"{component_id}, {e}"
                )

    async def hot_swap_component(
        self,
        component_id: str,
        new_component_class: type,
        new_version: str = "1.0.0",
        **kwargs
    ) -> Dict[str, Any]:
        """
        热替换组件到新版本

        Args:
            component_id: 组件ID（如 "file", "doc", "openhands"）
            new_component_class: 新组件类
            new_version: 新版本号
            **kwargs: 其他参数

        Returns:
            Dict: 热替换结果
        """
        if not self.enable_hot_swap:
            return {
                "success": False,
                "error": "热替换功能未启用"
            }

        try:
            logger.info(f"[增强协调器] 热替换组件: {component_id} -> v{new_version}")

            # 获取当前组件版本
            current_version = self.component_versions.get(component_id, "1.0.0")

            # 执行热替换
            result = await self.hot_swap_manager.swap_component(
                component_id=component_id,
                new_component_class=new_component_class,
                new_version=new_version,
                current_version=current_version,
                **kwargs
            )

            # 如果成功，更新注册表
            if result.status.name == "completed":
                # 更新注册表
                new_instance = self.hot_swap_manager.get_component(component_id)
                if new_instance:
                    self.registry[component_id] = new_instance
                    self.component_versions[component_id] = new_version

                    logger.info(
                        f"[增强协调器] 组件热替换成功: "
                        f"{component_id} v{current_version} -> v{new_version}, "
                        f"耗时: {result.swap_time_ms:.1f}ms, "
                        f"中断: {result.downtime_ms:.1f}ms"
                    )

                    return {
                        "success": True,
                        "component_id": component_id,
                        "old_version": current_version,
                        "new_version": new_version,
                        "swap_time_ms": result.swap_time_ms,
                        "downtime_ms": result.downtime_ms,
                        "migrated_attributes": result.migrated_attributes
                    }

            # 失败或回滚
            return {
                "success": False,
                "error": result.error_message or "热替换失败",
                "component_id": component_id
            }

        except Exception as e:
            logger.error(f"[增强协调器] 热替换异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "component_id": component_id
            }

    def rollback_component(self, component_id: str) -> Dict[str, Any]:
        """
        回滚组件到上一个版本

        Args:
            component_id: 组件ID

        Returns:
            Dict: 回滚结果
        """
        if not self.enable_hot_swap:
            return {
                "success": False,
                "error": "热替换功能未启用"
            }

        try:
            logger.info(f"[增强协调器] 回滚组件: {component_id}")

            # 从热替换管理器回滚
            success = self.hot_swap_manager.protocol.rollback_hot_swap(component_id)

            if success:
                # 恢复注册表
                restored_instance = self.hot_swap_manager.get_component(component_id)
                if restored_instance:
                    old_version = self.hot_swap_manager.get_component_version(component_id)

                    self.registry[component_id] = restored_instance
                    self.component_versions[component_id] = old_version

                    logger.info(
                        f"[增强协调器] 组件回滚成功: "
                        f"{component_id} v{old_version}"
                    )

                    return {
                        "success": True,
                        "component_id": component_id,
                        "version": old_version
                    }

            return {
                "success": False,
                "error": "回滚失败"
            }

        except Exception as e:
            logger.error(f"[增强协调器] 回滚异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "component_id": component_id
            }

    def get_component_version(self, component_id: str) -> Optional[str]:
        """获取组件版本"""
        if self.enable_hot_swap:
            return self.hot_swap_manager.get_component_version(component_id)
        elif component_id in self.component_versions:
            return self.component_versions[component_id]
        else:
            return None

    def get_hot_swap_statistics(self) -> Dict[str, Any]:
        """获取热替换统计信息"""
        if not self.enable_hot_swap:
            return {"enabled": False}

        return self.hot_swap_manager.get_statistics()

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """获取增强统计信息（包含基类统计和热替换统计）"""
        # 获取基类统计
        base_stats = self.get_stats()

        # 获取热替换统计
        hot_swap_stats = self.get_hot_swap_statistics()

        # 合并统计
        enhanced_stats = base_stats.copy()
        enhanced_stats['hot_swap'] = hot_swap_stats

        if self.enable_hot_swap:
            enhanced_stats['component_versions'] = self.component_versions.copy()

        return enhanced_stats


def create_enhanced_coordinator(
    agi_system: Any,
    enable_hot_swap: bool = True
) -> EnhancedComponentCoordinator:
    """创建增强型组件协调器"""
    return EnhancedComponentCoordinator(
        agi_system=agi_system,
        enable_hot_swap=enable_hot_swap
    )
