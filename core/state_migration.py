#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Migration Protocol - 状态迁移协议
====================================

功能：
1. 状态序列化与反序列化
2. 跨版本状态迁移
3. 状态验证
4. 回滚点管理

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import pickle
import logging
import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MigrationRecord:
    """迁移记录"""
    from_version: str
    to_version: str
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    migrated_attributes: List[str] = field(default_factory=list)
    skipped_attributes: List[str] = field(default_factory=list)


@dataclass
class RollbackPoint:
    """回滚点"""
    component_id: str
    version: str
    state_data: bytes  # 序列化的状态数据
    state_schema_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateMigration:
    """状态迁移协议"""

    def __init__(self):
        self.migration_history: List[MigrationRecord] = []
        self.rollback_points: Dict[str, RollbackPoint] = {}  # component_id -> RollbackPoint
        logger.info("[状态迁移] 初始化完成")

    def serialize_state(self, component: Any) -> bytes:
        """
        序列化组件状态

        Args:
            component: 组件实例

        Returns:
            bytes: 序列化的状态数据
        """
        try:
            # 提取状态
            state_dict = self._extract_state(component)

            # 添加元数据
            state_dict['_metadata'] = {
                'component_type': type(component).__name__,
                'module': type(component).__module__,
                'timestamp': datetime.now().isoformat()
            }

            # 序列化
            state_data = pickle.dumps(state_dict)

            logger.debug(
                f"[状态迁移] 序列化状态: {type(component).__name__}, "
                f"size: {len(state_data)} bytes"
            )

            return state_data

        except Exception as e:
            logger.error(f"[状态迁移] 序列化失败: {e}")
            raise

    def deserialize_state(
        self,
        state_data: bytes,
        target_class: type,
        validate: bool = True
    ) -> Any:
        """
        反序列化状态到目标类

        Args:
            state_data: 序列化的状态数据
            target_class: 目标类
            validate: 是否验证状态

        Returns:
            Any: 反序列化的组件实例
        """
        try:
            # 反序列化
            state_dict = pickle.loads(state_data)

            # 移除元数据
            metadata = state_dict.pop('_metadata', {})

            # 验证类型匹配
            if validate and metadata.get('component_type') != target_class.__name__:
                logger.warning(
                    f"[状态迁移] 类型不匹配: "
                    f"expected={metadata.get('component_type')}, "
                    f"got={target_class.__name__}"
                )

            # 创建实例并设置状态
            instance = target_class.__new__(target_class)

            # 设置属性
            for attr_name, attr_value in state_dict.items():
                setattr(instance, attr_name, attr_value)

            logger.debug(
                f"[状态迁移] 反序列化状态: {target_class.__name__}, "
                f"attrs: {len(state_dict)}"
            )

            return instance

        except Exception as e:
            logger.error(f"[状态迁移] 反序列化失败: {e}")
            raise

    def migrate_state(
        self,
        old_state: Dict[str, Any],
        old_schema: Dict[str, str],
        new_schema: Dict[str, str],
        mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        迁移状态到新Schema

        Args:
            old_state: 旧状态
            old_schema: 旧Schema（属性名到类型的映射）
            new_schema: 新Schema
            mapping: 可选的属性映射（旧属性名 -> 新属性名）

        Returns:
            Dict: 迁移后的状态
        """
        migrated_state = {}
        mapping = mapping or {}

        # 1. 复制匹配的属性
        for attr_name, attr_value in old_state.items():
            # 检查是否有映射
            new_attr_name = mapping.get(attr_name, attr_name)

            # 检查新Schema中是否需要该属性
            if new_attr_name in new_schema:
                # 简化版类型检查（实际应该更严格）
                migrated_state[new_attr_name] = attr_value

        # 2. 添加新Schema中新增的属性（使用默认值）
        for attr_name, attr_type in new_schema.items():
            if attr_name not in migrated_state:
                migrated_state[attr_name] = self._get_default_value(attr_type)

        logger.debug(
            f"[状态迁移] 状态迁移: "
            f"{len(old_state)} -> {len(migrated_state)} attrs"
        )

        return migrated_state

    def validate_state(
        self,
        state: Dict[str, Any],
        schema: Dict[str, str],
        required_attrs: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证状态

        Args:
            state: 状态字典
            schema: Schema定义
            required_attrs: 必需属性列表

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 检查必需属性
        if required_attrs:
            for attr in required_attrs:
                if attr not in state:
                    errors.append(f"Missing required attribute: {attr}")

        # 检查属性类型（简化版）
        for attr_name, expected_type in schema.items():
            if attr_name in state:
                value = state[attr_name]
                if not self._check_type(value, expected_type):
                    errors.append(
                        f"Attribute {attr_name} type mismatch: "
                        f"expected {expected_type}, got {type(value).__name__}"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    def create_rollback_point(
        self,
        component: Any,
        version: str,
        component_id: str
    ) -> RollbackPoint:
        """
        创建回滚点

        Args:
            component: 组件实例
            version: 当前版本
            component_id: 组件ID

        Returns:
            RollbackPoint
        """
        # 序列化状态
        state_data = self.serialize_state(component)

        # 计算Schema哈希
        state_dict = pickle.loads(state_data)
        schema_hash = self._compute_schema_hash(state_dict)

        # 创建回滚点
        rollback_point = RollbackPoint(
            component_id=component_id,
            version=version,
            state_data=state_data,
            state_schema_hash=schema_hash,
            timestamp=datetime.now().isoformat()
        )

        # 保存回滚点
        self.rollback_points[component_id] = rollback_point

        logger.info(
            f"[状态迁移] 创建回滚点: {component_id} v{version}, "
            f"size: {len(state_data)} bytes"
        )

        return rollback_point

    def rollback_to_point(
        self,
        component_id: str,
        target_class: type
    ) -> Any:
        """
        回滚到指定回滚点

        Args:
            component_id: 组件ID
            target_class: 目标类

        Returns:
            Any: 回滚后的组件实例
        """
        if component_id not in self.rollback_points:
            raise ValueError(f"No rollback point found for {component_id}")

        rollback_point = self.rollback_points[component_id]

        # 反序列化状态
        component = self.deserialize_state(
            rollback_point.state_data,
            target_class
        )

        logger.info(
            f"[状态迁移] 回滚完成: {component_id} -> v{rollback_point.version}"
        )

        return component

    def _extract_state(self, component: Any) -> Dict[str, Any]:
        """提取组件状态"""
        state_dict = {}

        # 获取实例属性
        for attr_name in dir(component):
            if attr_name.startswith('_'):
                continue  # 跳过私有属性

            try:
                attr_value = getattr(component, attr_name)

                # 跳过方法和特殊属性
                if callable(attr_value):
                    continue
                if attr_name.startswith('__') and attr_name.endswith('__'):
                    continue

                state_dict[attr_name] = attr_value

            except Exception as e:
                logger.debug(f"Failed to extract attribute {attr_name}: {e}")
                continue

        return state_dict

    def _get_default_value(self, attr_type: str) -> Any:
        """获取类型的默认值"""
        defaults = {
            'int': 0,
            'float': 0.0,
            'str': '',
            'bool': False,
            'list': [],
            'dict': {},
            'set': set(),
            'tuple': (),
            'Any': None,
        }
        return defaults.get(attr_type, None)

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查类型是否匹配"""
        type_map = {
            'int': int,
            'float': (int, float),
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # 未知类型，跳过检查

        return isinstance(value, expected_python_type)

    def _compute_schema_hash(self, state_dict: Dict[str, Any]) -> str:
        """计算Schema哈希"""
        # 基于属性名和类型生成哈希
        items = sorted(state_dict.items())
        schema_str = "|".join([f"{k}:{type(v).__name__}" for k, v in items])
        return hashlib.md5(schema_str.encode()).hexdigest()

    def get_migration_statistics(self) -> Dict[str, Any]:
        """获取迁移统计信息"""
        successful = sum(1 for r in self.migration_history if r.success)
        total = len(self.migration_history)

        return {
            'total_migrations': total,
            'successful_migrations': successful,
            'failed_migrations': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'rollback_points': len(self.rollback_points)
        }


class StateMigrationManager:
    """状态迁移管理器（高层接口）"""

    def __init__(self):
        self.migration = StateMigration()
        logger.info("[状态迁移管理器] 初始化完成")

    def prepare_component_for_upgrade(
        self,
        component: Any,
        component_id: str,
        current_version: str
    ) -> bytes:
        """
        为组件升级做准备

        Returns:
            bytes: 序列化的状态数据
        """
        # 创建回滚点
        self.migration.create_rollback_point(
            component=component,
            version=current_version,
            component_id=component_id
        )

        # 序列化状态
        state_data = self.migration.serialize_state(component)

        return state_data

    def upgrade_component(
        self,
        state_data: bytes,
        new_component_class: type,
        migration_mapping: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        升级组件到新版本

        Args:
            state_data: 序列化的旧状态
            new_component_class: 新组件类
            migration_mapping: 状态迁移映射

        Returns:
            Any: 升级后的组件实例
        """
        # 反序列化到新类
        new_component = self.migration.deserialize_state(
            state_data=state_data,
            target_class=new_component_class,
            validate=False  # 升级时暂时不验证
        )

        return new_component

    def rollback_component(
        self,
        component_id: str,
        old_component_class: type
    ) -> Any:
        """
        回滚组件到旧版本

        Args:
            component_id: 组件ID
            old_component_class: 旧组件类

        Returns:
            Any: 回滚后的组件实例
        """
        return self.migration.rollback_to_point(
            component_id=component_id,
            target_class=old_component_class
        )


# 全局单例
_global_migration_manager: Optional[StateMigrationManager] = None


def get_state_migration_manager() -> StateMigrationManager:
    """获取全局状态迁移管理器"""
    global _global_migration_manager
    if _global_migration_manager is None:
        _global_migration_manager = StateMigrationManager()
    return _global_migration_manager
