#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Component Versioning System - 组件版本管理系统
===========================================

功能：
1. 组件版本跟踪（语义化版本号）
2. API兼容性检查
3. 依赖关系管理
4. 版本升级路径规划

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
import importlib
import sys

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """兼容性级别"""
    FULL = "full"  # 完全兼容，无需迁移
    MINOR = "minor"  # 小版本不兼容，需要简单迁移
    MAJOR = "major"  # 大版本不兼容，需要复杂迁移
    INCOMPATIBLE = "incompatible"  # 完全不兼容


@dataclass
class APIDefinition:
    """API定义"""
    name: str
    signature: str  # 方法签名
    parameters: Dict[str, str]  # 参数名到类型的映射
    return_type: str
    is_async: bool = False
    docstring: Optional[str] = None

    def __hash__(self):
        """基于签名生成哈希"""
        sig_str = f"{self.name}:{self.signature}:{self.is_async}"
        return hash(sig_str)

    def __eq__(self, other):
        """API定义相等性比较"""
        if not isinstance(other, APIDefinition):
            return False
        return hash(self) == hash(other)


@dataclass
class StateSchema:
    """状态Schema定义"""
    attributes: Dict[str, str]  # 属性名到类型的映射
    required_attributes: Set[str] = field(default_factory=set)
    optional_attributes: Set[str] = field(default_factory=set)

    def validate(self, state_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证状态是否符合Schema

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 检查必需属性
        for attr in self.required_attributes:
            if attr not in state_dict:
                errors.append(f"Missing required attribute: {attr}")

        # 检查类型（简单验证）
        for attr, expected_type in self.attributes.items():
            if attr in state_dict:
                value = state_dict[attr]
                # 简化版类型检查
                if expected_type == "int" and not isinstance(value, int):
                    errors.append(f"Attribute {attr} expected int, got {type(value).__name__}")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Attribute {attr} expected float, got {type(value).__name__}")
                elif expected_type == "str" and not isinstance(value, str):
                    errors.append(f"Attribute {attr} expected str, got {type(value).__name__}")
                elif expected_type == "list" and not isinstance(value, list):
                    errors.append(f"Attribute {attr} expected list, got {type(value).__name__}")
                elif expected_type == "dict" and not isinstance(value, dict):
                    errors.append(f"Attribute {attr} expected dict, got {type(value).__name__}")

        return len(errors) == 0, errors


@dataclass
class ComponentVersion:
    """组件版本信息"""
    component_id: str  # 组件唯一标识
    version: str  # 语义化版本号 (e.g., "1.2.3")
    api_hash: str  # API接口哈希
    state_schema: StateSchema  # 状态结构定义
    dependencies: List[str] = field(default_factory=list)  # 依赖的组件ID列表
    api_definitions: List[APIDefinition] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.created_at:
            import time
            self.created_at = time.time()

    def get_version_tuple(self) -> Tuple[int, int, int]:
        """获取版本号元组"""
        parts = self.version.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {self.version}")

        return (int(parts[0]), int(parts[1]), int(parts[2]))

    def is_compatible_with(self, other: 'ComponentVersion') -> CompatibilityLevel:
        """
        检查与另一个版本的兼容性

        规则：
        - 主版本号相同：完全兼容
        - 主版本号相同，API哈希相同：完全兼容
        - 主版本号相同，API哈希不同：小版本不兼容
        - 主版本号不同：大版本不兼容
        """
        try:
            self_major, self_minor, self_patch = self.get_version_tuple()
            other_major, other_minor, other_patch = other.get_version_tuple()
        except ValueError:
            return CompatibilityLevel.INCOMPATIBLE

        # 主版本号不同 -> 不兼容
        if self_major != other_major:
            return CompatibilityLevel.MAJOR

        # API哈希相同 -> 完全兼容
        if self.api_hash == other.api_hash:
            return CompatibilityLevel.FULL

        # API哈希不同 -> 需要迁移
        if self.api_hash != other.api_hash:
            return CompatibilityLevel.MINOR

        return CompatibilityLevel.FULL


class ComponentVersionManager:
    """组件版本管理器"""

    def __init__(self):
        self.registry: Dict[str, ComponentVersion] = {}
        self.version_history: Dict[str, List[str]] = {}  # component_id -> [versions]
        logger.info("[组件版本管理] 初始化完成")

    def register_component(
        self,
        component_id: str,
        component_class: type,
        version: str = "1.0.0"
    ) -> ComponentVersion:
        """
        注册组件及其版本信息

        Args:
            component_id: 组件唯一标识
            component_class: 组件类
            version: 版本号（默认1.0.0）

        Returns:
            ComponentVersion: 版本信息对象
        """
        # 提取API定义
        api_definitions = self._extract_api_definitions(component_class)

        # 计算API哈希
        api_hash = self._compute_api_hash(api_definitions)

        # 提取状态Schema
        state_schema = self._extract_state_schema(component_class)

        # 创建版本信息
        version_info = ComponentVersion(
            component_id=component_id,
            version=version,
            api_hash=api_hash,
            state_schema=state_schema,
            api_definitions=api_definitions
        )

        # 注册到版本历史
        if component_id not in self.version_history:
            self.version_history[component_id] = []
        self.version_history[component_id].append(version)

        # 注册到版本注册表
        key = f"{component_id}:{version}"
        self.registry[key] = version_info

        logger.info(
            f"[组件版本管理] 注册组件: {component_id} v{version}, "
            f"APIs: {len(api_definitions)}, "
            f"State attrs: {len(state_schema.attributes)}"
        )

        return version_info

    def get_version(self, component_id: str, version: str = "latest") -> Optional[ComponentVersion]:
        """
        获取组件版本信息

        Args:
            component_id: 组件ID
            version: 版本号（"latest"表示最新版本）

        Returns:
            ComponentVersion or None
        """
        if version == "latest":
            if component_id in self.version_history and self.version_history[component_id]:
                latest_version = self.version_history[component_id][-1]
                key = f"{component_id}:{latest_version}"
                return self.registry.get(key)
            return None

        key = f"{component_id}:{version}"
        return self.registry.get(key)

    def check_compatibility(
        self,
        from_version: ComponentVersion,
        to_version: ComponentVersion
    ) -> CompatibilityLevel:
        """
        检查两个版本是否兼容

        Returns:
            CompatibilityLevel
        """
        return from_version.is_compatible_with(to_version)

    def get_upgrade_path(
        self,
        component_id: str,
        from_version: str,
        to_version: str
    ) -> List[str]:
        """
        获取升级路径

        Returns:
            版本号列表，表示升级路径
        """
        if component_id not in self.version_history:
            return []

        all_versions = self.version_history[component_id]

        # 查找from_version和to_version的索引
        try:
            from_idx = all_versions.index(from_version)
            to_idx = all_versions.index(to_version)
        except ValueError:
            return []

        if from_idx >= to_idx:
            return []  # 不是升级路径

        # 返回升级路径
        return all_versions[from_idx+1:to_idx+1]

    def _extract_api_definitions(self, component_class: type) -> List[APIDefinition]:
        """提取组件的API定义"""
        api_definitions = []

        # 获取所有公共方法
        for name, method in inspect.getmembers(component_class, predicate=inspect.isfunction):
            if name.startswith('_'):
                continue  # 跳过私有方法

            try:
                sig = inspect.signature(method)
                params = {}
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    annotation = param.annotation
                    if annotation == inspect.Parameter.empty:
                        annotation = 'Any'
                    params[param_name] = str(annotation)

                return_annotation = sig.return_annotation
                if return_annotation == inspect.Signature.empty:
                    return_annotation = 'None'
                else:
                    return_annotation = str(return_annotation)

                api_def = APIDefinition(
                    name=name,
                    signature=str(sig),
                    parameters=params,
                    return_type=return_annotation,
                    is_async=inspect.iscoroutinefunction(method),
                    docstring=inspect.getdoc(method)
                )
                api_definitions.append(api_def)

            except Exception as e:
                logger.debug(f"Failed to extract API for {name}: {e}")
                continue

        return api_definitions

    def _compute_api_hash(self, api_definitions: List[APIDefinition]) -> str:
        """计算API哈希"""
        # 基于API定义生成哈希
        api_strs = sorted([f"{api.name}:{api.signature}" for api in api_definitions])
        combined = "|".join(api_strs)
        return hashlib.md5(combined.encode()).hexdigest()

    def _extract_state_schema(self, component_class: type) -> StateSchema:
        """提取组件的状态Schema"""
        attributes = {}
        required = set()
        optional = set()

        # 获取类属性（从__init__方法推断）
        if hasattr(component_class, '__init__'):
            try:
                sig = inspect.signature(component_class.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue

                    # 推断类型
                    annotation = param.annotation
                    if annotation == inspect.Parameter.empty:
                        attr_type = 'Any'
                    else:
                        attr_type = str(annotation)

                    attributes[param_name] = attr_type

                    # 检查是否必需
                    if param.default == inspect.Parameter.empty:
                        required.add(param_name)
                    else:
                        optional.add(param_name)
            except Exception as e:
                logger.debug(f"Failed to extract schema from __init__: {e}")

        return StateSchema(
            attributes=attributes,
            required_attributes=required,
            optional_attributes=optional
        )

    def list_components(self) -> List[str]:
        """列出所有已注册的组件ID"""
        return list(set([key.split(':')[0] for key in self.registry.keys()]))

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_components': len(self.version_history),
            'total_versions': len(self.registry),
            'components_with_history': len([v for v in self.version_history.values() if len(v) > 1])
        }


# 全局单例
_global_version_manager: Optional[ComponentVersionManager] = None


def get_component_version_manager() -> ComponentVersionManager:
    """获取全局组件版本管理器"""
    global _global_version_manager
    if _global_version_manager is None:
        _global_version_manager = ComponentVersionManager()
    return _global_version_manager
