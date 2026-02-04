#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime Code Loader - 运行时代码加载器
==========================================

功能：
1. 从字符串动态加载模块
2. 热重载单个函数
3. 运行时替换类方法
4. 模块版本控制

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import ast
import importlib.util
import logging
import sys
import types
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModuleLoadRecord:
    """模块加载记录"""
    module_name: str
    source_code: str
    timestamp: str
    version: str
    functions_loaded: List[str]
    classes_loaded: List[str]
    metadata: Dict[str, Any]


class RuntimeCodeLoader:
    """
    运行时代码加载器

    支持从字符串动态加载Python模块和函数
    """

    def __init__(self):
        """初始化运行时代码加载器"""
        self.loaded_modules: Dict[str, types.ModuleType] = {}
        self.load_history: List[ModuleLoadRecord] = []
        logger.info("[运行时加载器] 初始化完成")

    def load_module_from_string(
        self,
        module_name: str,
        code: str,
        version: str = "1.0.0"
    ) -> Tuple[bool, Optional[types.ModuleType], Optional[str]]:
        """
        从字符串加载模块

        Args:
            module_name: 模块名称
            code: 模块源代码
            version: 模块版本

        Returns:
            (成功, 模块对象, 错误信息)
        """
        try:
            logger.info(f"[运行时加载器] 加载模块: {module_name} v{version}")

            # 创建模块规范
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            if spec is None:
                return False, None, "无法创建模块规范"

            # 创建模块
            module = importlib.util.module_from_spec(spec)

            # 执行代码
            exec(code, module.__dict__)

            # 注册到sys.modules
            sys.modules[module_name] = module

            # 保存到已加载列表
            self.loaded_modules[module_name] = module

            # 记录加载历史
            record = ModuleLoadRecord(
                module_name=module_name,
                source_code=code,
                timestamp=datetime.now().isoformat(),
                version=version,
                functions_loaded=self._extract_functions(code),
                classes_loaded=self._extract_classes(code),
                metadata={'size': len(code)}
            )

            self.load_history.append(record)

            logger.info(
                f"[运行时加载器] ✅ 模块加载成功: "
                f"{module_name} ({len(code)} bytes, "
                f"{len(record.functions_loaded)} functions, "
                f"{len(record.classes_loaded)} classes)"
            )

            return True, module, None

        except Exception as e:
            error_msg = f"模块加载失败: {e}"
            logger.error(f"[运行时加载器] {error_msg}")
            return False, None, error_msg

    def hot_reload_function(
        self,
        module_name: str,
        function_name: str,
        new_code: str
    ) -> Tuple[bool, Optional[str]]:
        """
        热重载单个函数

        Args:
            module_name: 模块名称
            function_name: 函数名称
            new_code: 新函数代码

        Returns:
            (成功, 错误信息)
        """
        try:
            logger.info(
                f"[运行时加载器] 热重载函数: "
                f"{module_name}.{function_name}"
            )

            # 获取模块
            module = sys.modules.get(module_name)
            if module is None:
                return False, f"模块未找到: {module_name}"

            # 解析新代码
            tree = ast.parse(new_code)

            # 提取函数定义
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    func_def = node
                    break

            if func_def is None:
                return False, f"函数未找到: {function_name}"

            # 编译函数
            code_obj = compile(
                ast.Module(body=[func_def], type_ignores=[]),
                '<string>',
                'exec'
            )

            # 执行并获取函数
            namespace = {}
            exec(code_obj, namespace)

            new_function = namespace.get(function_name)
            if new_function is None:
                return False, "编译后未找到函数"

            # 替换函数
            setattr(module, function_name, new_function)

            # 更新加载历史
            for record in reversed(self.load_history):
                if record.module_name == module_name:
                    if function_name not in record.functions_loaded:
                        record.functions_loaded.append(function_name)
                    break

            logger.info(
                f"[运行时加载器] ✅ 函数热重载成功: "
                f"{module_name}.{function_name}"
            )

            return True, None

        except Exception as e:
            error_msg = f"函数热重载失败: {e}"
            logger.error(f"[运行时加载器] {error_msg}")
            return False, error_msg

    def hot_reload_method(
        self,
        module_name: str,
        class_name: str,
        method_name: str,
        new_code: str
    ) -> Tuple[bool, Optional[str]]:
        """
        热重载类方法

        Args:
            module_name: 模块名称
            class_name: 类名称
            method_name: 方法名称
            new_code: 新方法代码

        Returns:
            (成功, 错误信息)
        """
        try:
            logger.info(
                f"[运行时加载器] 热重载方法: "
                f"{module_name}.{class_name}.{method_name}"
            )

            # 获取模块
            module = sys.modules.get(module_name)
            if module is None:
                return False, f"模块未找到: {module_name}"

            # 获取类
            target_class = getattr(module, class_name, None)
            if target_class is None:
                return False, f"类未找到: {class_name}"

            # 编译新方法
            namespace = {}
            exec(new_code, namespace)

            new_method = namespace.get(method_name)
            if new_method is None:
                return False, f"编译后未找到方法: {method_name}"

            # 替换方法
            setattr(target_class, method_name, new_method)

            logger.info(
                f"[运行时加载器] ✅ 方法热重载成功: "
                f"{class_name}.{method_name}"
            )

            return True, None

        except Exception as e:
            error_msg = f"方法热重载失败: {e}"
            logger.error(f"[运行时加载器] {error_msg}")
            return False, error_msg

    def reload_module(
        self,
        module_name: str,
        new_code: str
    ) -> Tuple[bool, Optional[str]]:
        """
        完全重载模块

        Args:
            module_name: 模块名称
            new_code: 新模块代码

        Returns:
            (成功, 错误信息)
        """
        try:
            logger.info(f"[运行时加载器] 重载模块: {module_name}")

            # 获取旧模块
            old_module = sys.modules.get(module_name)

            # 加载新模块
            success, new_module, error = self.load_module_from_string(
                module_name=module_name,
                code=new_code,
                version=self._get_next_version(module_name)
            )

            if not success:
                return False, error

            # 保留旧模块的引用（用于回滚）
            if old_module:
                if not hasattr(new_module, '_old_version'):
                    new_module._old_version = old_module

            logger.info(
                f"[运行时加载器] ✅ 模块重载成功: {module_name}"
            )

            return True, None

        except Exception as e:
            error_msg = f"模块重载失败: {e}"
            logger.error(f"[运行时加载器] {error_msg}")
            return False, error_msg

    def rollback_module(self, module_name: str) -> Tuple[bool, Optional[str]]:
        """
        回滚模块到上一个版本

        Args:
            module_name: 模块名称

        Returns:
            (成功, 错误信息)
        """
        try:
            logger.info(f"[运行时加载器] 回滚模块: {module_name}")

            # 获取当前模块
            current_module = sys.modules.get(module_name)
            if current_module is None:
                return False, f"模块未找到: {module_name}"

            # 检查是否有旧版本
            if not hasattr(current_module, '_old_version'):
                return False, "没有可回滚的版本"

            old_module = current_module._old_version

            # 恢复旧模块
            sys.modules[module_name] = old_module
            self.loaded_modules[module_name] = old_module

            logger.info(
                f"[运行时加载器] ✅ 模块回滚成功: {module_name}"
            )

            return True, None

        except Exception as e:
            error_msg = f"模块回滚失败: {e}"
            logger.error(f"[运行时加载器] {error_msg}")
            return False, error_msg

    def _extract_functions(self, code: str) -> List[str]:
        """提取代码中的函数名"""
        try:
            tree = ast.parse(code)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            return functions
        except Exception:
            return []

    def _extract_classes(self, code: str) -> List[str]:
        """提取代码中的类名"""
        try:
            tree = ast.parse(code)
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            return classes
        except Exception:
            return []

    def _get_next_version(self, module_name: str) -> str:
        """获取下一个版本号"""
        for record in reversed(self.load_history):
            if record.module_name == module_name:
                # 简单版本递增
                parts = record.version.split('.')
                if len(parts) >= 3:
                    parts[2] = str(int(parts[2]) + 1)
                    return '.'.join(parts)
        return "1.0.1"

    def get_loaded_modules(self) -> List[str]:
        """获取已加载的模块列表"""
        return list(self.loaded_modules.keys())

    def get_load_statistics(self) -> Dict[str, Any]:
        """获取加载统计信息"""
        return {
            'total_modules_loaded': len(self.loaded_modules),
            'total_loads': len(self.load_history),
            'modules_by_version': {
                record.module_name: record.version
                for record in self.load_history
            },
            'recent_loads': [
                {
                    'module': r.module_name,
                    'timestamp': r.timestamp,
                    'version': r.version
                }
                for r in self.load_history[-5:]
            ]
        }


# 全局单例
_global_loader: Optional[RuntimeCodeLoader] = None


def get_runtime_code_loader() -> RuntimeCodeLoader:
    """获取全局运行时代码加载器"""
    global _global_loader
    if _global_loader is None:
        _global_loader = RuntimeCodeLoader()
    return _global_loader
