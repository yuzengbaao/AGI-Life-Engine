#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function Level Patcher - 函数级补丁器
========================================

功能：
1. 运行时替换类的方法
2. 函数级代码热替换
3. 替换验证和回滚
4. 替换历史记录

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import ast
import inspect
import logging
import sys
import types
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FunctionPatchRecord:
    """函数补丁记录"""
    class_name: str
    method_name: str
    old_code: str
    new_code: str
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FunctionLevelPatcher:
    """函数级补丁器 - 运行时方法替换"""

    def __init__(self):
        """初始化函数级补丁器"""
        self.patch_history: List[FunctionPatchRecord] = []
        self.rollback_stack: List[FunctionPatchRecord] = []
        logger.info("[函数补丁器] 初始化完成")

    def locate_class(self, class_name: str, module_name: Optional[str] = None) -> Optional[type]:
        """
        定位目标类

        Args:
            class_name: 类名
            module_name: 模块名（可选，用于缩小搜索范围）

        Returns:
            类对象或None
        """
        try:
            # 如果指定了模块，直接从模块获取
            if module_name:
                if module_name not in sys.modules:
                    import importlib
                    module = importlib.import_module(module_name)
                else:
                    module = sys.modules[module_name]

                return getattr(module, class_name, None)

            # 否则在所有已加载的模块中搜索
            for module in sys.modules.values():
                if module is None:
                    continue

                try:
                    cls = getattr(module, class_name, None)
                    if cls and isinstance(cls, type):
                        return cls
                except Exception:
                    continue

            logger.warning(f"[函数补丁器] 类未找到: {class_name}")
            return None

        except Exception as e:
            logger.error(f"[函数补丁器] 定位类失败: {e}")
            return None

    def replace_method(
        self,
        class_name: str,
        method_name: str,
        new_code: str,
        module_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        替换单个方法

        Args:
            class_name: 类名
            method_name: 方法名
            new_code: 新方法代码
            module_name: 模块名（可选）

        Returns:
            (成功与否, 错误信息)
        """
        try:
            logger.info(f"[函数补丁器] 替换方法: {class_name}.{method_name}")

            # 1. 定位目标类
            target_class = self.locate_class(class_name, module_name)
            if target_class is None:
                error_msg = f"类未找到: {class_name}"
                logger.error(f"[函数补丁器] {error_msg}")
                return False, error_msg

            # 2. 保存旧方法
            old_method = getattr(target_class, method_name, None)
            if old_method is None:
                error_msg = f"方法未找到: {class_name}.{method_name}"
                logger.error(f"[函数补丁器] {error_msg}")
                return False, error_msg

            # 获取旧方法代码
            try:
                old_code = inspect.getsource(old_method)
            except (OSError, TypeError):
                # 无法获取源代码（可能是内置方法），使用签名代替
                old_code = f"# 无法获取源代码\n{inspect.signature(old_method)}"

            # 3. 编译新代码
            namespace = {'__name__': f'{class_name}_patch'}

            try:
                exec(new_code, namespace)
            except Exception as e:
                error_msg = f"代码编译失败: {e}"
                logger.error(f"[函数补丁器] {error_msg}")
                return False, error_msg

            # 4. 提取新方法
            new_method = namespace.get(method_name)
            if new_method is None:
                error_msg = f"编译后的代码中未找到方法: {method_name}"
                logger.error(f"[函数补丁器] {error_msg}")
                return False, error_msg

            # 5. 替换方法
            setattr(target_class, method_name, new_method)

            # 6. 验证替换成功
            replaced_method = getattr(target_class, method_name, None)
            if replaced_method is None or replaced_method is not new_method:
                error_msg = "方法替换验证失败"
                logger.error(f"[函数补丁器] {error_msg}")
                return False, error_msg

            # 7. 记录补丁
            record = FunctionPatchRecord(
                class_name=class_name,
                method_name=method_name,
                old_code=old_code,
                new_code=new_code,
                timestamp=datetime.now().isoformat(),
                success=True,
                metadata={
                    'module': module_name,
                    'old_method_id': id(old_method),
                    'new_method_id': id(new_method)
                }
            )

            self.patch_history.append(record)
            self.rollback_stack.append(record)

            logger.info(
                f"[函数补丁器] ✅ 方法已替换: "
                f"{class_name}.{method_name} "
                f"(旧ID: {id(old_method)}, 新ID: {id(new_method)})"
            )

            return True, None

        except Exception as e:
            error_msg = f"方法替换异常: {e}"
            logger.error(f"[函数补丁器] {error_msg}")
            return False, error_msg

    def rollback_last_patch(self) -> Tuple[bool, Optional[str]]:
        """
        回滚最后一次补丁

        Returns:
            (成功与否, 错误信息)
        """
        if not self.rollback_stack:
            return False, "没有可回滚的补丁"

        try:
            # 获取最后一次补丁记录
            record = self.rollback_stack.pop()

            # 定位类
            target_class = self.locate_class(record.class_name)
            if target_class is None:
                return False, f"类未找到: {record.class_name}"

            # 恢复旧方法
            namespace = {'__name__': f'{record.class_name}_rollback'}
            exec(record.old_code, namespace)

            old_method = namespace.get(record.method_name)
            if old_method is None:
                return False, f"无法恢复旧方法: {record.method_name}"

            # 设置回滚的方法
            setattr(target_class, record.method_name, old_method)

            logger.info(
                f"[函数补丁器] ✅ 方法已回滚: "
                f"{record.class_name}.{record.method_name}"
            )

            return True, None

        except Exception as e:
            error_msg = f"回滚失败: {e}"
            logger.error(f"[函数补丁器] {error_msg}")
            return False, error_msg

    def verify_method(
        self,
        class_name: str,
        method_name: str,
        expected_signature: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        验证方法存在性和签名

        Args:
            class_name: 类名
            method_name: 方法名
            expected_signature: 期望的签名（可选）

        Returns:
            (验证成功, 验证详情)
        """
        result = {
            'class_exists': False,
            'method_exists': False,
            'signature': None,
            'signature_match': False,
            'method_id': None
        }

        try:
            # 定位类
            target_class = self.locate_class(class_name)
            if target_class is None:
                return False, result

            result['class_exists'] = True

            # 检查方法
            method = getattr(target_class, method_name, None)
            if method is None:
                return False, result

            result['method_exists'] = True
            result['method_id'] = id(method)

            # 获取签名
            try:
                sig = inspect.signature(method)
                result['signature'] = str(sig)

                # 验证签名匹配
                if expected_signature:
                    result['signature_match'] = (str(sig) == expected_signature)

                return True, result

            except Exception as e:
                logger.warning(f"无法获取签名: {e}")
                return True, result

        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False, result

    def get_patch_statistics(self) -> Dict[str, Any]:
        """获取补丁统计信息"""
        successful = sum(1 for r in self.patch_history if r.success)
        failed = len(self.patch_history) - successful

        # 按类统计
        by_class: Dict[str, int] = {}
        for record in self.patch_history:
            key = f"{record.class_name}.{record.method_name}"
            by_class[key] = by_class.get(key, 0) + 1

        return {
            'total_patches': len(self.patch_history),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self.patch_history) if self.patch_history else 0.0,
            'pending_rollback': len(self.rollback_stack),
            'most_patched': list(by_class.items())[:5]
        }


# 全局单例
_global_patcher: Optional[FunctionLevelPatcher] = None


def get_function_patcher() -> FunctionLevelPatcher:
    """获取全局函数补丁器"""
    global _global_patcher
    if _global_patcher is None:
        _global_patcher = FunctionLevelPatcher()
    return _global_patcher
