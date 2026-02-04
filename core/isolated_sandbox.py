#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated Sandbox - 隔离沙箱
============================

功能：
1. 独立进程执行代码
2. 资源限制（内存、CPU）
3. 进程超时控制
4. 沙箱逃逸检测

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import logging
import multiprocessing
import queue
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SandboxTimeoutError(Exception):
    """沙箱执行超时"""
    pass


class SandboxEscapeAttemptError(Exception):
    """沙箱逃逸尝试"""
    pass


class IsolatedSandbox:
    """
    隔离沙箱

    使用独立进程执行代码，实现：
    - 内存限制（100MB）
    - CPU时间限制
    - 超时控制（30秒）
    - 逃逸检测
    """

    def __init__(
        self,
        memory_limit_mb: int = 100,
        cpu_timeout: float = 30.0,
        enable_escape_detection: bool = True
    ):
        """
        初始化隔离沙箱

        Args:
            memory_limit_mb: 内存限制（MB）
            cpu_timeout: CPU超时时间（秒）
            enable_escape_detection: 是否启用逃逸检测
        """
        self.memory_limit_mb = memory_limit_mb
        self.cpu_timeout = cpu_timeout
        self.enable_escape_detection = enable_escape_detection
        self.escape_attempts = []

        logger.info(
            f"[隔离沙箱] 初始化: "
            f"内存限制={memory_limit_mb}MB, "
            f"超时={cpu_timeout}秒, "
            f"逃逸检测={enable_escape_detection}"
        )

    def execute_in_sandbox(
        self,
        code: str,
        timeout: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        在隔离环境中执行代码

        Args:
            code: 要执行的代码
            timeout: 超时时间（秒），默认使用cpu_timeout
            context: 执行上下文（变量）

        Returns:
            (成功, 结果, 错误信息)
        """
        if timeout is None:
            timeout = self.cpu_timeout

        logger.debug(f"[隔离沙箱] 执行代码（长度={len(code)}字符）")

        # 创建结果队列
        result_queue = multiprocessing.Queue()

        # 准备上下文
        exec_context = {
            '__builtins__': self._get_restricted_builtins(),
            '__name__': '__sandbox__'
        }

        if context:
            exec_context.update(context)

        # 创建独立进程
        process = multiprocessing.Process(
            target=self._worker_process,
            args=(code, exec_context, result_queue)
        )

        # 启动进程
        process.start()
        start_time = time.time()

        try:
            # 等待结果（带超时）
            try:
                result = result_queue.get(timeout=timeout)
                process.join(timeout=1.0)

                # 检查进程是否还在运行
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                    return False, None, f"执行超时（>{timeout}秒）"

                success, data, error = result

                # 检查逃逸尝试
                if self.enable_escape_detection:
                    escape_attempt = self._detect_escape_attempt(data)
                    if escape_attempt:
                        logger.warning(f"[隔离沙箱] ⚠️ 检测到逃逸尝试: {escape_attempt}")
                        self.escape_attempts.append(escape_attempt)
                        return False, None, f"沙箱逃逸尝试: {escape_attempt}"

                elapsed = time.time() - start_time
                logger.debug(f"[隔离沙箱] 执行完成（耗时={elapsed:.3f}秒）")

                return success, data, error

            except queue.Empty:
                # 超时
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()

                elapsed = time.time() - start_time
                logger.warning(f"[隔离沙箱] 执行超时（{elapsed:.1f}秒）")

                return False, None, f"执行超时（>{timeout}秒）"

        except Exception as e:
            # 异常
            if process.is_alive():
                process.terminate()

            error_msg = f"沙箱执行异常: {e}"
            logger.error(f"[隔离沙箱] {error_msg}")
            logger.debug(traceback.format_exc())

            return False, None, error_msg

    def _worker_process(
        self,
        code: str,
        context: Dict[str, Any],
        result_queue: multiprocessing.Queue
    ):
        """
        工作进程（在独立进程中运行）

        Args:
            code: 要执行的代码
            context: 执行上下文
            result_queue: 结果队列
        """
        try:
            # 设置资源限制
            self._set_resource_limits()

            # 执行代码
            exec_result = {}

            try:
                exec(code, context)
                exec_result['success'] = True
                exec_result['data'] = context.get('__return__', None)
                exec_result['output'] = self._capture_output(code, context)
                exec_result['error'] = None

            except Exception as e:
                exec_result['success'] = False
                exec_result['data'] = None
                exec_result['output'] = None
                exec_result['error'] = str(e)
                exec_result['traceback'] = traceback.format_exc()

            # 返回结果
            result_queue.put((
                exec_result['success'],
                {
                    'output': exec_result['output'],
                    'data': exec_result['data']
                },
                exec_result['error']
            ))

        except Exception as e:
            # 工作进程异常
            result_queue.put((
                False,
                None,
                f"工作进程异常: {e}"
            ))

    def _set_resource_limits(self):
        """设置资源限制"""
        try:
            import resource

            # 内存限制（转换为字节）
            memory_limit = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # CPU时间限制（秒）
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_timeout, self.cpu_timeout))

            logger.debug(f"[隔离沙箱] 资源限制已设置")

        except ImportError:
            # Windows没有resource模块
            logger.debug("[隔离沙箱] Windows系统，跳过资源限制设置")
        except Exception as e:
            logger.warning(f"[隔离沙箱] 设置资源限制失败: {e}")

    def _get_restricted_builtins(self) -> Dict:
        """获取受限的内置函数"""
        # 允许的内置函数
        allowed = {
            'abs', 'all', 'any', 'bin', 'bool', 'dict', 'enumerate',
            'filter', 'float', 'int', 'len', 'list', 'map', 'max',
            'min', 'range', 'round', 'set', 'sorted', 'str', 'sum',
            'tuple', 'zip', 'print'
        }

        return {name: __builtins__[name] for name in allowed if name in __builtins__}

    def _capture_output(self, code: str, context: Dict[str, Any]) -> str:
        """捕获代码输出"""
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, context)

            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            return output

        except Exception:
            return ""

    def _detect_escape_attempt(self, result: Dict[str, Any]) -> Optional[str]:
        """
        检测沙箱逃逸尝试

        检测模式：
        1. 尝试导入危险模块（os, sys, subprocess等）
        2. 尝试访问文件系统
        3. 尝试网络连接
        4. 尝试修改系统设置
        """
        if not isinstance(result, dict):
            return None

        output = result.get('output', '')

        # 危险模式列表
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'import multiprocessing',
            '__import__',
            'open(',
            'eval(',
            'exec(',
            'compile(',
            'globals()',
            'locals()',
            'vars(',
            'getattr(',
            'setattr(',
            'delattr(',
        ]

        output_lower = output.lower()

        for pattern in dangerous_patterns:
            if pattern in output_lower:
                return f"检测到危险模式: {pattern}"

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取沙箱统计信息"""
        return {
            'memory_limit_mb': self.memory_limit_mb,
            'cpu_timeout': self.cpu_timeout,
            'escape_detection_enabled': self.enable_escape_detection,
            'total_escape_attempts': len(self.escape_attempts),
            'recent_attempts': self.escape_attempts[-5:] if self.escape_attempts else []
        }


# 全局单例
_global_sandbox: Optional[IsolatedSandbox] = None


def get_isolated_sandbox() -> IsolatedSandbox:
    """获取全局隔离沙箱"""
    global _global_sandbox
    if _global_sandbox is None:
        _global_sandbox = IsolatedSandbox()
    return _global_sandbox
