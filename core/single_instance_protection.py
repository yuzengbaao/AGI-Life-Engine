#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单实例保护 - 防止多进程运行

在 AGI_Life_Engine.py 启动时调用此模块
"""

import os
import sys
import subprocess
from pathlib import Path


def check_existing_instances(script_name="AGI_Life_Engine.py"):
    """
    检查是否已有实例运行

    Args:
        script_name: 要检查的脚本名称

    Returns:
        bool: True 表示已有实例运行，应该退出
    """
    current_pid = os.getpid()

    try:
        # 使用 wmic 命令查找进程
        result = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
            capture_output=True,
            text=True,
            timeout=10
        )

        existing_pids = []

        for line in result.stdout.split('\n'):
            if script_name in line and 'python.exe' in line:
                # 提取PID
                parts = line.strip().split()
                if parts:
                    pid_str = parts[-1]
                    if pid_str.isdigit() and int(pid_str) != current_pid:
                        existing_pids.append(int(pid_str))

        if existing_pids:
            print(f"\n{'='*60}")
            print(f"[WARNING] 检测到已有 {script_name} 实例在运行")
            print(f"[WARNING] 现有进程 PID: {existing_pids}")
            print(f"[WARNING] 当前进程 PID: {current_pid}")
            print(f"[WARNING]")
            print(f"[WARNING] 为了避免进程冲突和锁竞争，")
            print(f"[WARNING] 系统将终止当前实例的启动。")
            print(f"{'='*60}\n")

            print("[提示] 如需重启系统，请先停止现有进程：")
            print(f"  powershell -Command \"Stop-Process -Id {existing_pids[0]} -Force\"")
            print()

            return True  # 应该退出

        return False  # 可以继续运行

    except Exception as e:
        print(f"[ERROR] 进程检测失败: {e}")
        print("[INFO] 继续启动...")
        return False


def ensure_single_instance(script_name="AGI_Life_Engine.py"):
    """
    确保单实例运行（如果发现已有实例，则退出）

    用法：在 AGI_Life_Engine.py 的 main 入口添加：
    ```python
    from single_instance_protection import ensure_single_instance
    if ensure_single_instance():
        sys.exit(1)
    ```
    """
    if check_existing_instances(script_name):
        sys.exit(1)


if __name__ == '__main__':
    # 测试模式
    print("测试单实例保护...")

    if ensure_single_instance():
        print("检测到已有实例，退出")
    else:
        print("没有发现其他实例，可以启动")
