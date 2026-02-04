#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
停止A系统运行进程
找出并终止所有AGI相关的Python进程
"""

import sys
import psutil
import time
from pathlib import Path

def find_agi_processes():
    """查找所有AGI相关进程"""
    print("[查找] 搜索AGI相关进程...")

    agi_processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline:
                continue

            # 将命令行转换为字符串进行匹配
            cmdline_str = ' '.join(cmdline)

            # 检查是否包含AGI相关的关键字
            agi_keywords = [
                'AGI_Life_Engine.py',
                'AGI',
                'agi',
                'AGI Life Engine'
            ]

            if any(keyword in cmdline_str for keyword in agi_keywords):
                agi_processes.append(proc)
                print(f"[发现] PID {proc.info['pid']}: {cmdline_str[:100]}")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return agi_processes

def stop_processes(processes):
    """停止指定的进程列表"""
    if not processes:
        print("[信息] 未发现运行中的AGI进程")
        return True

    print(f"\n[停止] 找到 {len(processes)} 个AGI进程")
    print("[确认] 准备停止这些进程...")

    for proc in processes:
        try:
            pid = proc.info['pid']
            print(f"\n[停止] 终止进程 PID {pid}...")

            # 先尝试优雅终止
            proc.terminate()

            # 等待进程结束
            try:
                proc.wait(timeout=5)
                print(f"[成功] 进程 {pid} 已优雅终止")
            except psutil.TimeoutExpired:
                # 强制终止
                print(f"[强制] 进程 {pid} 未响应，强制终止...")
                proc.kill()
                proc.wait(timeout=2)
                print(f"[成功] 进程 {pid} 已强制终止")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[警告] 无法终止进程 {proc.info['pid']}: {e}")

    # 等待一下确保所有进程都结束了
    time.sleep(2)

    return True

def verify_stopped():
    """验证进程是否已停止"""
    print("\n[验证] 检查是否还有AGI进程运行...")

    remaining = find_agi_processes()

    if remaining:
        print(f"[警告] 仍有 {len(remaining)} 个AGI进程运行")
        return False
    else:
        print("[成功] 所有AGI进程已停止")
        return True

def main():
    """主函数"""
    print("="*60)
    print("[停止] A系统进程终止工具")
    print("="*60)

    # 查找进程
    processes = find_agi_processes()

    if not processes:
        print("\n[信息] 未发现运行中的AGI进程")
        print("[信息] A系统可能已经停止")
        return

    # 停止进程
    stop_processes(processes)

    # 验证
    if verify_stopped():
        print("\n[完成] A系统已完全停止")
        print("[提示] 现在可以运行B系统了")
    else:
        print("\n[警告] 部分进程可能仍在运行")
        print("[建议] 请手动检查任务管理器")

    print("="*60)

if __name__ == "__main__":
    main()
