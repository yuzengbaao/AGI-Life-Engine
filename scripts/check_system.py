#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前运行的系统
"""

import sys
import psutil
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_running_processes():
    """检查运行中的AGI进程"""
    print("[检查] 搜索运行中的AGI系统进程...")

    found = False

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline:
                continue

            cmdline_str = ' '.join(cmdline)

            # 检查AGI相关进程
            if any(keyword in cmdline_str for keyword in ['AGI_Life_Engine.py', 'AGI', 'agi']):
                print(f"[发现] PID {proc.info['pid']}: {cmdline_str}")
                found = True

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("[信息] 未发现运行中的AGI系统进程")

    return found

def check_b_system_status():
    """检查B系统状态"""
    print("\n[检查] B系统状态...")

    startup_file = project_root / "monitoring" / "system_startup.json"
    if startup_file.exists():
        import json
        with open(startup_file, 'r', encoding='utf-8') as f:
            startup_info = json.load(f)
        print(f"[发现] B系统启动信息:")
        print(f"  启动时间: {startup_info.get('startup_time')}")
        print(f"  模式: {startup_info.get('mode')}")
        print(f"  置信度阈值: {startup_info.get('confidence_threshold')}")
        print(f"  设备: {startup_info.get('device')}")
        print(f"  监控: {startup_info.get('monitoring')}")
        return True
    else:
        print("[信息] 未找到B系统启动信息")
        return False

def check_monitoring():
    """检查监控系统"""
    print("\n[检查] 监控系统...")

    # 检查监控文件
    monitoring_dir = project_root / "monitoring"
    if monitoring_dir.exists():
        files = list(monitoring_dir.glob("*.json"))
        if files:
            print(f"[发现] 监控数据文件: {len(files)} 个")
            for f in files[-5:]:  # 显示最近5个
                print(f"  - {f.name}")
            return True
        else:
            print("[信息] 监控目录为空")
    else:
        print("[信息] 监控目录不存在")

    return False

if __name__ == "__main__":
    print("="*60)
    print("[检查] AGI系统状态检查")
    print("="*60)

    # 检查进程
    has_process = check_running_processes()

    # 检查B系统
    has_b_system = check_b_system_status()

    # 检查监控
    has_monitoring = check_monitoring()

    print("\n" + "="*60)
    print("[总结]")
    if has_process:
        print("  AGI系统进程: 运行中")
    else:
        print("  AGI系统进程: 未运行")

    if has_b_system:
        print("  B系统状态: 已启动")
    else:
        print("  B系统状态: 未启动")

    if has_monitoring:
        print("  监控系统: 已安装")
    else:
        print("  监控系统: 未安装")

    print("="*60)
