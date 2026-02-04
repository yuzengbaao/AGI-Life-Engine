# 知识图谱锁文件问题诊断报告

**时间**: 2026-01-29 23:30
**问题**: `TimeoutError: Failed to acquire lock within 15.0s: data\knowledge\arch_graph.lock`

---

## 🔍 问题诊断

### 错误信息
```
[WorkingMemory] [COOLDOWN-PRECHECK] 概念冷却: C59c5cad2896a → C6f14daba8284 (尝试 1)
Traceback (most recent call last):
  File "D:\TRAE_PROJECT\AGI\core\knowledge_graph_exporter.py", line 343, in export_now
  File "D:\TRAE_PROJECT\AGI\core\knowledge_graph_exporter.py", line 159, in __enter__
    lock_content = f.read().strip()
TimeoutError: Failed to acquire lock within 15.0s: data\knowledge\arch_graph.lock
```

---

## 📊 根本原因分析

### 原因1：僵尸锁（Zombie Lock）

**锁文件内容：**
```
32532:ac6d7665-7444-4e76-9fef-1893838a9c0a:1769700116.713757
```

**解析：**
- **PID**: 32532
- **UUID**: ac6d7665-7444-4e76-9fef-1893838a9c0a
- **时间戳**: 1769700116.713757 (23:21:56)
- **当前时间**: 23:32:00
- **存在时长**: 10.1 分钟

**进程状态：**
- ❌ 进程 32532 **不存在**
- 结论：**僵尸锁** - 进程已终止但锁文件未清理

**原因推测：**
1. 进程崩溃
2. 进程被强制终止（如用户停止系统）
3. 锁未被正确释放

---

### 原因2：多进程竞争（Multi-Process Race）

**检测到的进程：**
```
AGI_Life_Engine.py 进程: 3个
- PID 44196 (运行中)
- PID 32532 (已不存在)
- PID 2264 (运行中)

live_monitor.py 进程: 4个
```

**问题：**
- 多个 AGI_Life_Engine.py 同时运行
- 可能是之前的修复脚本多次启动导致
- 多个进程竞争同一个锁文件

**影响：**
- 进程间相互干扰
- 锁竞争加剧
- 资源浪费

---

### 原因3：锁超时设计不足

**代码：**
```python
# core/knowledge_graph_exporter.py:159
with FileLock(str(lock_file), timeout=15.0, poll_interval=0.2):
```

**问题：**
- 15秒超时对于知识图谱导出可能不够
- 特别是多进程竞争时
- 没有重试机制

---

## 🛠️ 已采取的修复措施

### 1. 删除僵尸锁文件
```bash
rm -f "D:/TRAE_PROJECT/AGI/data/knowledge/arch_graph.lock"
```
✅ **已完成**

### 2. 清理多余进程
```
保留: PID 44196 (最新的 AGI_Life_Engine.py)
终止: PID 32532, 2264 (旧进程)
```
✅ **部分完成** - 仍有2个 AGI_Life_Engine.py 进程运行

---

## ⚠️ 当前状态

**锁文件：**
- ✅ 已清理

**进程状态：**
- ⚠️ AGI_Life_Engine.py: 2个进程（PID 44196, 37784）
- ⚠️ live_monitor.py: 4个进程

**问题：**
- 仍有多个进程运行
- 可能会再次产生锁竞争

---

## 🔧 建议的预防措施

### 1. 进程启动检测

**在 AGI_Life_Engine.py 启动时：**
```python
import psutil

def check_existing_instances():
    """检查是否已有实例运行"""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] != current_pid:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'AGI_Life_Engine.py' in cmdline:
                    print(f"[WARNING] 已有实例运行 (PID {proc.info['pid']})")
                    print("[WARNING] 正在终止当前实例")
                    sys.exit(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
```

### 2. 锁超时和重试机制

**改进 knowledge_graph_exporter.py：**
```python
# 添加重试机制
max_retries = 3
for attempt in range(max_retries):
    try:
        with FileLock(str(lock_file), timeout=30.0, poll_interval=0.5):
            # 执行导出
            break
    except TimeoutError:
        if attempt == max_retries - 1:
            # 最后一次重试失败，强制清理锁
            print(f"[WARNING] 锁超时，尝试清理...")
            if lock_file.exists():
                lock_file.unlink()
                time.sleep(1)
        else:
            print(f"[WARNING] 第 {attempt + 1} 次尝试超时，重试中...")
            time.sleep(2)
```

### 3. 进程监控脚本

**创建 process_monitor.py：**
```python
import subprocess
import time

def monitor_duplicate_processes():
    """监控并终止重复进程"""
    while True:
        result = subprocess.run(
            ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
            capture_output=True, text=True
        )

        # 统计各类进程
        agi_procs = []
        monitor_procs = []

        for line in result.stdout.split('\n'):
            if 'AGI_Life_Engine.py' in line and 'python.exe' in line:
                pid = line.strip().split()[-1]
                if pid.isdigit():
                    agi_procs.append(pid)
            elif 'live_monitor.py' in line and 'python.exe' in line:
                pid = line.strip().split()[-1]
                if pid.isdigit():
                    monitor_procs.append(pid)

        # 终止多余的进程
        if len(agi_procs) > 1:
            print(f"[Monitor] 发现 {len(agi_procs)} 个 AGI 进程，终止旧的...")
            for pid in agi_procs[:-1]:
                subprocess.run(['powershell', '-Command', f'Stop-Process -Id {pid} -Force'])

        if len(monitor_procs) > 1:
            print(f"[Monitor] 发现 {len(monitor_procs)} 个监控进程，终止旧的...")
            for pid in monitor_procs[:-1]:
                subprocess.run(['powershell', '-Command', f'Stop-Process -Id {pid} -Force'])

        time.sleep(60)  # 每分钟检查一次

if __name__ == '__main__':
    monitor_duplicate_processes()
```

---

## 📋 总结

### 问题根源
1. **僵尸锁** - 进程终止后锁文件未清理
2. **多进程** - 修复脚本启动了多个实例
3. **设计缺陷** - 锁超时不足，无重试机制

### 已修复
✅ 删除僵尸锁文件

### 待修复
⚠️ 清理多余进程（需要手动或脚本）
⚠️ 添加进程启动检测
⚠️ 改进锁超时和重试机制

### 预防措施
1. 确保只有一个 AGI_Life_Engine.py 进程运行
2. 添加进程启动检测
3. 增加锁超时时间或使用重试机制
4. 实现进程监控脚本

---

**报告生成时间**: 2026-01-29 23:35
**状态**: 僵尸锁已清理，多进程问题待解决
