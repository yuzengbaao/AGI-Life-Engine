import psutil
import os
import logging
from typing import Dict, Any

class ProprioceptiveMonitor:
    """
    本体感知监控器 (Resource Proprioception)
    负责实时监控系统资源（内存、CPU），并实施红线机制。
    """
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger("Monitor")
        self._setup_logging()
        # Cache process and system memory once per instance to avoid redundant calls
        self._process = psutil.Process(os.getpid())

    def _setup_logging(self) -> None:
        """设置日志格式，避免重复添加处理器"""
        if self.logger.hasHandlers():
            return
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况，优化为单次调用psutil"""
        try:
            # 获取进程内存信息
            mem_info = self._process.memory_info()
            # 获取系统内存和CPU一次性
            sys_mem = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            return {
                "process_memory_mb": mem_info.rss / (1024 * 1024),
                "system_memory_percent": sys_mem.percent,
                "cpu_percent": cpu_percent
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            self.logger.error(f"Failed to retrieve resource usage: {e}")
            # Return safe defaults in case of error
            return {
                "process_memory_mb": 0.0,
                "system_memory_percent": 0.0,
                "cpu_percent": 0.0
            }
        except Exception as e:
            self.logger.exception(f"Unexpected error during resource monitoring: {e}")
            return {
                "process_memory_mb": 0.0,
                "system_memory_percent": 0.0,
                "cpu_percent": 0.0
            }

    def check_health(self) -> str:
        """
        健康检查
        Returns: 'OK', 'WARNING', 'CRITICAL'
        """
        try:
            usage = self.get_resource_usage()
            mem_percent = usage['system_memory_percent']

            if not isinstance(mem_percent, (int, float)):
                raise ValueError(f"Invalid memory percent type: {type(mem_percent)}")

            if mem_percent > self.critical_threshold:
                self.logger.critical(f"Memory CRITICAL: {mem_percent:.1f}% > {self.critical_threshold}%")
                return "CRITICAL"
            elif mem_percent > self.warning_threshold:
                self.logger.warning(f"Memory WARNING: {mem_percent:.1f}% > {self.warning_threshold}%")
                return "WARNING"
            
            return "OK"
        except Exception as e:
            self.logger.exception(f"Error during health check: {e}")
            return "CRITICAL"  # Fail-safe to indicate potential danger

    def suggest_action(self, health_status: str) -> str:
        """根据健康状态建议行动，输入验证增强"""
        if not isinstance(health_status, str):
            self.logger.error(f"Invalid health_status type: {type(health_status)}, defaulting to CONTINUE")
            return "CONTINUE"

        action_map = {
            "CRITICAL": "EMERGENCY_SAVE_AND_EXIT",
            "WARNING": "GC_AND_THROTTLE",
            "OK": "CONTINUE"
        }

        action = action_map.get(health_status)
        if action is None:
            self.logger.warning(f"Unknown health status '{health_status}', defaulting to CONTINUE")
            return "CONTINUE"

        return action