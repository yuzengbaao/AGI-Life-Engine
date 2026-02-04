import inspect
import os
import logging
from typing import Dict, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

class RuntimeMonitor:
    """
    Phase 2.3: 动态运行时监视器 (The All-Seeing Eye)
    
    核心功能:
    1. 注册对象 (Register): 在对象创建时记录其源码位置 (文件, 行号)。
    2. 检查对象 (Inspect): 给定一个对象实例, 返回其定义源头。
    """
    
    _registry: Dict[int, Dict[str, Any]] = {}  # { id(obj): metadata }
    _enabled = True

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def register(cls, obj: Any, context_info: str = None):
        """
        在对象创建时调用此方法，记录其“出生地”。
        通常在 __init__ 中调用: RuntimeMonitor.register(self)
        """
        if not cls._enabled:
            return

        try:
            obj_id = id(obj)
            # 获取调用栈的上一帧 (即调用 register 的地方, 通常是 __init__)
            # stack[0] is current, stack[1] is caller
            stack = inspect.stack()
            if len(stack) < 2:
                return
            
            frame = stack[1]
            filename = frame.filename
            lineno = frame.lineno
            function = frame.function
            
            # 记录元数据
            cls._registry[obj_id] = {
                'type': type(obj).__name__,
                'file_path': os.path.abspath(filename),
                'line_number': lineno,
                'created_in_function': function,
                'context': context_info
            }
            # logger.debug(f"Registered object {type(obj).__name__} from {filename}:{lineno}")
            
        except Exception as e:
            logger.warning(f"Failed to register object: {e}")

    @classmethod
    def inspect_object(cls, obj: Any) -> Optional[Dict[str, Any]]:
        """
        查询对象的来源信息。
        如果对象已注册，返回注册信息；否则尝试通过类名查找 (Static Fallback)。
        """
        obj_id = id(obj)
        if obj_id in cls._registry:
            return cls._registry[obj_id]
        
        return None

    @classmethod
    def get_registry_size(cls):
        return len(cls._registry)

    @classmethod
    def clear_registry(cls):
        cls._registry.clear()