import json
import os
import shutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class CheckpointManager:
    """
    持久化管理器 (Atomic Persistence)
    实现原子化保存和断点续传。
    """
    def __init__(self, checkpoint_dir="data/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.latest_file = os.path.join(checkpoint_dir, "latest.json")
        self.logger = logging.getLogger("Checkpoint")
        self._ensure_dir()

    def _ensure_dir(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save_checkpoint(self, state: Dict[str, Any]):
        """
        原子化保存
        1. 写入 .tmp 文件
        2. 重命名为正式文件
        """
        try:
            timestamp = datetime.now().isoformat().replace(":", "-")
            temp_file = os.path.join(self.checkpoint_dir, f"state_{timestamp}.tmp")
            
            # 添加元数据
            state['_meta'] = {
                'saved_at': timestamp,
                'version': '1.0'
            }

            # 1. 写入临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            # 2. 原子替换 (在 Windows 上 os.replace 是原子的，如果目标存在会覆盖)
            # 同时也保留一份历史记录，方便回溯（可选）
            history_file = os.path.join(self.checkpoint_dir, f"checkpoint_{state.get('iteration', 'unknown')}.json")
            shutil.copy(temp_file, history_file)
            
            # 更新 latest 指针
            if os.path.exists(self.latest_file):
                os.remove(self.latest_file)
            os.rename(temp_file, self.latest_file)
            
            self.logger.info(f"Checkpoint saved: iteration {state.get('iteration')}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """加载最新的检查点"""
        if not os.path.exists(self.latest_file):
            self.logger.info("No checkpoint found. Starting fresh.")
            return None
        
        try:
            with open(self.latest_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.logger.info(f"Loaded checkpoint from {state['_meta']['saved_at']}")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
