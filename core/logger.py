import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Generator, Optional

class ThoughtLogger:
    def __init__(self, log_dir: str = "data/logs") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = self._get_log_file_path()
        self.console_logger = self._setup_console_logger()

    def _get_log_file_path(self) -> str:
        """Generate the current log file path based on today's date."""
        date_str = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.log_dir, f"thought_stream_{date_str}.jsonl")

    def _setup_console_logger(self) -> logging.Logger:
        """Set up and return the console logger with proper formatting."""
        logger = logging.getLogger("AGI_Mind")
        logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if already set
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def log_thought(self, phase: str, content: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured thought unit.
        
        Args:
            phase: Current phase (e.g., "THINKING", "CODING", "REFLECTION")
            content: The core content of the thought
            meta: Metadata like iteration number, resource usage, etc.
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "content": content,
            "meta": meta or {}
        }
        
        # Ensure we're writing to today's file
        current_file = self._get_log_file_path()
        if current_file != self.current_log_file:
            self.current_log_file = current_file

        # Write to JSONL file with error handling
        try:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, IOError) as e:
            self.console_logger.error(f"Failed to write log entry to {self.current_log_file}: {e}")

        # Output brief summary to console with truncation safety
        try:
            content_str = json.dumps(content, ensure_ascii=False)
            truncated_content = content_str[:100] + "..." if len(content_str) > 100 else content_str
            self.console_logger.info(f"[{phase}] {truncated_content}")
        except (TypeError, ValueError) as e:
            self.console_logger.error(f"Failed to serialize content for console output: {e}")

    def replay_thoughts(self, date_str: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Generator to replay thoughts from a specific date.
        
        Args:
            date_str: Date string in YYYYMMDD format. If None, uses today's date.
            
        Yields:
            Parsed JSON entries from the log file.
        """
        target_date = date_str or datetime.now().strftime('%Y%m%d')
        target_file = os.path.join(self.log_dir, f"thought_stream_{target_date}.jsonl")
        
        if not os.path.exists(target_file):
            self.console_logger.warning(f"Log file not found for date {target_date}: {target_file}")
            return

        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        self.console_logger.error(f"Invalid JSON in log file {target_file} at line {line_num}: {e}")
        except (OSError, IOError) as e:
            self.console_logger.error(f"Failed to read log file {target_file}: {e}")