import json
import os
import time
from typing import Dict, Any, Optional

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "system_state.json")

class AGITelemetry:
    """
    Real-time system state bridge.
    Allows the 'Body' (Worker Scripts) to report physical sensations to the 'Soul' (Life Engine).
    """
    
    @staticmethod
    def update_state(phase: str, details: Dict[str, Any]) -> bool:
        """
        Called by worker scripts (e.g. batch_drafter) to report status.
        
        Args:
            phase: Current execution phase (e.g., "CAD_PROCESSING", "IDLE", "ERROR")
            details: Dictionary with additional context (e.g., project, metrics)
            
        Returns:
            bool: True if write succeeded, False otherwise
        """
        state: Dict[str, Any] = {
            "timestamp": time.time(),
            "phase": phase,
            "details": details or {},
            "last_active": time.time()
        }
        
        temp_file: str = STATE_FILE + ".tmp"
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            
            # Atomic write using context manager
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, STATE_FILE)
            return True
        except (OSError, IOError, json.JSONDecodeError) as e:
            print(f"Telemetry Write Error: {e}")
            return False

    @staticmethod
    def get_state() -> Dict[str, Any]:
        """
        Called by the Life Engine to 'feel' what the body is doing.
        
        Returns:
            Dict containing current system state, or default error/offline state
        """
        if not os.path.exists(STATE_FILE):
            return {"phase": "OFFLINE", "details": {}, "timestamp": 0}
            
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validate required keys
                if isinstance(data, dict):
                    return data
                else:
                    return {"phase": "READ_ERROR", "details": {}, "timestamp": 0}
        except (OSError, IOError, json.JSONDecodeError) as e:
            print(f"Telemetry Read Error: {e}")
            return {"phase": "READ_ERROR", "details": {}, "timestamp": 0}