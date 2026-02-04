import os
import re
from typing import Dict, Any, List


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _detect_imports(source: str) -> List[str]:
    pattern = r"from\s+([\w\.]+)\s+import"
    return re.findall(pattern, source)


def scan_current_layout(base_dir: str = ".") -> Dict[str, Any]:
    engine_path = os.path.join(base_dir, "AGI_Life_Engine.py")
    summary: Dict[str, Any] = {
        "type": "unknown",
        "central_coordinator": False,
        "event_bus": False,
        "feedback_loop": False,
        "imports": [],
    }
    if not os.path.exists(engine_path):
        return summary
    try:
        src = _read_file(engine_path)
        summary["imports"] = _detect_imports(src)
        summary["event_bus"] = "EventBus" in src
        summary["central_coordinator"] = "ComponentCoordinator" in src
        has_monitor = "RuntimeMonitor" in src or "SystemMonitor" in src
        has_events = "event_bus" in src
        summary["feedback_loop"] = has_monitor and has_events
        if "PlannerAgent" in src and "ExecutorAgent" in src and "CriticAgent" in src:
            summary["type"] = "layered_pipeline"
        else:
            summary["type"] = "mesh_candidate"
    except Exception:
        return summary
    return summary

