import importlib.util
import os
import sys


def _load_legacy_experience_memory():
    legacy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory.py"))
    if not os.path.exists(legacy_path):
        raise ImportError("Legacy memory implementation not found")

    module_name = "core._legacy_memory"
    for _ in range(2):
        legacy_mod = sys.modules.get(module_name)
        if legacy_mod is not None and hasattr(legacy_mod, "ExperienceMemory"):
            return legacy_mod.ExperienceMemory

        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, legacy_path)
        if spec is None or spec.loader is None:
            raise ImportError("Failed to load legacy memory implementation")

        legacy_mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = legacy_mod
        try:
            spec.loader.exec_module(legacy_mod)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    raise ImportError("ExperienceMemory not found in legacy memory implementation")


ExperienceMemory = _load_legacy_experience_memory()

__all__ = ["ExperienceMemory"]
