"""
Core package initialization.
"""

import importlib
import importlib.util
import os
import sys


def _install_experience_memory() -> None:
    module_name = "core._legacy_memory"
    installed_module = False
    try:
        mem_pkg = importlib.import_module("core.memory")
        if getattr(mem_pkg, "ExperienceMemory", None) is not None:
            return

        legacy_path = os.path.join(os.path.dirname(__file__), "memory.py")
        if not os.path.exists(legacy_path):
            return

        spec = importlib.util.spec_from_file_location(module_name, legacy_path)
        if spec is None or spec.loader is None:
            return

        legacy_mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = legacy_mod
        installed_module = True
        spec.loader.exec_module(legacy_mod)

        if hasattr(legacy_mod, "ExperienceMemory"):
            mem_pkg.ExperienceMemory = legacy_mod.ExperienceMemory
    except Exception:
        if installed_module:
            sys.modules.pop(module_name, None)
        return


_install_experience_memory()
