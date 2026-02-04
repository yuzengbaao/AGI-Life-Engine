# -*- coding: utf-8 -*-
"""Stage 2 minimal integration tests for P1 memory (module-level integration).

These tests validate that the P1 three-layer memory module can be driven by
the config structure and behaves correctly across a simple usage flow. We avoid
importing the full AGI orchestrator to keep dependencies minimal.
"""
import json
from pathlib import Path

from three_layer_memory import ThreeLayerMemory


def test_config_enables_initialization():
    cfg = json.loads(Path("agi_integrated_config.json").read_text(encoding="utf-8"))
    mem_cfg = cfg.setdefault("p1_modules", {}).setdefault("memory", {})
    mem_cfg["enabled"] = True
    mem = ThreeLayerMemory()
    assert mem.initialize(mem_cfg.get("config", {})) is True


def test_end_to_end_usage_flow():
    mem = ThreeLayerMemory()
    assert mem.initialize({"enabled": True, "warmup_ms": 0, "max_episodes": 5})
    for i in range(7):
        mem.save_episode({"i": i})
    out = mem.recall({})
    assert out["result"] == {"i": 6}
    assert out["count"] == 5
    health = mem.health_status()
    assert health["initialized"] is True and health["episodes"] == 5
