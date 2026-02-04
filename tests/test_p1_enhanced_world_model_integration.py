"""
Integration tests for P1 Enhanced World Model.

Tests config-driven behavior and end-to-end usage flow.
"""

import json
import pytest
from enhanced_world_model import EnhancedWorldModel


def test_config_enables_initialization():
    """Test that config file enables world model initialization."""
    with open("agi_integrated_config.json") as f:
        config = json.load(f)

    p1_wm_cfg = config.get("p1_modules", {}).get("world_model", {})
    enabled = p1_wm_cfg.get("enabled", False)

    # Verify default state (disabled)
    assert enabled is False

    # Test initialization with config
    model = EnhancedWorldModel()
    ok = model.initialize(p1_wm_cfg.get("config", {}))
    assert ok is True


def test_end_to_end_usage_flow():
    """Test complete workflow: init → update → predict → query → health."""
    model = EnhancedWorldModel()

    # Step 1: Initialize
    config = {"max_history": 5, "warmup_ms": 5}
    ok = model.initialize(config)
    assert ok is True

    # Step 2: Update states (7 times, max=5)
    for i in range(7):
        model.update_state({
            "entities": {"step": i, "time": i * 0.1},
            "relations": [("agent", "at", f"pos{i}")]
        })

    # Step 3: Predict future state
    action = {"add_entity": {"new_item": "acquired"}}
    prediction = model.predict(action, steps=2)
    assert "predicted_state" in prediction
    assert prediction["confidence"] > 0

    # Step 4: Query counterfactual
    condition = {"premise": "item_was_dropped", "hypothetical_entities": {"dropped": True}}
    result = model.query_counterfactual(condition)
    assert "result" in result
    assert "reasoning" in result

    # Step 5: Health check (should keep only last 5 states)
    health = model.health_status()
    assert health["initialized"] is True
    assert health["state_count"] == 5  # Max capacity enforced
    assert health["has_current_state"] is True
    assert health["uptime_sec"] >= 0

    # Step 6: Shutdown
    model.shutdown()
    assert model.health_status()["initialized"] is False
