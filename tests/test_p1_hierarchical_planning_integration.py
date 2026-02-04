"""
Integration tests for P1 Hierarchical Planning.

Tests config-driven behavior and end-to-end usage flow with other P1 modules.
"""

import json
import pytest
from hierarchical_planning import HierarchicalPlanning, PlanningMode


def test_config_enables_initialization():
    """Test that config file enables planning initialization."""
    with open("agi_integrated_config.json") as f:
        config = json.load(f)

    p1_plan_cfg = config.get("p1_modules", {}).get("planning", {})
    enabled = p1_plan_cfg.get("enabled", False)

    # Verify default state (disabled)
    assert enabled is False

    # Test initialization with config
    planner = HierarchicalPlanning()
    ok = planner.initialize(p1_plan_cfg.get("config", {}))
    assert ok is True


def test_end_to_end_usage_flow():
    """Test complete workflow: init → generate → execute → refine → health."""
    planner = HierarchicalPlanning()

    # Step 1: Initialize with custom config
    config = {"max_plans": 3, "default_mode": "tree_of_thoughts", "warmup_ms": 5}
    ok = planner.initialize(config)
    assert ok is True

    # Step 2: Generate plans in different modes
    plan1 = planner.generate_plan("Build AGI system", PlanningMode.COT)
    assert plan1 is not None
    assert plan1.mode == PlanningMode.COT
    assert len(plan1.steps) == 4  # CoT has 4 steps

    plan2 = planner.generate_plan("Optimize performance", PlanningMode.TOT)
    assert plan2 is not None
    assert plan2.mode == PlanningMode.TOT

    plan3 = planner.generate_plan("Debug system", PlanningMode.SOT)
    assert plan3 is not None
    assert plan3.mode == PlanningMode.SOT
    assert len(plan3.steps) == 3  # SoT has 3 steps

    # Step 3: Capacity enforcement (generate 4th plan, should evict first)
    plan4 = planner.generate_plan("New task", PlanningMode.COT)
    assert plan4 is not None
    health = planner.health_status()
    assert health["plan_count"] == 3  # Max capacity enforced

    # Step 4: Execute steps
    step_id = plan2.steps[0].step_id
    result = planner.execute_step(plan2.plan_id, step_id)
    assert result["status"] in ["completed", "failed"]

    # Step 5: Refine plan with feedback
    refined_plan = planner.refine_plan(plan2.plan_id, "Need more verification")
    assert refined_plan is not None
    assert len(refined_plan.steps) > len(plan2.steps)  # Added verification step
    assert refined_plan.metadata["refinement_count"] == 1

    # Step 6: Health check
    health = planner.health_status()
    assert health["initialized"] is True
    assert health["plan_count"] <= 3
    assert health["default_mode"] == "tree_of_thoughts"
    assert health["uptime_sec"] >= 0

    # Step 7: Shutdown
    planner.shutdown()
    assert planner.health_status()["initialized"] is False


def test_integration_with_memory_and_world_model():
    """Test planning system works alongside memory and world model."""
    try:
        from three_layer_memory import ThreeLayerMemory
        from enhanced_world_model import EnhancedWorldModel
    except ImportError:
        pytest.skip("P1 memory or world_model not available")

    # Initialize all 3 P1 modules
    memory = ThreeLayerMemory()
    world_model = EnhancedWorldModel()
    planner = HierarchicalPlanning()

    assert memory.initialize({"max_episodes": 10}) is True
    assert world_model.initialize({"max_history": 10}) is True
    assert planner.initialize({"max_plans": 5}) is True

    # Workflow: Store experience → Model state → Plan action
    # Step 1: Store experience in memory
    memory.save_episode({"task_id": "task_01", "action": "explore", "result": "success"})

    # Step 2: Update world model with new state
    world_model.update_state({
        "entities": {"task": "task_01", "status": "completed"},
        "relations": [("agent", "completed", "task_01")]
    })

    # Step 3: Generate plan for next action
    plan = planner.generate_plan("Continue exploration", PlanningMode.COT, context={
        "memory": "Previous task succeeded",
        "world_state": "Environment stable"
    })
    assert plan is not None
    assert "memory" in plan.metadata

    # Verify all modules healthy
    assert memory.health_status()["initialized"] is True
    assert world_model.health_status()["initialized"] is True
    assert planner.health_status()["initialized"] is True

    # Cleanup
    memory.shutdown()
    world_model.shutdown()
    planner.shutdown()
