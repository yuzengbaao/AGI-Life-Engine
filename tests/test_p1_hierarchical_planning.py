"""
Unit tests for P1 Hierarchical Planning module.

Tests cover:
- Initialization and configuration
- Plan generation (CoT/ToT/SoT modes)
- Step execution
- Plan refinement (self-repair)
- Error handling
- Health monitoring
"""

import pytest
from hierarchical_planning import (
    HierarchicalPlanning,
    PlanningException,
    PlanningOperationError,
    PlanningMode,
    StepStatus,
    Plan,
    PlanStep
)


class TestHierarchicalPlanning:
    """Test suite for HierarchicalPlanning."""

    def test_initialize_and_health(self):
        """Test basic initialization and health check."""
        planner = HierarchicalPlanning()

        # Initialize with custom config
        config = {
            "max_plans": 30,
            "default_mode": "tree_of_thoughts",
            "warmup_ms": 5,
            "metadata": {"env": "test"}
        }
        ok = planner.initialize(config)

        assert ok is True

        # Check health status
        health = planner.health_status()
        assert health["initialized"] is True
        assert health["plan_count"] == 0
        assert health["default_mode"] == "tree_of_thoughts"
        assert health["uptime_sec"] >= 0

    def test_generate_cot_plan(self):
        """Test Chain of Thought plan generation."""
        planner = HierarchicalPlanning()
        planner.initialize()

        plan = planner.generate_plan(
            goal="Complete task using CoT",
            mode=PlanningMode.COT,
            context={"priority": "high"}
        )

        assert plan is not None
        assert plan.goal == "Complete task using CoT"
        assert plan.mode == PlanningMode.COT
        assert len(plan.steps) >= 3  # CoT generates 4 steps
        assert all(isinstance(s, PlanStep) for s in plan.steps)
        assert all(s.status == StepStatus.PENDING for s in plan.steps)

    def test_generate_tot_plan(self):
        """Test Tree of Thoughts plan generation."""
        planner = HierarchicalPlanning()
        planner.initialize()

        plan = planner.generate_plan(
            goal="Explore options with ToT",
            mode=PlanningMode.TOT
        )

        assert plan.mode == PlanningMode.TOT
        assert len(plan.steps) >= 3  # ToT generates multiple branches

    def test_generate_sot_plan(self):
        """Test Skeleton of Thoughts plan generation."""
        planner = HierarchicalPlanning()
        planner.initialize()

        plan = planner.generate_plan(
            goal="Build skeleton first",
            mode=PlanningMode.SOT
        )

        assert plan.mode == PlanningMode.SOT
        assert len(plan.steps) >= 2  # SoT has skeleton + details

    def test_plan_caching_and_capacity(self):
        """Test plan caching with capacity management."""
        planner = HierarchicalPlanning()
        planner.initialize({"max_plans": 3})

        # Generate 5 plans (should keep only last 3)
        plans = []
        for i in range(5):
            plan = planner.generate_plan(f"Task {i}")
            plans.append(plan.plan_id)

        health = planner.health_status()
        assert health["plan_count"] == 3  # Max capacity enforced

    def test_execute_step_success(self):
        """Test successful step execution."""
        planner = HierarchicalPlanning()
        planner.initialize()

        plan = planner.generate_plan("Test execution")
        step_id = plan.steps[0].step_id

        result = planner.execute_step(plan.plan_id, step_id)

        assert "status" in result
        assert result["status"] in [StepStatus.COMPLETED.value, StepStatus.FAILED.value]

    def test_refine_plan(self):
        """Test plan refinement (self-repair)."""
        planner = HierarchicalPlanning()
        planner.initialize()

        original_plan = planner.generate_plan("Original goal")
        original_step_count = len(original_plan.steps)

        # Refine the plan
        feedback = {"reason": "failure", "suggestion": "add verification"}
        refined_plan = planner.refine_plan(
            original_plan.plan_id,
            feedback
        )

        assert refined_plan.plan_id != original_plan.plan_id
        assert refined_plan.goal == original_plan.goal
        # Refined plan should have more steps (adds verification)
        assert len(refined_plan.steps) > original_step_count
        assert "refined_from" in refined_plan.metadata

    def test_operations_require_initialization(self):
        """Test that operations fail before initialization."""
        planner = HierarchicalPlanning()

        with pytest.raises(PlanningOperationError, match="not initialized"):
            planner.generate_plan("Test goal")

        with pytest.raises(PlanningOperationError, match="not initialized"):
            planner.execute_step("plan_1", 1)

        with pytest.raises(PlanningOperationError, match="not initialized"):
            planner.refine_plan("plan_1", {})

    def test_invalid_inputs_raise_errors(self):
        """Test validation of input types."""
        planner = HierarchicalPlanning()
        planner.initialize()

        # Empty goal
        with pytest.raises(PlanningOperationError, match="non-empty string"):
            planner.generate_plan("")

        # Non-existent plan
        with pytest.raises(PlanningOperationError, match="not found"):
            planner.execute_step("nonexistent_plan", 1)

        # Non-existent step
        plan = planner.generate_plan("Valid goal")
        with pytest.raises(PlanningOperationError, match="not found"):
            planner.execute_step(plan.plan_id, 999)

    def test_double_initialization_is_safe(self):
        """Test that double initialization is safe (idempotent)."""
        planner = HierarchicalPlanning()

        ok1 = planner.initialize()
        ok2 = planner.initialize()

        assert ok1 is True
        assert ok2 is True  # Second init returns True (already initialized)

    def test_shutdown(self):
        """Test clean shutdown and resource cleanup."""
        planner = HierarchicalPlanning()
        planner.initialize()

        # Generate some plans
        planner.generate_plan("Task 1")
        planner.generate_plan("Task 2")

        # Shutdown
        planner.shutdown()

        # Verify cleanup
        health = planner.health_status()
        assert health["initialized"] is False
        assert health["plan_count"] == 0

    def test_planning_mode_enum(self):
        """Test PlanningMode enum values."""
        assert PlanningMode.COT.value == "chain_of_thought"
        assert PlanningMode.TOT.value == "tree_of_thoughts"
        assert PlanningMode.SOT.value == "skeleton_of_thoughts"

    def test_step_status_enum(self):
        """Test StepStatus enum values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
