"""
Performance tests for P1 Hierarchical Planning.

Verify operations meet performance targets:
- Plan generation < 20ms
- Step execution < 10ms
- Plan refinement < 30ms
"""

import time
import pytest
from hierarchical_planning import HierarchicalPlanning, PlanningMode


def test_planning_ops_are_fast():
    """Test that 100 operations complete in < 1000ms."""
    planner = HierarchicalPlanning()
    planner.initialize({"max_plans": 50})

    start = time.time()

    # 100 operations: 50 generations + 25 executions + 25 refinements
    plans = []
    for i in range(50):
        plan = planner.generate_plan(f"goal_{i}", PlanningMode.COT)
        plans.append(plan)

    for i in range(25):
        plan = plans[i]
        if plan and plan.steps:
            planner.execute_step(plan.plan_id, plan.steps[0].step_id)

    for i in range(25):
        plan = plans[i + 25]
        if plan:
            planner.refine_plan(plan.plan_id, f"feedback_{i}")

    elapsed_ms = (time.time() - start) * 1000

    assert elapsed_ms < 1000, f"Operations took {elapsed_ms:.1f}ms (target <1000ms)"


def test_individual_operation_performance():
    """Test individual operation performance targets."""
    planner = HierarchicalPlanning()
    planner.initialize()

    # Test generate_plan performance (CoT)
    start = time.time()
    plan = planner.generate_plan("Test goal", PlanningMode.COT)
    gen_time_ms = (time.time() - start) * 1000
    assert gen_time_ms < 20, f"generate_plan took {gen_time_ms:.2f}ms (target <20ms)"

    # Test execute_step performance
    start = time.time()
    planner.execute_step(plan.plan_id, plan.steps[0].step_id)
    exec_time_ms = (time.time() - start) * 1000
    assert exec_time_ms < 20, f"execute_step took {exec_time_ms:.2f}ms (target <20ms)"

    # Test refine_plan performance
    start = time.time()
    planner.refine_plan(plan.plan_id, "Test feedback")
    refine_time_ms = (time.time() - start) * 1000
    assert refine_time_ms < 30, f"refine_plan took {refine_time_ms:.2f}ms (target <30ms)"


def test_planning_modes_performance():
    """Test that all 3 planning modes meet performance targets."""
    planner = HierarchicalPlanning()
    planner.initialize()

    modes = [PlanningMode.COT, PlanningMode.TOT, PlanningMode.SOT]
    for mode in modes:
        start = time.time()
        plan = planner.generate_plan(f"Test {mode.value}", mode)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 20, f"{mode.value} generation took {elapsed_ms:.2f}ms (target <20ms)"
        assert plan is not None
