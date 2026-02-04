"""
Performance tests for P1 Enhanced World Model.

Verify operations meet performance targets:
- State updates < 10ms
- Predictions < 50ms
- Counterfactual queries < 50ms
"""

import time
import pytest
from enhanced_world_model import EnhancedWorldModel


def test_world_model_ops_are_fast():
    """Test that 100 operations complete in < 500ms."""
    model = EnhancedWorldModel()
    model.initialize({"max_history": 50})

    start = time.time()

    # 100 operations: 50 updates + 25 predictions + 25 queries
    for i in range(50):
        model.update_state({"entities": {"step": i}})

    for i in range(25):
        model.predict({"action": f"action_{i}"}, steps=1)

    for i in range(25):
        model.query_counterfactual({"premise": f"premise_{i}"})

    elapsed_ms = (time.time() - start) * 1000

    assert elapsed_ms < 500, f"Operations took {elapsed_ms:.1f}ms (target <500ms)"


def test_individual_operation_performance():
    """Test individual operation performance targets."""
    model = EnhancedWorldModel()
    model.initialize()

    # Test update_state performance
    start = time.time()
    model.update_state({"entities": {"test": "data"}})
    update_time_ms = (time.time() - start) * 1000
    assert update_time_ms < 10, f"update_state took {update_time_ms:.2f}ms (target <10ms)"

    # Test predict performance
    start = time.time()
    model.predict({"action": "test"})
    predict_time_ms = (time.time() - start) * 1000
    assert predict_time_ms < 50, f"predict took {predict_time_ms:.2f}ms (target <50ms)"

    # Test query_counterfactual performance
    start = time.time()
    model.query_counterfactual({"premise": "test"})
    query_time_ms = (time.time() - start) * 1000
    assert query_time_ms < 50, f"query_counterfactual took {query_time_ms:.2f}ms (target <50ms)"
