# -*- coding: utf-8 -*-
"""Minimal tests for P1 three_layer_memory skeleton.

These tests aim to validate the minimal contract and provide good coverage for
Stage 1 gate with `.coveragerc` restricting coverage to this module.
"""
from three_layer_memory import ThreeLayerMemory, MemoryOperationError


def test_initialize_and_health():
    mem = ThreeLayerMemory()
    assert mem.initialize({"enabled": True, "warmup_ms": 0}) is True
    health = mem.health_status()
    assert health["initialized"] is True
    assert health["episodes"] == 0


def test_save_and_recall_last():
    mem = ThreeLayerMemory()
    assert mem.initialize({"enabled": True, "warmup_ms": 0}) is True
    mem.save_episode({"k": 1})
    mem.save_episode({"k": 2})
    out = mem.recall({})
    assert out["result"] == {"k": 2}
    assert out["count"] == 2


def test_save_and_recall_with_key():
    mem = ThreeLayerMemory()
    mem.initialize({"enabled": True, "warmup_ms": 0})
    for i in range(5):
        mem.save_episode({"k": i})
    out = mem.recall({"key": "k", "value": 3})
    assert out["result"] == {"k": 3}


def test_operations_require_initialization():
    mem = ThreeLayerMemory()
    try:
        mem.save_episode({})
        assert False, "Expected MemoryOperationError"
    except MemoryOperationError:
        pass
    try:
        mem.recall({})
        assert False, "Expected MemoryOperationError"
    except MemoryOperationError:
        pass


def test_shutdown():
    mem = ThreeLayerMemory()
    mem.initialize({"enabled": True, "warmup_ms": 0})
    mem.save_episode({"k": 1})
    mem.shutdown()
    health = mem.health_status()
    assert health["initialized"] is False
    assert health["episodes"] == 0


def test_recall_when_empty_returns_none():
    mem = ThreeLayerMemory()
    mem.initialize({"enabled": True, "warmup_ms": 0})
    out = mem.recall({})
    assert out["result"] is None
    assert out["count"] == 0


def test_trim_episodes_when_exceeding_capacity():
    mem = ThreeLayerMemory()
    mem.initialize({"enabled": True, "warmup_ms": 0, "max_episodes": 3})
    for i in range(10):
        mem.save_episode({"k": i})
    # Only the last 3 should remain
    out = mem.recall({})
    assert out["count"] == 3
    assert out["result"] == {"k": 9}


def test_invalid_inputs_raise_errors():
    mem = ThreeLayerMemory()
    mem.initialize({"enabled": True, "warmup_ms": 0})
    try:
        mem.save_episode([1, 2, 3])  # type: ignore[arg-type]
        assert False, "Expected MemoryOperationError for non-dict episode"
    except MemoryOperationError:
        pass
    try:
        mem.recall([1, 2, 3])  # type: ignore[arg-type]
        assert False, "Expected MemoryOperationError for non-dict query"
    except MemoryOperationError:
        pass
