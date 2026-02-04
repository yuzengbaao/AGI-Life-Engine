# -*- coding: utf-8 -*-
"""Stage 2 minimal performance test for P1 memory.

We validate basic operations are fast enough under tiny workloads. The exact
thresholds are intentionally generous to be robust on CI.
"""
import time
from three_layer_memory import ThreeLayerMemory


def test_memory_ops_are_fast():
    mem = ThreeLayerMemory()
    assert mem.initialize({"enabled": True, "warmup_ms": 0})

    t0 = time.time()
    for i in range(200):
        mem.save_episode({"i": i})
        mem.recall({})
    elapsed_ms = (time.time() - t0) * 1000

    # Should be well below 500ms given trivial in-memory operations
    assert elapsed_ms < 500, f"elapsed {elapsed_ms:.2f}ms too slow"
