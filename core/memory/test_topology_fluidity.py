
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.memory.topology_memory import TopologicalMemoryCore

def test_fluid_intelligence_ops():
    print("=== Testing Fluid Intelligence Topology Operations ===")
    
    # Initialize Core
    topo = TopologicalMemoryCore(max_degree=5, min_edge_weight=0.1)
    
    # 1. Simulate Growth (Adding Nodes)
    print("\n[1] Growth & Connection:")
    # Create 5 nodes
    for i in range(5):
        topo.ensure_size(i + 1)
    
    # Connect 0 -> 1 -> 2 -> 3 -> 4 in a chain
    topo.connect(0, 1, weight=0.8)
    topo.connect(1, 2, weight=0.7)
    topo.connect(2, 3, weight=0.9)
    topo.connect(3, 4, weight=0.6)
    
    print(f"Node 1 neighbors: {[e.to_idx for e in topo.get_edges(1)]}")
    print(f"Node 2 neighbors: {[e.to_idx for e in topo.get_edges(2)]}")
    
    # 2. Test Disconnection (Plasticity)
    print("\n[2] Disconnection (Plasticity):")
    print("Disconnecting 1 <-> 2...")
    topo.disconnect(1, 2)
    print(f"Node 1 neighbors after disconnect: {[e.to_idx for e in topo.get_edges(1)]}")
    print(f"Node 2 neighbors after disconnect: {[e.to_idx for e in topo.get_edges(2)]}")
    
    # Reconnect for cloning test
    topo.connect(1, 2, weight=0.7)
    
    # 3. Test Subgraph Cloning (Fractal/Copy)
    print("\n[3] Subgraph Cloning:")
    # We want to clone the subgraph {1, 2, 3}
    # This should create new nodes that replicate the structure 1-2-3
    # The new nodes will be appended to the end. Current max index is 4.
    # So new nodes should be 5, 6, 7 corresponding to 1, 2, 3.
    
    mapping = topo.clone_subgraph([1, 2, 3])
    print(f"Cloned mapping: {mapping}")
    
    # Check connections of the clone
    new_1 = mapping[1] # 5
    new_2 = mapping[2] # 6
    new_3 = mapping[3] # 7
    
    print(f"Original 1-2 weight: {topo.get_edge_weight(1, 2)}")
    print(f"Cloned {new_1}-{new_2} weight: {topo.get_edge_weight(new_1, new_2)}")
    
    print(f"Original 2-3 weight: {topo.get_edge_weight(2, 3)}")
    print(f"Cloned {new_2}-{new_3} weight: {topo.get_edge_weight(new_2, new_3)}")
    
    # Verify the clone is detached from the original (unless we explicitly connected them, but clone_subgraph is usually internal structure)
    # Check if 1 is connected to 5 (it shouldn't be by default unless the logic does so)
    print(f"Is Original 1 connected to Clone {new_1}? {topo.get_edge_weight(1, new_1) > 0}")

    # 4. Test Homeostasis (Energy/Decay)
    print("\n[4] Homeostasis & Pruning:")
    # Add a weak connection
    topo.connect(0, 4, weight=0.12) # Just above min_edge_weight 0.1
    print(f"Weak edge 0-4 created with weight 0.12")
    
    # Test Homeostasis (Normalization)
    # Homeostasis ensures sum of outgoing weights is normalized (or balanced). 
    # The current implementation sums weights and divides by total, so sum becomes 1.0.
    print("Running homeostasis (normalization)...")
    topo.homeostasis()
    
    edges_0 = topo.get_edges(0)
    total_w = sum(e.weight for e in edges_0)
    print(f"Sum of weights for Node 0 after homeostasis: {total_w:.4f}")
    
    # Test Pruning (Decay)
    # To force pruning, we can temporarily increase decay or just call prune many times.
    # Current weight decay is 0.999 (very slow).
    # Let's manually set a high decay for testing.
    topo.weight_decay = 0.5 
    print("Set aggressive weight_decay = 0.5")
    
    print("Running prune()...")
    topo.prune()
    
    # Check if edge 0-4 (which was weak) survived.
    # It started at 0.12 (normalized).
    # If Node 0 had other edges (to 1), the normalized weight of 0-4 might be small.
    # Node 0 was connected to 1 (weight 0.8) and 4 (weight 0.12).
    # Total = 0.92. 
    # Normalized 0-4 = 0.12 / 0.92 = 0.13.
    # Normalized 0-1 = 0.8 / 0.92 = 0.87.
    # After prune (decay 0.5):
    # 0-4 -> 0.13 * 0.5 = 0.065.
    # min_edge_weight is 0.1 (default in test init was 0.1? No, wait.)
    # In test init: TopologicalMemoryCore(max_degree=5, min_edge_weight=0.1)
    # So 0.065 < 0.1 => Pruned!
    
    w_0_4_pruned = topo.get_edge_weight(0, 4)
    print(f"Edge 0-4 exists after prune? {w_0_4_pruned > 0}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_fluid_intelligence_ops()
