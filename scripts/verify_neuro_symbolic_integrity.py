import sys
import os
import numpy as np
import networkx as nx
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neuro_symbolic_bridge import NeuroSymbolicBridge
from core.knowledge_graph import ArchitectureKnowledgeGraph

def verify_integrity():
    print("🔬 Starting Neuro-Symbolic Bridge Integrity Verification...")
    report = []

    # 1. Attribute Initialization Check
    print("\n[Test 1] Checking Attribute Initialization...")
    try:
        bridge = NeuroSymbolicBridge()
        if hasattr(bridge, 'concept_states') and isinstance(bridge.concept_states, dict):
            print("   ✅ 'concept_states' attribute initialized correctly.")
            report.append("Attribute 'concept_states': PASS")
        else:
            print("   ❌ 'concept_states' attribute MISSING or Invalid.")
            report.append("Attribute 'concept_states': FAIL")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        report.append(f"Initialization: FAIL ({e})")
        return

    # 2. Knowledge Graph Hydration Check
    print("\n[Test 2] Checking Knowledge Graph Hydration Logic...")
    try:
        # Load real graph
        kg = ArchitectureKnowledgeGraph()
        node_count = kg.graph.number_of_nodes()
        edge_count = kg.graph.number_of_edges()
        print(f"   ℹ️  Real Knowledge Graph contains {node_count} nodes and {edge_count} edges.")

        if node_count == 0:
            print("   ⚠️  Warning: Real Knowledge Graph is empty. Cannot verify hydration fully.")
            report.append("Hydration: SKIPPED (Empty Graph)")
        else:
            # Simulate Hydration
            start_time = time.time()
            bridge.update_topology(
                nodes=list(kg.graph.nodes()),
                edges=list(kg.graph.edges())
            )
            duration = time.time() - start_time
            
            bridge_nodes = bridge.graph.number_of_nodes()
            bridge_edges = bridge.graph.number_of_edges()
            
            if bridge_nodes == node_count and bridge_edges == edge_count:
                print(f"   ✅ Hydration Successful. Bridge synced {bridge_nodes} nodes in {duration:.4f}s.")
                report.append("Hydration: PASS")
            else:
                print(f"   ❌ Hydration Mismatch! Bridge has {bridge_nodes} nodes (Expected {node_count}).")
                report.append("Hydration: FAIL (Mismatch)")

    except Exception as e:
        print(f"   ❌ Hydration test failed: {e}")
        report.append(f"Hydration: FAIL ({e})")

    # 3. Functional Logic Check (evaluate_neuro_symbolic_state)
    print("\n[Test 3] Checking Core Evaluation Logic...")
    try:
        # Mock inputs
        concept_id = "test_concept_v1"
        vector = np.random.rand(128)
        
        # Run evaluation
        result = bridge.evaluate_neuro_symbolic_state(concept_id, vector, related_concepts=[])
        
        # Verify result structure
        required_keys = ["status", "recommended_action", "confidence", "drift", "surprise"]
        if all(k in result for k in required_keys):
            print(f"   ✅ Evaluation returned valid structure. Action: {result['recommended_action']}")
            report.append("Evaluation Logic: PASS")
        else:
            print(f"   ❌ Evaluation returned missing keys. Got: {result.keys()}")
            report.append("Evaluation Logic: FAIL (Structure)")
            
        # Verify state persistence
        saved_state = bridge.get_concept_state(concept_id)
        if saved_state == result['recommended_action']:
            print(f"   ✅ Concept state persisted correctly: {saved_state}")
            report.append("State Persistence: PASS")
        else:
            print(f"   ❌ Concept state NOT persisted. Expected {result['recommended_action']}, got {saved_state}")
            report.append("State Persistence: FAIL")

    except Exception as e:
        print(f"   ❌ Evaluation test failed: {e}")
        report.append(f"Evaluation Logic: FAIL ({e})")

    # Final Summary
    print("\n" + "="*40)
    print("VERIFICATION SUMMARY")
    print("="*40)
    for item in report:
        print(f"- {item}")
    print("="*40)

if __name__ == "__main__":
    verify_integrity()
