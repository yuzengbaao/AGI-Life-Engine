import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.distance import cosine
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

class NeuroSymbolicBridge:
    """
    The Neuro-Symbolic Bridge (NSB) connects the high-dimensional vector space (Neuro)
    with the structured knowledge graph (Symbolic).
    
    It addresses the 'Vector-Graph Disconnect' by monitoring:
    1. Semantic Drift: Divergence between a concept's original anchor vector and its current usage.
    2. Topological Surprise: Unexpected structural changes in the knowledge graph (e.g., new shortcuts).
    
    Mechanism:
    - If Drift is HIGH and Surprise is HIGH -> Valid Paradigm Shift (Update Anchor)
    - If Drift is HIGH but Surprise is LOW -> Hallucination/Noise (Reject)
    """

    def __init__(self, drift_threshold: float = 0.25, surprise_threshold: float = 0.6):
        """
        Initialize the bridge.
        
        Args:
            drift_threshold: Cosine distance threshold to flag semantic drift.
            surprise_threshold: Score threshold to flag structural surprise.
        """
        self.graph = nx.Graph()
        self.anchors: Dict[str, np.ndarray] = {}  # {concept_id: original_vector}
        self.current_embeddings: Dict[str, np.ndarray] = {} # {concept_id: current_vector}
        
        # Hyperparameters
        self.drift_threshold = drift_threshold
        self.surprise_threshold = surprise_threshold
        
        # State tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self.concept_states: Dict[str, str] = {}
        self.last_update_ts = time.time()

    def register_anchor(self, concept_id: str, vector: np.ndarray):
        """Register a semantic anchor (ground truth vector) for a concept."""
        if concept_id not in self.anchors:
            self.anchors[concept_id] = vector
            # logger.info(f"Anchor registered for {concept_id}")

    def update_topology(self, nodes: List[str], edges: List[Tuple[str, str]]):
        """Update the internal graph representation."""
        for node in nodes:
            self.graph.add_node(node)
        for u, v in edges:
            self.graph.add_edge(u, v)

    def calculate_semantic_drift(self, concept_id: str, current_vector: np.ndarray) -> float:
        """
        Calculate semantic drift (cosine distance) between anchor and current vector.
        Returns: 0.0 (identical) to 2.0 (opposite).
        """
        if concept_id not in self.anchors:
            # If no anchor exists, treat as 0 drift (or auto-anchor?)
            # For now, auto-anchor new concepts
            self.register_anchor(concept_id, current_vector)
            return 0.0
            
        anchor_vec = self.anchors[concept_id]
        # Ensure vectors are 1-D arrays
        v1 = anchor_vec.flatten()
        v2 = current_vector.flatten()
        
        try:
            dist = cosine(v1, v2)
            return float(dist)
        except Exception as e:
            logger.error(f"Error calculating drift for {concept_id}: {e}")
            return 0.0

    def calculate_topological_surprise(self, u: str, v: str) -> float:
        """
        Calculate 'Topological Surprise' when a new link (u, v) is proposed/observed.
        
        Logic:
        - If u and v were already close (short path), surprise is LOW.
        - If u and v were distant or disconnected, surprise is HIGH.
        - Closing a 'structural hole' yields high surprise.
        """
        if u not in self.graph or v not in self.graph:
            return 0.5 # Moderate surprise for new nodes
            
        try:
            # Check if path existed before this direct link
            if nx.has_path(self.graph, u, v):
                path_len = nx.shortest_path_length(self.graph, u, v)
                # If they were far apart (>3 hops), direct link is surprising
                if path_len > 3:
                    return 0.9 # High surprise (Bridge created)
                elif path_len == 3:
                    return 0.6
                elif path_len == 2:
                    return 0.2 # Triadic closure (expected)
                else:
                    return 0.0 # Already connected
            else:
                return 1.0 # Infinite distance -> Direct link is Maximum Surprise
        except Exception as e:
            logger.error(f"Error calculating surprise for {u}-{v}: {e}")
            return 0.0

    def evaluate_neuro_symbolic_state(self, concept_id: str, current_vector: np.ndarray, related_concepts: List[str] = []) -> Dict[str, Any]:
        """
        Core logic: Evaluate the state of a concept to determine if it's a valid evolution or hallucination.
        """
        # 1. Calculate Drift
        drift = self.calculate_semantic_drift(concept_id, current_vector)
        
        # 2. Calculate Structural Surprise (average of relations)
        surprise_scores = []
        for related in related_concepts:
            score = self.calculate_topological_surprise(concept_id, related)
            surprise_scores.append(score)
        
        avg_surprise = sum(surprise_scores) / len(surprise_scores) if surprise_scores else 0.0
        
        # 3. Decision Logic
        status = "STABLE"
        action = "MAINTAIN"
        confidence = 1.0
        
        if drift > self.drift_threshold:
            if avg_surprise > self.surprise_threshold:
                status = "PARADIGM_SHIFT"
                action = "UPDATE_ANCHOR" # Valid evolution
                confidence = 0.9 # High confidence in change
            else:
                status = "SEMANTIC_DRIFT" # High drift, low structural support
                action = "REJECT_NOISE" # Likely hallucination
                confidence = 0.4 # Low confidence
        else:
            if avg_surprise > self.surprise_threshold:
                status = "STRUCTURAL_DISCOVERY"
                action = "DEEPEN_CONNECTION"
            
        result = {
            "timestamp": time.time(),
            "concept_id": concept_id,
            "drift": drift,
            "surprise": avg_surprise,
            "status": status,
            "recommended_action": action,
            "confidence": confidence
        }
        
        self.metrics_history.append(result)
        self.concept_states[concept_id] = action # Store the action (REJECT_NOISE, MAINTAIN, etc.)
        return result

    def get_concept_state(self, concept_id: str) -> Optional[str]:
        """Get the last recorded state/action for a concept."""
        return self.concept_states.get(concept_id)

    def get_system_metrics(self) -> Dict[str, float]:
        """Return aggregated metrics for system dashboard."""
        if not self.metrics_history:
            return {"avg_drift": 0.0, "avg_surprise": 0.0}
            
        recent = self.metrics_history[-50:] # Last 50 events
        return {
            "avg_drift": sum(m["drift"] for m in recent) / len(recent),
            "avg_surprise": sum(m["surprise"] for m in recent) / len(recent),
            "anchors_count": len(self.anchors),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }
