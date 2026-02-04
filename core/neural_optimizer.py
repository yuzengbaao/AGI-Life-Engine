import json
import os
import time
import logging

# Configure logger
logger = logging.getLogger("NeuralOptimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class NeuralOptimizer:
    def __init__(self, memory_dir=r"d:\TRAE_PROJECT\AGI\data\neural_memory"):
        self.memory_dir = memory_dir
        self.topology_path = os.path.join(memory_dir, "topology_graph.json")
        self.topology = {"n": 0, "adj": {}, "params": {}}
        self.loaded = False

    def load_graph(self):
        """Loads the topology graph from disk."""
        if not os.path.exists(self.topology_path):
            logger.warning(f"Topology file not found at {self.topology_path}")
            return False
        
        try:
            with open(self.topology_path, 'r', encoding='utf-8') as f:
                self.topology = json.load(f)
            self.loaded = True
            logger.info(f"Loaded topology with {self.topology.get('n', 0)} nodes.")
            return True
        except Exception as e:
            logger.error(f"Failed to load topology: {e}")
            return False

    def save_graph(self):
        """Saves the topology graph back to disk."""
        if not self.loaded:
            logger.warning("No graph loaded to save.")
            return False
            
        try:
            # Create backup
            if os.path.exists(self.topology_path):
                backup_path = self.topology_path + ".bak"
                with open(self.topology_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            with open(self.topology_path, 'w', encoding='utf-8') as f:
                json.dump(self.topology, f, indent=2)
            logger.info("Topology graph saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to save topology: {e}")
            return False

    def apply_hebbian_learning(self, source_idx: int, target_idx: int, learning_rate: float = 0.05):
        """
        Strengthens the connection between source and target nodes.
        'Neurons that fire together, wire together.'
        """
        if not self.loaded:
            self.load_graph()
            
        source_str = str(source_idx)
        adj = self.topology.get("adj", {})
        
        if source_str not in adj:
            adj[source_str] = []
        
        # Check if edge exists
        edge_found = False
        for edge in adj[source_str]:
            if edge["to_idx"] == target_idx:
                # Update existing edge
                old_weight = edge["weight"]
                edge["weight"] = min(1.0, old_weight + learning_rate)
                edge["usage"] = edge.get("usage", 0) + 1
                edge["last_active"] = time.time()
                
                logger.info(f"Hebbian update: Node {source_idx} -> Node {target_idx} (Weight: {old_weight:.2f} -> {edge['weight']:.2f})")
                edge_found = True
                break
        
        if not edge_found:
            # Create new connection (spontaneous association)
            new_edge = {
                "to_idx": target_idx,
                "weight": 0.1 + learning_rate, # Start low but boosted
                "usage": 1,
                "last_active": time.time()
            }
            adj[source_str].append(new_edge)
            logger.info(f"New synaptic connection formed: Node {source_idx} -> Node {target_idx}")

    def decay_and_prune(self, decay_rate: float = 0.001, prune_threshold: float = 0.05, max_age_days: int = 30):
        """
        Weakens all connections slightly (forgetting) and removes very weak/old ones (pruning).
        """
        if not self.loaded:
            self.load_graph()
            
        logger.info("Starting decay and pruning cycle...")
        adj = self.topology.get("adj", {})
        pruned_count = 0
        total_edges = 0
        
        now = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for src, edges in adj.items():
            surviving_edges = []
            for edge in edges:
                total_edges += 1
                
                # Apply decay
                # Don't decay if recently active (e.g., within last hour)
                last_active = edge.get("last_active", 0)
                if now - last_active > 3600:
                    edge["weight"] = max(0.0, edge["weight"] - decay_rate)
                
                # Pruning logic
                is_weak = edge["weight"] < prune_threshold
                is_old = (now - last_active) > max_age_seconds
                
                # Keep if strong OR active recently
                if not (is_weak and is_old):
                    surviving_edges.append(edge)
                else:
                    pruned_count += 1
            
            adj[src] = surviving_edges
            
        logger.info(f"Decay cycle complete. Pruned {pruned_count} synapses out of {total_edges}.")
        
    def optimize_cycle(self):
        """Runs a standard optimization cycle."""
        self.load_graph()
        self.decay_and_prune()
        self.save_graph()

if __name__ == "__main__":
    # Test run
    optimizer = NeuralOptimizer()
    optimizer.load_graph()
    
    # Simulate a learning event
    print("Simulating Hebbian learning event...")
    optimizer.apply_hebbian_learning(0, 5) # Strengthen connection between Node 0 and Node 5
    
    # Run maintenance
    print("Running maintenance...")
    optimizer.decay_and_prune()
    
    optimizer.save_graph()
