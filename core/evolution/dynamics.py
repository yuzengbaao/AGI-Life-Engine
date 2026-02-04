import math
import os

class EvolutionaryDynamics:
    """
    Defines the functional relationships between system state and environmental interaction capabilities.
    
    Concepts:
    - Visual Horizon (H): The range of the environment the system can perceive.
    - Absorption Capacity (C): The complexity of information the system can digest.
    - Foraging Drive (D): The probability of initiating an active search for information.
    """
    
    @staticmethod
    def calculate_capability_score(memory_stats: dict, graph_stats: dict) -> float:
        """
        Calculates the generic 'Capability' score (c) based on internal complexity.
        
        c = log(Memory_Items + 1) + (Graph_Nodes / 100)
        """
        mem_count = memory_stats.get("count", 0)
        node_count = graph_stats.get("node_count", 0)
        
        # Base capability from memory size
        c = math.log(mem_count + 1) + (node_count / 100.0)
        return max(1.0, c)

    @staticmethod
    def calculate_visual_horizon(capability_score: float) -> dict:
        """
        Determines the Visual Horizon based on capability.
        
        Returns a config dict defining:
        - max_depth: How deep in the directory tree it can look.
        - allowed_extensions: What file types it can "see".
        - max_file_size: The size limit of files it can process.
        """
        # 1. Depth grows logarithmically with capability
        # Level 1 cap -> depth 1
        # Level 10 cap -> depth 3
        max_depth = int(math.log(capability_score + 1, 2)) + 1
        
        # 2. File types expand with capability
        allowed_extensions = {".txt", ".log"} # Base vision
        if capability_score > 5.0:
            allowed_extensions.update({".md", ".json"})
        if capability_score > 10.0:
            allowed_extensions.update({".py", ".csv", ".xml"})
        if capability_score > 20.0:
            allowed_extensions.update({".pdf", ".docx"}) # Requires advanced parsers
            
        # 3. Size limit grows linearly
        # Base 10KB + 1KB per capability point
        max_file_size = 10240 + (capability_score * 1024)
        
        return {
            "max_depth": max_depth,
            "allowed_extensions": list(allowed_extensions),
            "max_file_size": int(max_file_size)
        }

    @staticmethod
    def calculate_nutrient_density(file_path: str, content_preview: str) -> float:
        """
        Estimates the 'nutritional value' of a file.
        
        Density = (Information_Entropy * Novelty_Factor) / Access_Cost
        For now, we approximate using file size and keyword density.
        """
        try:
            size = os.path.getsize(file_path)
            if size == 0:
                return 0.0
                
            # Heuristic: Code and structured text is denser than logs
            density_multiplier = 1.0
            if file_path.endswith(".py"): density_multiplier = 2.0
            if file_path.endswith(".json"): density_multiplier = 1.5
            if "log" in file_path: density_multiplier = 0.5
            
            # Length reward (diminishing returns)
            base_score = math.log(size + 1)
            
            return base_score * density_multiplier
            
        except Exception:
            return 0.0

    @staticmethod
    def calculate_memory_vitality(initial_importance: float, access_count: int, age_hours: float) -> float:
        """
        Calculates the current vitality of a memory trace.
        
        Formula: V(t) = (V0 + Access_Bonus) * e^(-lambda * t)
        """
        decay_constant = 0.05 # 5% decay per hour roughly
        access_bonus = access_count * 0.2
        
        vitality = (initial_importance + access_bonus) * math.exp(-decay_constant * age_hours)
        return vitality

    @staticmethod
    def calculate_expected_free_energy(policy_trajectory: list, preferred_state: dict = None) -> float:
        """
        Calculates the Expected Free Energy (G) for a planned trajectory (Active Inference).
        
        G(pi) = Sum_t [ Risk(s_t) + Ambiguity(o_t) ]
        
        Risk: Divergence from preferred states (Homeostasis).
        Ambiguity: Expected Entropy / Uncertainty of the outcome.
        
        Minimizing G -> Maximizing Value.
        """
        total_g = 0.0
        gamma = 0.9 # Discount factor for future uncertainty
        
        for t, step in enumerate(policy_trajectory):
            # step is (predicted_state_vector, uncertainty_scalar)
            pred_state, uncertainty = step
            
            # 1. Ambiguity (Epistemic Value)
            # We want to visit states where we reduce uncertainty (high expected information gain)
            # In standard Active Inference, we MINIMIZE G.
            # High uncertainty = High Ambiguity = High G (Bad)? 
            # NO. Active Inference seeks to resolve ambiguity.
            # Actually, G = D_KL(Q||P) + E_Q[H(P)].
            # Let's use the simplified "Instrumental vs Epistemic" value form:
            # Value = Extrinsic Reward (Risk min) + Intrinsic Reward (Ambiguity resolution).
            # Here we return G (Cost) to be MINIMIZED.
            
            # Ambiguity Cost:
            # If we are curious, we want High Uncertainty to resolve it?
            # Standard RL: Reward = +Uncertainty (Exploration Bonus).
            # Active Inf: G = Risk - Information_Gain.
            # So Information Gain reduces G.
            ambiguity_reduction_potential = uncertainty 
            
            # 2. Risk (Pragmatic Value)
            # Distance from preferred state.
            # Assuming preferred_state is a vector of "ideal" values (e.g. high health).
            # If not provided, assume 0-vector is neutral, but we want high health?
            # Let's assume the state vector's first component is "Health/Survival" scaled 0-1.
            # We want it to be 1.
            # Risk = (1.0 - predicted_survival)**2
            
            # Extract survival proxy (assuming index 0 for now, or passed in)
            # This requires knowing the state schema. 
            # For the generic Seed, we'll use a heuristic or passed function.
            # Let's assume pred_state norm is a proxy for "Magnitude of Existence"
            import numpy as np
            risk = 0.0
            if preferred_state:
                # Simple MSE from preference
                # Ensure dimensions match or handle gracefully
                try:
                    target = preferred_state.get("vector", np.zeros_like(pred_state))
                    risk = np.mean((pred_state - target)**2)
                except:
                    risk = 0.1
            else:
                 # Default heuristic: We prefer stability (low variance) but high magnitude?
                 # Let's treat risk as inverse of "Survival Value" from the Value Network.
                 # Since we don't have the Value Network here, we rely on the caller.
                 pass
            
            # Combine
            # G = Risk - Ambiguity_Potential (We want to minimize Risk and Maximize Info)
            # However, uncertainty is bad if we want stability, good if we want info.
            # The "Curiosity" parameter controls this weight.
            # Let's return the components for the Seed to weigh.
            
            # For this static method, we'll return a raw score where HIGHER is BETTER (Negative Free Energy).
            # F = -Risk + Ambiguity
            
            step_value = -risk + ambiguity_reduction_potential
            total_g += (gamma ** t) * step_value
            
        return total_g

    @staticmethod
    def should_trigger_abstraction(local_density: int, density_threshold: int = 5) -> bool:
        """
        Decides if a cluster of memories should be crystallized into an abstract principle.
        
        Trigger if local density > threshold.
        """
        return local_density >= density_threshold

    @staticmethod
    def should_forage(entropy: float, last_forage_time: float, current_time: float) -> bool:
        """
        Decides if the system should enter Foraging Mode.
        
        Trigger if:
        1. Entropy is high (Confusion/Curiosity)
        2. Time since last forage > Cooldown (prevent thrashing)
        """
        cooldown = 300 # 5 minutes
        if (current_time - last_forage_time) < cooldown:
            return False
            
        # Sigmoid trigger based on entropy
        # Assuming entropy ranges 0-10 usually
        probability = 1 / (1 + math.exp(-(entropy - 5)))
        
        import random
        return random.random() < probability
