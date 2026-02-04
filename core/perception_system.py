
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer

class PerceptionSystem:
    def __init__(self):
        try:
            print("   [Perception] 🧠 Loading Neural Embedding Model (all-MiniLM-L6-v2)...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_dim = 384
            print("   [Perception] ✅ Model Loaded Successfully.")
        except Exception as e:
            print(f"   [Perception] ⚠️ Failed to load SentenceTransformer: {e}")
            self.embedder = None
            self.model_dim = 384

    def encode_text(self, text: str) -> np.ndarray:
        """
        Convert text to a real semantic vector (384 dim).
        Fallback to random if model is missing (to prevent crash).
        """
        if not text:
            return np.zeros(self.model_dim, dtype=np.float32)
            
        if self.embedder:
            try:
                # Return vector directly
                return self.embedder.encode(text, convert_to_numpy=True)
            except Exception as e:
                print(f"   [Perception] ⚠️ Encoding failed: {e}")
                return np.random.randn(self.model_dim).astype(np.float32)
        else:
            # Fallback: Deterministic Random Projection (Better than MD5 hash)
            # Use hash as seed to generate a consistent pseudo-random vector
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            return rng.standard_normal(self.model_dim, dtype=np.float32)

    def encode_helix_state(self, context: dict, target_dim: int = 64) -> np.ndarray:
        """
        Generate the 64-dim state vector for DoubleHelixEngine.
        Uses PCA-like projection from 384-dim semantic space to 64-dim state space.
        """
        # 1. Construct rich semantic representation
        goal_desc = context.get('goal', '')
        if isinstance(goal_desc, dict): 
            goal_desc = goal_desc.get('description', '')
            
        visual_desc = str(context.get('visual_context', '') or '')
        
        rich_text = f"Goal: {goal_desc} | Visual: {visual_desc[:200]}"
        
        # 2. Get high-dim semantic vector (384)
        semantic_vec = self.encode_text(rich_text)
        
        # 3. Project to target dimension (e.g. 64)
        # Simple slicing is effective for distributed representations, 
        # but averaging pooling is better if we treat it as chunks.
        # Here we just resize/slice for speed and compatibility.
        if semantic_vec.shape[0] >= target_dim:
            state_vec = semantic_vec[:target_dim]
        else:
            state_vec = np.pad(semantic_vec, (0, target_dim - semantic_vec.shape[0]))
            
        # 4. Inject explicit scalars (Priority, Urgency) into the last few dimensions
        # This ensures the network 'knows' these critical values explicitly
        state_vec[-1] = float(context.get('priority_score', 0.5))
        state_vec[-2] = float(context.get('urgency', 0.0))
        state_vec[-3] = float(context.get('success_probability', 0.5))
        
        return state_vec
