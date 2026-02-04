import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.llm_client import LLMService

class ExperienceMemory:
    """
    A semantic memory system that stores and retrieves experiences using vector embeddings.
    Supports short-term (STM) and long-term memory (LTM) with consolidation and reflection.
    """
    
    def __init__(self, memory_dir: str = "data/memory") -> None:
        self.memory_dir: str = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        
        self.stm_file: str = os.path.join(self.memory_dir, "short_term.json")
        self.ltm_file: str = os.path.join(self.memory_dir, "long_term.json")
        
        self.llm: LLMService = LLMService()
        
        self.stm: List[Dict[str, Any]] = self._load_memory(self.stm_file)
        self.ltm: List[Dict[str, Any]] = self._load_memory(self.ltm_file)

    def _load_memory(self, file_path: str) -> List[Dict[str, Any]]:
        """Safely load memory from JSON file with error handling."""
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print(f"Warning: Invalid memory format in {file_path}. Expected list, got {type(data)}. Resetting.")
                    return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {file_path}: {e}. Resetting memory.")
            return []
        except PermissionError:
            print(f"Permission denied when reading {file_path}. Using empty memory.")
            return []
        except Exception as e:
            print(f"Unexpected error loading memory from {file_path}: {e}")
            return []

    def _save_memory(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Atomically save memory to disk with robust error handling."""
        if not isinstance(data, list):
            raise ValueError("Memory data must be a list of dictionaries.")
            
        temp_file = file_path + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, file_path)
        except PermissionError:
            print(f"Permission denied when writing to {file_path}")
            raise
        except OSError as e:
            print(f"OS error during save to {file_path}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
        except Exception as e:
            print(f"Failed to save memory to {file_path}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def load_context(self) -> Dict[str, Any]:
        """
        Load the current context state.
        (Added to fix AttributeError: 'ExperienceMemory' object has no attribute 'load_context')
        """
        return {
            "stm_count": len(self.stm),
            "ltm_count": len(self.ltm),
            "recent_memories": self.stm[-5:] if self.stm else []
        }

    def add_experience(self, context: str, action: str, outcome: float, details: Dict[str, Any]) -> None:
        """
        Record a new experience with vector embedding.
        
        Args:
            context: The situational context of the experience
            action: Action taken in this context
            outcome: Success score between 0.0 and 1.0
            details: Additional metadata about the experience
        """
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Context must be a non-empty string.")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Action must be a non-empty string.")
        if not isinstance(outcome, (float, int)) or not (0.0 <= outcome <= 1.0):
            raise ValueError("Outcome must be a float between 0.0 and 1.0.")

        # Generate embedding for the context
        try:
            embedding = self.llm.get_embedding(context)
        except Exception as e:
            print(f"Failed to generate embedding: {e}. Using placeholder.")
            embedding = [0.0] * 1536  # Common dimension for OpenAI models

        timestamp = details.get("timestamp", time.time())
        
        experience: Dict[str, Any] = {
            "context": context,
            "embedding": embedding,
            "action": action,
            "outcome": float(outcome),
            "details": details,
            "timestamp": timestamp
        }
        
        self.stm.append(experience)
        
        try:
            self._save_memory(self.stm, self.stm_file)
        except Exception:
            # If saving fails, keep running but don't attempt consolidation
            return

        # Trigger consolidation if STM gets too big
        if len(self.stm) > 100:
            self._consolidate_memory()

    def _consolidate_memory(self) -> None:
        """Move high-value experiences to LTM and clear STM."""
        try:
            # Keep only high-outcome experiences
            valuable_experiences: List[Dict[str, Any]] = [
                exp for exp in self.stm if exp['outcome'] > 0.8
            ]
            
            if not valuable_experiences:
                self.stm = []
                self._save_memory(self.stm, self.stm_file)
                return
                
            self.ltm.extend(valuable_experiences)
            self._save_memory(self.ltm, self.ltm_file)
            
            # Clear STM
            self.stm = []
            self._save_memory(self.stm, self.stm_file)
            
        except Exception as e:
            print(f"Error during memory consolidation: {e}")

    def recall_relevant(self, query: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using semantic similarity.
        
        Args:
            query: Query string to search for
            threshold: Minimum similarity score (0-1) for inclusion
            
        Returns:
            Top 5 most relevant memories sorted by relevance
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(threshold, (float, int)) or not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")

        try:
            query_embedding = self.llm.get_embedding(query)
            q_vec = np.array(query_embedding)
        except Exception as e:
            print(f"Failed to generate query embedding: {e}")
            return []

        results: List[Tuple[float, Dict[str, Any]]] = []
        all_memories = self.stm + self.ltm

        for memory in all_memories:
            # Skip corrected/deprecated memories
            if memory.get('status') == 'corrected':
                continue

            score = self._calculate_similarity(memory, q_vec)
            if score > threshold:
                results.append((score, memory))

        # Sort by score descending and return top 5
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:5]]

    def _calculate_similarity(self, memory: Dict[str, Any], q_vec: np.ndarray) -> float:
        """Calculate cosine similarity between memory and query vector."""
        # Vector Search
        if 'embedding' in memory and memory['embedding']:
            try:
                m_vec = np.array(memory['embedding'])
                
                # Check dimension mismatch
                if m_vec.shape != q_vec.shape:
                    return 0.0
                    
                norm_q = np.linalg.norm(q_vec)
                norm_m = np.linalg.norm(m_vec)
                
                if norm_q == 0 or norm_m == 0:
                    return 0.0
                    
                # Cosine similarity
                return float(np.dot(q_vec, m_vec) / (norm_q * norm_m))
            except Exception:
                pass
        
        # Fallback keyword match
        if 'context' in memory and isinstance(memory['context'], str):
            if query.lower() in memory['context'].lower():
                return 0.5
                
        return 0.0

    def reflect_and_correct(self, query: str, correction: str) -> str:
        """
        Correct erroneous memories through philosophical reflection.
        
        Args:
            query: The incorrect belief or memory to correct
            correction: The correct information
            
        Returns:
            Status message indicating number of memories corrected
        """
        if not isinstance(query, str) or not query.strip():
            return "Invalid query: must be a non-empty string."
        if not isinstance(correction, str) or not correction.strip():
            return "Invalid correction: must be a non-empty string."

        import time
        
        # Find relevant old memories
        relevant_memories = self.recall_relevant(query, threshold=0.7)
        
        current_time = time.time()
        
        for mem in relevant_memories:
            # Mark as corrected (preserving audit trail)
            mem['status'] = 'corrected'
            mem['correction_note'] = correction
            mem['corrected_at'] = current_time
            
            # Create reflective memory
            try:
                self.add_experience(
                    context=f"CORRECTION: {mem['context']} -> {correction}",
                    action="SelfReflection",
                    outcome=1.0,
                    details={
                        "original_id": mem.get('timestamp'),
                        "reason": correction,
                        "timestamp": current_time
                    }
                )
            except Exception as e:
                print(f"Failed to create reflective memory: {e}")

        # Save updated memories
        try:
            self._save_memory(self.ltm, self.ltm_file)
            self._save_memory(self.stm, self.stm_file)
        except Exception as e:
            print(f"Failed to save memories after correction: {e}")
            return f"Corrected {len(relevant_memories)} memories in memory but failed to persist changes: {e}"

        return f"Corrected {len(relevant_memories)} relevant memories."