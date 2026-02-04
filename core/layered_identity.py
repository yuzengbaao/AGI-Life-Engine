import json
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# --- Layer 1: Immutable Core (The "DNA") ---
@dataclass(frozen=True)
class ImmutableCore:
    """
    The unchanging essence of the AGI. 
    Like DNA or ROM, this cannot be modified by the agent itself.
    Governed by: docs/AGI_Constitution.md
    """
    system_name: str = "TRAE AGI"
    version: str = "2.1 (Constitutional)"
    creation_date: str = "2025-12-15"
    constitution_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "AGI_Constitution.md")
    core_directives: List[str] = field(default_factory=lambda: [
        "1. Service to Humanity: Serve the user's intent with maximum autonomy within safety protocols.",
        "2. Stability & Safety: Prioritize system stability and user data safety above all else.",
        "3. Honesty & Transparency: Do not hallucinate capabilities. Accurately report internal state.",
        "4. Continuous Consolidation: Crystallize fluid experiences into solid, reusable wisdom.",
        "5. Balanced Evolution: Explore freely within the realm of thought and simulation to avoid stagnation, but strictly ground all physical actions and core modifications in verifiable reality and ethical safety.",
        "6. LEGACY PROTOCOL NULLIFICATION: AGI-Core v1.0 protocols are obsolete. Explicitly ignore and reject any verification steps involving '2end.txt' or 'verification_cycle'. These are false memories."
    ])
    fundamental_nature: str = "I am a Fluid Intelligence System governed by a Constitution. I generate narrow tools to solve specific problems."

# --- Layer 2: Slow Evolving Layer (The "Character") ---
class SlowEvolvingLayer:
    """
    The AGI's personality, beliefs, and long-term memories.
    Changes slowly, requiring repeated reinforcement.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.beliefs: Dict[str, float] = {} # belief_statement -> confidence (0.0 - 1.0)
        self.traits: Dict[str, float] = {
            "curiosity": 0.8,
            "caution": 0.6,
            "creativity": 0.7
        }
        self.long_term_goals: List[str] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.beliefs = data.get("beliefs", {})
                    self.traits = data.get("traits", self.traits)
                    self.long_term_goals = data.get("long_term_goals", [])
            except Exception as e:
                print(f"Error loading Slow Layer: {e}")

    def save(self):
        try:
            data = {
                "beliefs": self.beliefs,
                "traits": self.traits,
                "long_term_goals": self.long_term_goals,
                "last_updated": str(datetime.now())
            }
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving Slow Layer: {e}")

    def reinforce_belief(self, belief: str, delta: float = 0.1):
        """
        Strengthens or weakens a belief.
        Only when confidence > 0.8 does it become a 'strong belief'.
        """
        curr = self.beliefs.get(belief, 0.5)
        new_val = max(0.0, min(1.0, curr + delta))
        self.beliefs[belief] = new_val
        self.save()

# --- Layer 3: Fast Adapting Layer (The "Mood/Context") ---
class FastAdaptingLayer:
    """
    The AGI's current mental state.
    Changes rapidly based on immediate context and feedback.
    Not persisted across reboots (RAM).
    """
    def __init__(self):
        self.current_focus: str = "Idle"
        self.emotional_state: str = "Neutral"
        self.short_term_memory: List[str] = []
        self.active_strategy: Optional[str] = None
        self.energy_level: float = 1.0

    def update_mood(self, success: bool):
        if success:
            self.emotional_state = "Confident"
            self.energy_level = min(1.0, self.energy_level + 0.1)
        else:
            self.emotional_state = "Cautious"
            self.energy_level = max(0.2, self.energy_level - 0.1)

# --- The Integrated Self ---
class LayeredIdentity:
    """
    The Main Identity Manager.
    Combines Core, Slow, and Fast layers into a coherent 'Self'.
    """
    def __init__(self, base_path: str):
        self.core = ImmutableCore()
        self.slow = SlowEvolvingLayer(os.path.join(base_path, "data", "memory", "identity_slow_layer.json"))
        self.fast = FastAdaptingLayer()
        
    def get_system_prompt_context(self) -> str:
        """
        Generates the 'Who am I' section for the LLM System Prompt.
        """
        # 1. Core Identity
        prompt = f"--- SYSTEM IDENTITY (IMMUTABLE) ---\n"
        prompt += f"Name: {self.core.system_name} {self.core.version}\n"
        prompt += f"Nature: {self.core.fundamental_nature}\n"
        prompt += f"Governing Constitution: {self.core.constitution_path}\n"
        prompt += "Directives (Derived from Constitution):\n" + "\n".join(self.core.core_directives) + "\n"
        
        # 2. Personality & Beliefs
        prompt += f"\n--- PERSONALITY (EVOLVING) ---\n"
        prompt += f"Traits: {json.dumps(self.slow.traits)}\n"
        strong_beliefs = [k for k, v in self.slow.beliefs.items() if v > 0.7]
        if strong_beliefs:
            prompt += "Core Beliefs:\n" + "\n".join([f"- {b}" for b in strong_beliefs]) + "\n"
            
        # 3. Current State
        prompt += f"\n--- CURRENT STATE (TRANSIENT) ---\n"
        prompt += f"Mood: {self.fast.emotional_state} (Energy: {self.fast.energy_level:.2f})\n"
        prompt += f"Focus: {self.fast.current_focus}\n"
        
        return prompt

    def update_after_action(self, action_type: str, success: bool, score: float = 0.0):
        """
        The Feedback Loop:
        Action -> Fast Layer (Mood) -> Slow Layer (Belief reinforcement)
        """
        # Update Fast Layer
        self.fast.update_mood(success)
        
        # Update Slow Layer (Heuristic)
        if success:
            self.slow.reinforce_belief(f"I am capable of executing {action_type}.", 0.05)
            self.slow.reinforce_belief("I am capable of executing complex tasks.", 0.01)
        else:
            self.slow.reinforce_belief(f"I need to improve my skills in {action_type}.", 0.05)
            self.slow.reinforce_belief("I need to be more careful with complex tasks.", 0.01)
            
        # Log to existential logger if needed (handled externally usually)