import time
import json
import os
from typing import List, Dict, Any, Optional, Deque
from collections import deque
import logging

logger = logging.getLogger("GlobalWorkspace")

class GlobalWorkspace:
    """
    The 'Seat of Consciousness' for the AGI.
    Implements the Global Workspace Theory (GWT) to provide a unified
    context for decision making, bridging the gap between discrete modules.
    
    It maintains:
    1. Working Memory (Short-term active thoughts)
    2. Sensory Buffer (Recent inputs from all observers)
    3. Goal Stack (Hierarchical intentions)
    4. Attention Spotlight (What is currently important)
    """
    def __init__(self, max_working_memory: int = 10):
        self.working_memory: Deque[str] = deque(maxlen=max_working_memory)
        self.sensory_buffer: Dict[str, Any] = {}
        self.goal_stack: List[Dict[str, Any]] = []
        self.attention_spotlight: str = "IDLE"
        self.cognitive_state: str = "AWAKE" # AWAKE, REFLECTING, DREAMING
        self.start_time = time.time()
        self.state_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "consciousness.json")
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
    def _persist_state(self):
        """Dump current state to file for external observation."""
        try:
            state = {
                "timestamp": time.time(),
                "attention": self.attention_spotlight,
                "cognitive_state": self.cognitive_state,
                "goals": list(self.goal_stack),
                "thoughts": list(self.working_memory),
                "sensory_summary": {k: str(v['data'])[:50] for k, v in self.sensory_buffer.items()}
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error persisting consciousness: {e}")

    def update_sensory(self, source: str, data: Any):
        """Update the sensory buffer with fresh data from observers."""
        self.sensory_buffer[source] = {
            "data": data,
            "timestamp": time.time()
        }
        self._persist_state()
        
    def add_thought(self, thought: str):
        """Add a conscious thought to working memory."""
        timestamp = time.strftime("%H:%M:%S")
        self.working_memory.append(f"[{timestamp}] {thought}")
        # Log thoughts as "Inner Monologue"
        print(f"   💭 {thought}")
        self._persist_state()
        
    def push_goal(self, goal: str, priority: str = "medium"):
        """Push a new goal onto the stack."""
        self.goal_stack.append({
            "goal": goal,
            "priority": priority,
            "created_at": time.time(),
            "status": "active"
        })
        self.add_thought(f"New Goal Adopted: {goal}")
        self._persist_state()
        
    def pop_goal(self):
        """Complete or abandon the current top goal."""
        if self.goal_stack:
            done_goal = self.goal_stack.pop()
            self.add_thought(f"Goal Completed/Dropped: {done_goal['goal']}")
            self._persist_state()
            return done_goal
        return None
        
    def get_current_context_prompt(self) -> str:
        """
        Compile the entire Global Workspace into a prompt for the LLM.
        This provides the 'Continuity of Consciousness'.
        """
        # 1. Format Sensory Inputs
        sensory_text = []
        for source, info in self.sensory_buffer.items():
            # Only show fresh data (< 60s old)
            if time.time() - info['timestamp'] < 60:
                sensory_text.append(f"- {source}: {str(info['data'])[:200]}") # Truncate long data
        
        # 2. Format Goal Stack
        goals_text = []
        for g in reversed(self.goal_stack): # Show most recent first
            goals_text.append(f"- [{g['priority'].upper()}] {g['goal']}")
            
        # 3. Format Working Memory (Stream of Consciousness)
        memory_text = list(self.working_memory)
        
        prompt = f"""
        [GLOBAL WORKSPACE STATE]
        Current Attention: {self.attention_spotlight}
        Cognitive State: {self.cognitive_state}
        
        [ACTIVE GOALS]
        {chr(10).join(goals_text) if goals_text else "(No active goals)"}
        
        [SENSORY INPUTS (Fresh)]
        {chr(10).join(sensory_text) if sensory_text else "(No fresh inputs)"}
        
        [STREAM OF CONSCIOUSNESS (Recent Thoughts)]
        {chr(10).join(memory_text)}
        """
        return prompt

    def clear_stale_data(self):
        """Cleanup routine."""
        now = time.time()
        # Remove old sensory data
        stale_keys = [k for k, v in self.sensory_buffer.items() if now - v['timestamp'] > 300]
        for k in stale_keys:
            del self.sensory_buffer[k]
