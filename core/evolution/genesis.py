"""
Genesis Module
--------------
Handles the "Re-genesis" of the AGI system:
1. Verifies memory anchors (state files).
2. Restores/Synthesizes missing state from backups.
3. Ensures topological integrity of the cognitive engine.
"""

import os
import json
import shutil
import logging
import glob
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("Genesis")

# Paths
DATA_DIR = os.path.join(os.getcwd(), "data")
LOGS_DIR = os.path.join(os.getcwd(), "logs")
BACKBAG_DIR = os.path.join(os.getcwd(), "backbag", "latest")
BACKUP_DIR = os.path.join(os.getcwd(), "backbag")

# Target Files
CONSCIOUSNESS_FILE = os.path.join(DATA_DIR, "consciousness.json")
# The "Phantom" file the agent keeps looking for
COGNITIVE_ENGINE_STATE_FILE = "cognitive_engine_state.json" 

def _load_json(path: str) -> Optional[Dict]:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
    return None

def _save_json(path: str, data: Dict):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✅ Saved state to {path}")
    except Exception as e:
        logger.error(f"  ❌ Failed to save {path}: {e}")

def restore_memory_anchors():
    """
    Ensure critical state files exist. If not, synthesize them.
    """
    logger.info("🌌 Initiating Re-genesis: Checking Memory Anchors...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Check Consciousness State
    if not os.path.exists(CONSCIOUSNESS_FILE):
        logger.warning(f"  ⚠️ {CONSCIOUSNESS_FILE} missing. Attempting restoration...")
        
        # Try loading from backbag/latest/agi_llm_experience.json (as a proxy for state)
        experience_path = os.path.join(BACKBAG_DIR, "agi_llm_experience.json")
        learning_path = os.path.join(BACKBAG_DIR, "learning_state.json")
        
        experience = _load_json(experience_path)
        learning = _load_json(learning_path)
        
        # Synthesize a new state
        new_state = {
            "timestamp": time.time(),
            "attention": "RE-GENESIS",
            "cognitive_state": "AWAKE",
            "goals": [],
            "thoughts": ["System Re-genesis Initiated."],
            "sensory_summary": {"genesis": "Restored from backup artifacts."}
        }
        
        if learning:
             skills = list(learning.get('skills', {}).keys())
             new_state["thoughts"].append(f"Restored Skills: {skills}")
             
        _save_json(CONSCIOUSNESS_FILE, new_state)
    else:
        logger.info(f"  ✅ {CONSCIOUSNESS_FILE} exists.")

    # 2. Fix the "Phantom" Anchor (cognitive_engine_state.json)
    # The agent seems to expect this file in the CWD.
    phantom_path = os.path.join(os.getcwd(), COGNITIVE_ENGINE_STATE_FILE)
    if not os.path.exists(phantom_path):
        logger.warning(f"  ⚠️ {COGNITIVE_ENGINE_STATE_FILE} missing. Creating topological bridge...")
        
        # Map existing consciousness state to this file to satisfy the agent
        state = _load_json(CONSCIOUSNESS_FILE)
        if state:
            # Enforce the structure if needed, or just copy
            _save_json(phantom_path, state)
            logger.info(f"  🔗 Created topological bridge: {phantom_path}")
    else:
        logger.info(f"  ✅ {COGNITIVE_ENGINE_STATE_FILE} exists.")

async def perform_genesis(evolution_controller):
    """
    Execute the full Re-genesis sequence.
    """
    restore_memory_anchors()
    
    # 3. Force Dream Consolidation
    logger.info("💤 Forcing Dream Consolidation to break evolutionary stagnation...")
    if hasattr(evolution_controller, 'consolidate'):
        await evolution_controller.consolidate()
    elif hasattr(evolution_controller, 'memory') and hasattr(evolution_controller.memory, 'consolidate_memory_nocturnal'):
         await evolution_controller.memory.consolidate_memory_nocturnal()
         logger.info("  ✅ Consolidated memory via direct access.")
             
    logger.info("✨ Re-genesis Complete. System Integrity Restored.")
