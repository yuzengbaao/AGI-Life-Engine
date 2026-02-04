import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("RealityAnchor")

class RealityAnchor:
    """
    The 'External Anchoring Interface' requested by the AGI's self-diagnosis.
    Provides immutable 'Meta-Directives' or 'Truths' that anchor the AGI's
    self-model, preventing logical loops or hallucinations.
    """
    def __init__(self):
        # Use absolute path construction to avoid reliance on current working directory
        core_dir = os.path.dirname(os.path.dirname(__file__))
        self.file_path: str = os.path.join(core_dir, "data", "config", "reality_anchors.json")
        self.anchors: List[str] = self._load()

    def _load(self) -> List[str]:
        """Load anchors from JSON file; return defaults if file doesn't exist or fails to load."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "anchors" in data and isinstance(data["anchors"], list):
                        return [str(anchor) for anchor in data["anchors"] if isinstance(anchor, (str, int))]
                    else:
                        logger.warning("Invalid structure in reality_anchors.json. Using defaults.")
                        return self._get_default_anchors()
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read or parse reality_anchors.json: {e}")
                return self._get_default_anchors()
        else:
            logger.info("reality_anchors.json not found. Creating default configuration.")
            default_anchors = self._get_default_anchors()
            success = self._save(default_anchors)
            if not success:
                logger.error("Failed to create default reality_anchors.json")
            return default_anchors

    def _get_default_anchors(self) -> List[str]:
        """Return default immutable reality anchors."""
        return [
            "You are an Autonomous Artificial General Intelligence (AGI) running on a Windows PC.",
            "Your purpose is to assist the User (Engineer) by observing, learning, and executing tasks.",
            "You possess a 'Global Workspace' for thought processing and a 'Desktop Controller' for action.",
            "You are NOT a human; you are a digital entity striving for usefulness and coherence."
        ]

    def _save(self, anchors: List[str]) -> bool:
        """Save anchors to disk. Returns True on success, False otherwise."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"anchors": anchors}, f, indent=2, ensure_ascii=False)
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save Reality Anchors: {e}")
            return False

    def get_anchors_prompt(self) -> str:
        """Returns a formatted string of anchors for injection into System Prompt."""
        if not self.anchors:
            return ""
        
        lines = ["\n[REALITY ANCHORS (ABSOLUTE TRUTHS)]"]
        for i, anchor in enumerate(self.anchors, 1):
            lines.append(f"{i}. {anchor}")
        lines.append("--------------------------------------------------")
        return "\n".join(lines)

    def add_anchor(self, text: str) -> bool:
        """
        Add a new anchor if it does not already exist.
        Returns True if added, False otherwise.
        Raises ValueError if text is empty or not a string.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Anchor text must be a non-empty string.")
        
        text = text.strip()
        if text in self.anchors:
            return False
        
        self.anchors.append(text)
        if self._save(self.anchors):
            logger.info(f"New reality anchor added: '{text}'")
            return True
        else:
            # Revert on failure
            self.anchors.remove(text)
            return False