import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("SkillLibrary")

class SkillLibrary:
    """
    The AGI's "Fluid Skill Store".
    Stores:
    1. Declarative Knowledge (Workflows, Notes) - Crystallized
    2. Procedural Knowledge (Executable Code, Macros) - Fluid/Narrow Tools
    """
    def __init__(self):
        # Path: data/memory/learned_skills.json
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memory")
        self.skills_file = os.path.join(self.base_dir, "learned_skills.json")
        self.code_dir = os.path.join(self.base_dir, "executable_skills")
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        
        self.skills: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.skills_file):
            return []
        
        try:
            with open(self.skills_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    logger.error("Invalid skills file format: expected list.")
                    return []
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load or parse skills file: {e}")
            return []

    def save(self) -> bool:
        """Returns True on success, False on failure."""
        try:
            with open(self.skills_file, 'w', encoding='utf-8') as f:
                json.dump(self.skills, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Skill Library saved.")
            return True
        except (OSError, TypeError) as e:
            logger.error(f"Failed to save skills: {e}")
            return False

    def add_executable_skill(self, name: str, description: str, code: str, skill_type: str = "python_script") -> Optional[Dict[str, Any]]:
        """
        Saves a generated tool/script as a reusable skill.
        This is the "Fluid" part - creating new organs for new tasks.
        Returns the created skill entry on success, None on failure.
        """
        if not name or not code:
            logger.error("Name and code are required to add an executable skill.")
            return None

        # Sanitize name for filename
        safe_name = "".join([c if c.isalnum() else "_" for c in name])
        filename = f"{safe_name}.py"
        file_path = os.path.join(self.code_dir, filename)
        
        # Write code to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to write skill code to {file_path}: {e}")
            return None

        # Create metadata entry
        entry = {
            "id": f"skill_{int(datetime.now().timestamp())}_{safe_name}",
            "name": name,
            "type": "executable",
            "subtype": skill_type,
            "description": description,
            "file_path": file_path,
            "created_at": datetime.now(),
            "usage_count": 0,
            "success_rate": 1.0
        }
        
        # Remove any existing skill with same name
        self.skills[:] = [s for s in self.skills if s.get("name") != name]
        self.skills.append(entry)

        if not self.save():
            # If saving fails, optionally remove the written file to avoid inconsistency
            try:
                os.remove(file_path)
            except:
                pass  # Best effort cleanup
            return None

        logger.info(f"Added executable skill: {name}")
        return entry

    def get_skill_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a skill by its name."""
        if not name:
            return None
        return next((s for s in self.skills if s.get("name") == name), None)

    def list_executable_skills(self) -> List[str]:
        """Return names of all executable skills."""
        return [s["name"] for s in self.skills if s.get("type") == "executable"]

    def add_learned_session(
        self, 
        session_type: str, 
        raw_observations: List[str], 
        summary: str, 
        extracted_logic: str
    ) -> Optional[Dict[str, Any]]:
        """
        Record a learning session.
        Returns the created entry on success, None on failure.
        """
        if not session_type or not summary:
            logger.error("Session type and summary are required.")
            return None

        entry = {
            "id": f"skill_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now(),
            "type": session_type,
            "summary": summary,
            "logic_flow": extracted_logic,
            "raw_observations_count": len(raw_observations),
            "mastery_level": "Novice"
        }
        self.skills.append(entry)
        
        if not self.save():
            return None

        return entry

    def get_recent_learnings(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent learned sessions, up to `limit`."""
        if limit <= 0:
            return []
        return self.skills[-limit:]