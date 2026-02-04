import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from threading import Thread, Event
from pynput import mouse, keyboard
from core.desktop_automation import DesktopController

# Configure logging
logger = logging.getLogger(__name__)

ActionType = Literal["click", "type", "key", "wait", "move", "scroll"]

class ActionEvent:
    """Represents a single recorded user action."""
    def __init__(self, type: str, data: Dict[str, Any], timestamp: float):
        self.type = type
        self.data = data
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        }

class MacroRecorder:
    """Records user input events (mouse/keyboard)."""
    def __init__(self):
        self.events: List[ActionEvent] = []
        self.recording = False
        self.mouse_listener = None
        self.keyboard_listener = None
        self.stop_event = Event()

    def start(self):
        """Start recording asynchronously."""
        if self.recording:
            return
        self.recording = True
        self.events.clear()
        self.stop_event.clear()

        def on_move(x, y):
            if self.recording:
                self._add_event("mouse_move", {"x": x, "y": y})

        def on_click(x, y, button, pressed):
            if self.recording:
                action = "click" if pressed else "release"
                self._add_event(action, {"x": x, "y": y, "button": str(button), "pressed": pressed})

        def on_scroll(x, y, dx, dy):
            if self.recording:
                self._add_event("scroll", {"x": x, "y": y, "dx": dx, "dy": dy})

        def on_press(key):
            if self.recording:
                self._add_event("key_down", {"key": self._key_to_str(key)})

        def on_release(key):
            if self.recording:
                self._add_event("key_up", {"key": self._key_to_str(key)})
                if key == keyboard.Key.esc: # Optional: Stop on ESC
                    pass 

        self.mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        logger.info("MacroRecorder started.")

    def stop(self):
        """Stop recording."""
        if not self.recording:
            return
        self.recording = False
        self.stop_event.set()
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        logger.info(f"MacroRecorder stopped. Captured {len(self.events)} events.")

    def _add_event(self, type: str, data: Dict[str, Any]):
        self.events.append(ActionEvent(type, data, time.time()))

    def _key_to_str(self, key):
        try:
            return key.char
        except AttributeError:
            return str(key)

class SkillMetadata:
    def __init__(self, name: str, description: str, steps: List[Dict[str, Any]]) -> None:
        self.name: str = name
        self.description: str = description
        self.steps: List[Dict[str, Any]] = steps
        self.created_at: str = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "created_at": self.created_at
        }


class SkillLibrary:
    """
    Manages the storage and retrieval of macro skills with efficient disk I/O and error resilience.
    """
    def __init__(self, storage_path: str = "memory/skills") -> None:
        # Construct absolute path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.storage_path: str = os.path.join(base_dir, storage_path)
        self.skills: Dict[str, Dict[str, Any]] = {}
        
        # Ensure directory exists
        try:
            os.makedirs(self.storage_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create skill storage directory at {self.storage_path}: {e}")
            raise RuntimeError(f"Unable to initialize skill library storage: {e}")

        self._load_skills()

    def _load_skills(self) -> None:
        """Efficiently load all skills from disk with individual file fault tolerance."""
        if not os.path.exists(self.storage_path) or not os.path.isdir(self.storage_path):
            logger.warning(f"Skill storage path does not exist or is not a directory: {self.storage_path}")
            return

        supported_extensions = (".json",)
        loaded_count = 0

        for filename in os.listdir(self.storage_path):
            if not filename.endswith(supported_extensions):
                continue

            filepath = os.path.join(self.storage_path, filename)
            if not os.path.isfile(filepath):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data: Dict[str, Any] = json.load(f)
                
                name = data.get("name")
                if not name:
                    logger.warning(f"Invalid skill format in {filename}: missing 'name' field.")
                    continue

                self.skills[name] = data
                loaded_count += 1

            except PermissionError:
                logger.error(f"Permission denied when reading skill file: {filepath}")
            except json.JSONDecodeError as je:
                logger.error(f"Invalid JSON in skill file {filename}: {je}")
            except Exception as e:
                logger.error(f"Unexpected error loading skill from {filename}: {e}")

        logger.info(f"Loaded {loaded_count} skill(s) from {self.storage_path}")

    def save_skill(self, name: str, description: str, steps: List[Dict[str, Any]]) -> bool:
        """Save a new skill to the library with validation and safe write operations."""
        if not name.strip():
            logger.error("Cannot save skill: name is required.")
            return False

        name = name.strip()
        filename = f"{name.replace(' ', '_').lower()}.json"
        filepath = os.path.join(self.storage_path, filename)

        skill_data = {
            "name": name,
            "description": description.strip(),
            "steps": steps,
            "created_at": datetime.now().isoformat()
        }

        # Validate steps structure before saving
        if not isinstance(steps, list):
            logger.error("Steps must be a list of dictionaries.")
            return False

        try:
            # Write to temporary file first to prevent data loss
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(skill_data, f, indent=2, ensure_ascii=False)

            # Atomically replace old file
            if os.path.exists(temp_filepath):
                if os.path.exists(filepath):
                    os.replace(temp_filepath, filepath)  # Atomic on most systems
                else:
                    os.rename(temp_filepath, filepath)

            self.skills[name] = skill_data
            logger.info(f"Skill '{name}' saved successfully to {filepath}.")
            return True

        except PermissionError:
            logger.error(f"Permission denied writing to {filepath}")
        except OSError as oe:
            logger.error(f"OS error occurred while saving skill '{name}': {oe}")
        except Exception as e:
            logger.error(f"Failed to save skill '{name}': {e}")
            # Clean up temp file if it exists
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:  # noqa: E722
                    pass
        return False

    def get_skill(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a skill by name. Case-sensitive exact match."""
        if not name:
            return None
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """Return sorted list of available skill names."""
        return sorted(self.skills.keys())


class MacroPlayer:
    """
    Executes macro skills using the DesktopController with enhanced control and safety.
    """
    def __init__(self, controller: DesktopController, skill_library: SkillLibrary) -> None:
        self.controller: DesktopController = controller
        self.skill_library: SkillLibrary = skill_library
        self.is_playing: bool = False

    def play_macro(self, macro_name: str, speed: float = 1.0) -> bool:
        """
        Execute a named macro with configurable playback speed.

        Args:
            macro_name: Name of the macro to play.
            speed: Playback speed multiplier (e.g., 2.0 = twice as fast). Not yet fully implemented.

        Returns:
            True if execution completed successfully, False otherwise.
        """
        if not macro_name:
            logger.error("Macro name is required.")
            return False

        if speed <= 0:
            logger.error("Playback speed must be greater than zero.")
            return False

        skill = self.skill_library.get_skill(macro_name)
        if not skill:
            logger.error(f"Macro '{macro_name}' not found in library.")
            return False

        logger.info(f"▶️ Playing macro: {macro_name} (speed: {speed}x)")
        self.is_playing = True

        try:
            steps: List[Dict[str, Any]] = skill.get('steps', [])
            if not steps:
                logger.warning(f"No executable steps defined in macro '{macro_name}'.")
                return True  # Considered success – nothing to do

            for step in steps:
                if not self.is_playing:
                    logger.info("Macro playback stopped by user request.")
                    break

                try:
                    self._execute_step(step, speed)
                except Exception as step_exc:
                    logger.error(f"Error executing step {step} in macro '{macro_name}': {step_exc}")
                    # Optionally continue on error or fail-fast? We'll continue.
                    continue

            logger.info(f"✅ Macro '{macro_name}' execution complete.")
            return True

        except Exception as e:
            logger.exception(f"❌ Unexpected error during macro execution: {e}")
            return False

        finally:
            self.is_playing = False

    def _execute_step(self, step: Dict[str, Any], speed: float = 1.0) -> None:
        """Execute a single macro step with proper parameter handling."""
        action: Optional[str] = step.get('action')
        params: Dict[str, Any] = step.get('params', {})

        if not action:
            logger.warning("Step missing 'action' field; skipping.")
            return

        delay = max(0.01, 0.1 / speed)  # Minimum delay enforced

        try:
            if action == 'click':
                x = params.get('x')
                y = params.get('y')
                if x is not None and y is not None:
                    pyautogui.click(int(x), int(y))
                else:
                    logger.warning("Click action requires 'x' and 'y' coordinates.")

            elif action == 'type':
                text = params.get('text')
                if text:
                    pyautogui.write(str(text))
                else:
                    logger.warning("Type action has no text to input.")

            elif action == 'key':
                key = params.get('key')
                if key:
                    pyautogui.press(str(key))
                else:
                    logger.warning("Key action missing 'key' parameter.")

            elif action == 'wait':
                duration = float(params.get('duration', 1.0)) / speed
                if duration > 0:
                    time.sleep(duration)

            else:
                logger.warning(f"Unknown action type '{action}' in macro step.")

        except ValueError as ve:
            logger.error(f"Invalid parameter value in step {step}: {ve}")
        except Exception as e:
            logger.error(f"Failed to execute action '{action}': {e}")

        # Apply base inter-step delay unless overridden by explicit wait
        if action != 'wait':
            time.sleep(delay)

    def stop(self) -> None:
        """Stop current macro execution gracefully."""
        self.is_playing = False


class MacroSystem:
    """
    Unified facade for the Macro System (Library + Player + Controller).
    Provides high-level interface for learning, listing, and performing skills.
    """
    def __init__(self) -> None:
        try:
            self.controller: DesktopController = DesktopController()
            self.library: SkillLibrary = SkillLibrary()
            self.player: MacroPlayer = MacroPlayer(self.controller, self.library)
            logger.info("MacroSystem initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize MacroSystem: {e}")
            raise

    def learn_skill(self, name: str, description: str, steps: List[Dict[str, Any]]) -> bool:
        """
        Save a new skill into the library.

        Args:
            name: Unique identifier for the skill.
            description: Human-readable explanation.
            steps: Sequence of actions to perform.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not name or not name.strip():
            logger.error("Skill name cannot be empty.")
            return False
        return self.library.save_skill(name.strip(), description, steps)

    def perform_skill(self, name: str) -> bool:
        """
        Perform a previously learned skill.

        Args:
            name: Name of the skill to execute.

        Returns:
            True if execution succeeded, False otherwise.
        """
        if not name:
            logger.error("Cannot perform skill: name is required.")
            return False
        return self.player.play_macro(name)

    def list_skills(self) -> List[str]:
        """
        List all available skills in the library.

        Returns:
            Sorted list of skill names.
        """
        return self.library.list_skills()