import os
import subprocess
import logging
import platform
import ast
from typing import Optional, Tuple, Any, List
from core.perception.code_indexer import ProjectIndexer
from core.skill_manager import SkillManager
try:
    from core.memory_bridge import MemoryBridge
except ImportError:
    MemoryBridge = None

logger = logging.getLogger("SystemTools")

class SystemTools:
    """
    Provides the 'Hands' for the AGI to interact with the file system and terminal.
    Implements:
    - Programming Ability (Run Scripts)
    - Terminal Control (Run Commands)
    - Text Expression (Write Files)
    - System Control (List/Read Files)
    - Perception (Code Indexing, Tree View, Knowledge Retrieval)
    """
    
    def __init__(self, work_dir: str = None):
        self.work_dir = os.path.abspath(work_dir or os.getcwd())
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        # Enforce project-only storage for security
        self.log_dir = os.path.join(self.work_dir, "WorkLog")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.indexer = None # Lazy load
        self.memory_bridge = None # Lazy load
        self.skill_manager = SkillManager()
        self.biological_memory = None

    def _is_within_base(self, base: str, target: str) -> bool:
        try:
            base_abs = os.path.abspath(base)
            target_abs = os.path.abspath(target)
            # Allow any path within the TRAE_PROJECT directory to be accessed
            # This fixes the "TICK36" permission error where agents try to access project-level files
            trae_project_root = os.path.abspath(os.path.join(self.project_root, ".."))
            if os.path.commonpath([trae_project_root, target_abs]) == trae_project_root:
                return True
            return os.path.commonpath([base_abs, target_abs]) == base_abs
        except Exception:
            return False

    def _resolve_existing_path(self, raw_path: str) -> Optional[str]:
        """
        Resolve a raw path to an existing file path within allowed directories.
        
        Handles:
        - Absolute paths with drive letter (C:\\...)
        - Absolute paths without drive letter (\\results\\...) - Windows style
        - Relative paths (results/...)
        - Forward and backward slashes
        """
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = raw_path.strip()
        
        # Normalize slashes for cross-platform compatibility
        candidate = candidate.replace('/', os.sep).replace('\\', os.sep)
        
        # Handle Windows-style paths that start with backslash but have no drive letter
        # On Windows, os.path.isabs() returns True for paths like "\data\file",
        # but these don't have a drive letter, so we should treat them as relative to work_dir
        has_drive_letter = len(candidate) >= 2 and candidate[1] == ':'
        starts_with_sep = candidate.startswith(os.sep)
        
        if starts_with_sep and not has_drive_letter:
            # Remove leading separator(s) and treat as relative
            candidate = candidate.lstrip(os.sep)
        
        # Now check if it's truly absolute (has drive letter on Windows or starts with / on Unix)
        # After stripping, re-check isabs
        if os.path.isabs(candidate):
            abs_path = os.path.abspath(candidate)
            if self._is_within_base(self.work_dir, abs_path) or self._is_within_base(self.project_root, abs_path):
                return abs_path if os.path.exists(abs_path) else None
            return None
        
        # Try relative to work_dir first
        by_work_dir = os.path.abspath(os.path.join(self.work_dir, candidate))
        if os.path.exists(by_work_dir):
            return by_work_dir
        
        # Try relative to project_root
        by_project_root = os.path.abspath(os.path.join(self.project_root, candidate))
        if os.path.exists(by_project_root):
            return by_project_root
        
        # Log the failed path resolution for debugging
        logger.debug(f"Path resolution failed for: {raw_path} (tried: {by_work_dir}, {by_project_root})")
        return None

    def log_error(self, message: str, error: Exception, component: str = 'unknown'):
        """Log an error with context."""
        msg = f"[{component}] ERROR: {message} | {str(error)}"
        logger.error(msg)

    def write_file(self, filename: str, content: str) -> str:
        """Create or overwrite a file with content."""
        try:
            # 1. Handle absolute paths: Relaxed check
            if os.path.isabs(filename):
                safe_path = os.path.abspath(filename)
                if not self._is_within_base(self.project_root, safe_path):
                     # If still outside, try to redirect but allow if it's in the broader project
                     if "log" in filename.lower() or "summary" in filename.lower():
                         base_name = os.path.basename(filename)
                         safe_path = os.path.join(self.log_dir, base_name)
                         logger.warning(f"Redirecting external write attempt to project area: {safe_path}")
                     else:
                         # Final check: Is it in the workspace?
                         if not safe_path.startswith(os.path.abspath(self.work_dir)):
                             return f"Error: Access denied. Security Policy requires files to be saved within {self.project_root}"
            else:
                # 2. Relative path -> Default to project dir
                safe_path = os.path.abspath(os.path.join(self.work_dir, filename))
                if not safe_path.startswith(os.path.abspath(self.work_dir)):
                     return f"Error: Access denied. Cannot write outside {self.work_dir}"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Success: File '{safe_path}' written ({len(content)} chars)."
        except Exception as e:
            logger.error(f"Write failed: {e}")
            return f"Error writing file: {e}"

    def read_file(self, filename: str, limit: int = 2000) -> str:
        """Read content from a file."""
        try:
            safe_path = self._resolve_existing_path(filename)
            if not safe_path:
                return f"Error: File '{filename}' not found or access denied."
                
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(limit)
                if len(content) == limit:
                    content += "\n...[Truncated]"
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def get_file_size(self, filename: str) -> str:
        """Get the size of a file in bytes."""
        try:
            safe_path = self._resolve_existing_path(filename)
            if not safe_path:
                return "Error: File not found or access denied."
            return str(os.path.getsize(safe_path))
        except Exception as e:
            return f"Error getting file size: {e}"

    def save_interaction_to_memory(self, context: str, action: str, outcome: float = 1.0, details: Any = None) -> str:
        """
        Save an interaction to the experience memory system.
        
        Args:
            context: The context of the interaction (e.g., "SemanticLoopClosure")
            action: The action taken (e.g., "GenerateReceipt+Verify")
            outcome: Success score 0.0-1.0 (default 1.0)
            details: Additional metadata dict
        
        Returns:
            Success or error message
        """
        try:
            if MemoryBridge is None:
                return "Error: MemoryBridge not available."
            if self.memory_bridge is None:
                self.memory_bridge = MemoryBridge()
            if not self.memory_bridge or not getattr(self.memory_bridge, "memory", None):
                return "Error: Memory system not initialized."

            import time

            if details is None:
                details = {}
            if isinstance(details, dict) and "timestamp" not in details:
                details = {**details, "timestamp": time.time()}

            self.memory_bridge.memory.add_experience(
                context=str(context),
                action=str(action),
                outcome=float(outcome),
                details=details if isinstance(details, dict) else {"details": str(details), "timestamp": time.time()}
            )
            return "Success: Interaction archived to ExperienceMemory."
        except Exception as e:
            return f"Error archiving interaction: {e}"
            
    def list_files(self, directory: str = ".") -> str:
        """List files in a directory."""
        try:
            safe_path = self._resolve_existing_path(directory) if directory not in (".", "", None) else None
            if safe_path is None:
                safe_path = os.path.abspath(os.path.join(self.work_dir, directory or "."))
                if not os.path.exists(safe_path):
                    candidate = os.path.abspath(os.path.join(self.project_root, directory or "."))
                    if os.path.exists(candidate):
                        safe_path = candidate
                    else:
                        return f"Error: Directory '{directory}' not found or access denied."
            if not os.path.exists(safe_path):
                return f"Error: Directory '{directory}' not found or access denied."
            if not os.path.isdir(safe_path):
                return f"Error: Path '{directory}' is not a directory."
                
            items = os.listdir(safe_path)
            # Filter hidden
            items = [i for i in items if not i.startswith('.')]
            return f"Files in {directory}: {', '.join(items[:50])}"
        except Exception as e:
            return f"Error listing files: {e}"

    def inspect_project_structure(self, directory: str = ".", depth: int = 2) -> str:
        try:
             output = []
             root_level = directory.count(os.sep)
             for root, dirs, files in os.walk(directory):
                 level = root.count(os.sep) - root_level
                 if level > depth:
                     continue
                 indent = " " * 4 * level
                 output.append(f"{indent}{os.path.basename(root)}/")
                 subindent = " " * 4 * (level + 1)
                 for f in files:
                     output.append(f"{subindent}{f}")
             return "\n".join(output[:100])
        except Exception as e:
            return str(e)

    def run_command(self, command: str) -> str:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        except Exception as e:
            return f"Error running command: {e}"

    def run_command_with_retry(self, command: str, llm_client=None) -> str:
        return self.run_command(command)

    def run_python_script(self, script_name: str) -> str:
        return self.run_command(f"python {script_name}")

    def inspect_code(self, path: str, mode: str = "summary") -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            output = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    output.append(f"Class: {node.name}")
                    for m in [n.name for n in node.body if isinstance(n, ast.FunctionDef)]:
                        output.append(f"  Method: {m}")
                elif isinstance(node, ast.FunctionDef):
                    output.append(f"Function: {node.name}")
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if mode == 'imports':
                        output.append(f"  📎 Import: {ast.dump(node)}")
            return "\n".join(output)
        except Exception as e:
            return str(e)

    def analyze_traceback(self, traceback_str: str) -> str:
        """Parse Python traceback string into structured error diagnosis."""
        try:
            import re
            import json
            lines = traceback_str.strip().split('\n')
            error_message = lines[-1] if lines else "Unknown Error"
            diagnosis = {
                "error_type": error_message.split(':')[0] if ':' in error_message else "RuntimeError",
                "message": error_message,
                "suggestion": "Check the code at the specified location."
            }
            return json.dumps(diagnosis, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to analyze traceback: {e}", "raw": traceback_str[:200]})

    def web_search(self, query: str, engine: str = "bing") -> str:
        return "Web search simulation: No results (Offline)"

    def execute_cognitive_skill(self, skill_name: str, **kwargs) -> str:
        """
        Executes a dynamically acquired cognitive skill.
        """
        try:
            if not self.skill_manager:
                return "Error: Skill Manager not initialized."
            result = str(self.skill_manager.execute_skill(skill_name, **kwargs))
            try:
                if self.biological_memory is not None:
                    payload = {
                        "tool": "execute_cognitive_skill",
                        "skill": skill_name,
                        "args": kwargs,
                        "result": result,
                    }
                    self.biological_memory.record_online(
                        [
                            {
                                "id": f"skill_{skill_name}_{int(time.time() * 1000)}",
                                "content": json.dumps(payload, ensure_ascii=False),
                                "source": "system_tools",
                                "type": "skill_call",
                                "tool": "execute_cognitive_skill",
                                "skill": skill_name,
                            }
                        ],
                        connect_sequence=True,
                        seq_port="exec",
                        save=True,
                    )
            except Exception:
                pass
            return result
        except Exception as e:
            return f"Error executing cognitive skill {skill_name}: {e}"
