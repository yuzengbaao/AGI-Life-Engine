
import os
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any

# Ensure we can import the connector
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "mapping_tools"))

try:
    from agi_autocad_connector import AutoCADConnector
except ImportError:
    AutoCADConnector = None

from core.vision_observer import VisionObserver

logger = logging.getLogger("CAD_Observer")

class CADObserver:
    """
    AGI 'Eyes & Ears' for AutoCAD.
    Watches the user's actions via log files AND vision (VLM).
    Integrates with Global Workspace for cognitive processing.
    """
    def __init__(self):
        self.connector = None
        self.log_path = None
        self.last_position = 0
        self.is_watching = False
        self.session_buffer = [] # Store raw commands for this session
        self.vision = VisionObserver()
        
    def scan_instances(self):
        """
        Scan for running AutoCAD instances.
        Returns a list of dictionaries with instance details.
        """
        if not AutoCADConnector:
            return []
        return AutoCADConnector.scan_running_instances()

    def connect(self, prog_id=None):
        if not AutoCADConnector:
            logger.error("AutoCADConnector module not found.")
            return False
            
        try:
            self.connector = AutoCADConnector(specific_prog_id=prog_id)
            if self.connector.is_connected():
                # Report which instance we hooked into
                print(f"   🎯 TARGET ACQUIRED: {self.connector.caption}")
                
                if not self.connector.doc:
                    print("   ⚠️ Connected to AutoCAD, but NO DRAWING IS OPEN.")
                    print("   👉 Please open a drawing file to enable observation.")
                    return False
                    
                print(f"   📂 WATCHING DOCUMENT: {self.connector.doc.Name}")
                
                self.log_path = self.connector.ensure_logging_enabled()
                if self.log_path and os.path.exists(self.log_path):
                    # Move pointer to end of file (don't read old history)
                    self.last_position = os.path.getsize(self.log_path)
                    return True
        except Exception as e:
            logger.error(f"Failed to connect observer: {e}")
        return False

    async def observe_cycle_enriched(self) -> List[Dict[str, Any]]:
        """
        Read new lines from log AND capture screen context if significant action occurred.
        Returns a list of 'Enriched Action' dictionaries.
        """
        raw_actions = self._read_log()
        enriched_actions = []
        
        for action_text in raw_actions:
            # 1. Basic Action
            action_data = {
                "text": action_text,
                "timestamp": time.time(),
                "type": "command_log",
                "vlm_context": None
            }
            
            # 2. If action is significant (draw, modify), trigger Vision
            if self._is_significant_action(action_text):
                print(f"      👀 Visualizing action: {action_text}...")
                # Capture screen context (what does it look like now?)
                vlm_insight = self.vision.analyze_screen(
                    prompt=f"User just executed '{action_text}' in AutoCAD. Describe the geometric change on screen. What did they draw or modify?"
                )
                if "Cooldown" not in vlm_insight and "Error" not in vlm_insight:
                    action_data["vlm_context"] = vlm_insight
            
            enriched_actions.append(action_data)
            
        return enriched_actions

    def _read_log(self) -> List[str]:
        """Low-level log reader"""
        if not self.log_path or not os.path.exists(self.log_path):
            return []
            
        new_actions = []
        try:
            current_size = os.path.getsize(self.log_path)
            if current_size < self.last_position:
                self.last_position = 0
                
            if current_size > self.last_position:
                with open(self.log_path, 'r', encoding='utf-16-le', errors='ignore') as f:
                    f.seek(self.last_position)
                    content = f.read()
                    self.last_position = f.tell()
                    
                    lines = content.splitlines()
                    for line in lines:
                        clean_line = line.strip()
                        # Simple heuristic to clean AutoCAD prompt garbage
                        if clean_line and not clean_line.startswith("Command:") and len(clean_line) > 1:
                            new_actions.append(clean_line)
                            self.session_buffer.append(clean_line)
        except Exception as e:
            logger.error(f"Error reading CAD log: {e}")
        return new_actions

    def _is_significant_action(self, action: str) -> bool:
        """Filter to decide when to burn VLM tokens"""
        keywords = ["LINE", "PLINE", "CIRCLE", "ARC", "TRIM", "EXTEND", "COPY", "MOVE", "OFFSET", "HATCH", "TEXT", "MTEXT"]
        act_upper = action.upper()
        # Check if it matches a command (often logs are just coordinates, we need to infer context)
        # Actually, AutoCAD logs often just show coordinates "100,200".
        # We might need to look at the *previous* command in buffer.
        return any(k in act_upper for k in keywords)

    def observe_cycle(self) -> List[str]:
        """Legacy wrapper for compatibility"""
        return self._read_log()

    def _interpret_line(self, line: str) -> Optional[str]:
        """Legacy helper"""
        return line

    def _interpret_line(self, line: str) -> Optional[str]:
        """
        Translate raw CAD log lines into semantic actions.
        """
        # Basic filtering
        if len(line) < 2: return None
        if "Command:" in line: return None
        
        # Identify common commands
        upper_line = line.upper()
        if upper_line in ["L", "LINE"]: return "Action: Start Drawing Line"
        if upper_line in ["C", "CIRCLE"]: return "Action: Start Drawing Circle"
        if upper_line in ["PL", "PLINE"]: return "Action: Start Polyline"
        if upper_line in ["E", "ERASE"]: return "Action: Erase Object"
        if upper_line in ["M", "MOVE"]: return "Action: Move Object"
        if upper_line in ["CO", "COPY"]: return "Action: Copy Object"
        
        # Identify coordinates (roughly)
        if "," in line and any(c.isdigit() for c in line):
            return f"Input: Coordinate/Value [{line}]"
            
        return f"Command/Input: {line}"

    def summarize_session(self) -> str:
        """
        Summarize what happened in this observation session.
        """
        if not self.session_buffer:
            return "No actions observed."
        
        summary = "User AutoCAD Session Log:\n"
        for i, action in enumerate(self.session_buffer):
            summary += f"{i+1}. {action}\n"
        return summary
