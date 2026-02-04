import os
import time
import json
import logging
import threading
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import pyautogui
from io import BytesIO

from core.macro_system import MacroRecorder, ActionEvent
from core.llm_client import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PassiveObserver")

class ObservationSession:
    def __init__(self, output_dir: str):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, self.session_id)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.events: List[ActionEvent] = []
        self.screenshots: List[Dict[str, Any]] = [] # {timestamp, filename, context}
        self.start_time = time.time()
        self.end_time = 0.0

    def add_event(self, event: ActionEvent):
        self.events.append(event)

    def add_screenshot(self, filename: str, timestamp: float, context: str = "periodic"):
        self.screenshots.append({
            "timestamp": timestamp,
            "filename": filename,
            "context": context
        })

    def save_manifest(self):
        manifest = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "event_count": len(self.events),
            "screenshot_count": len(self.screenshots),
            "events": [e.to_dict() for e in self.events],
            "screenshots": self.screenshots
        }
        with open(os.path.join(self.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Session manifest saved to {self.output_dir}")

class PassiveObserver:
    """
    The 'Silent Learner'. 
    Records user actions and screen states to build a dataset for meaning extraction.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = os.path.abspath(storage_dir)
        else:
            # Default to PROJECT_ROOT/memory/observations
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_dir = os.path.join(project_root, "memory", "observations")
            
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        self.recorder = MacroRecorder()
        self.session: Optional[ObservationSession] = None
        self.is_observing = False
        self.screenshot_thread = None
        self.llm_service = LLMService()

    def start_observation(self):
        """Starts the observation loop (events + screenshots)."""
        if self.is_observing:
            logger.warning("Already observing.")
            return

        # Check disk space (ensure at least 500MB free)
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.storage_dir)
            if free < 500 * 1024 * 1024:
                logger.error("Insufficient disk space for observation.")
                return
        except:
            pass

        logger.info("Starting Passive Observation...")
        self.session = ObservationSession(self.storage_dir)
        self.is_observing = True

        # Start Event Recorder (Non-blocking)
        self.recorder.start()
        
        self.stop_event = threading.Event()
        self.screenshot_thread = threading.Thread(target=self._monitor_screen_loop)
        self.screenshot_thread.start()

    def stop_observation(self, cleanup_if_empty: bool = True) -> str:
        """
        Stops observation and returns the session path.
        Args:
            cleanup_if_empty: If True, deletes the session if no significant events occurred.
        """
        if not self.is_observing:
            return ""

        logger.info("Stopping Observation...")
        self.is_observing = False
        self.stop_event.set()
        
        # Stop Recorder
        self.recorder.stop()
        
        if self.screenshot_thread:
            self.screenshot_thread.join()

        self.session.end_time = time.time()
        
        # Merge events from recorder
        for evt in self.recorder.events:
            self.session.add_event(evt)
        
        # Cleanup logic: If session is too short or empty, delete it
        if cleanup_if_empty:
            duration = self.session.end_time - self.session.start_time
            if duration < 2.0 or len(self.session.events) < 2:
                logger.info(f"Session too short ({duration:.1f}s) or empty. Cleaning up...")
                import shutil
                shutil.rmtree(self.session.output_dir)
                return ""
            
        self.session.save_manifest()
        return self.session.output_dir

    def _monitor_screen_loop(self):
        """Captures screenshots periodically (Max 50 per session)."""
        logger.info("Screenshot monitor started.")
        count = 0
        max_screenshots = 50
        
        while self.is_observing and not self.stop_event.is_set():
            if count >= max_screenshots:
                time.sleep(1)
                continue
                
            try:
                timestamp = time.time()
                filename = f"screen_{int(timestamp * 1000)}.jpg"
                filepath = os.path.join(self.session.output_dir, filename)
                
                # Capture
                img = pyautogui.screenshot()
                # Resize to save space (1280x720 is good balance)
                img.thumbnail((1280, 720)) 
                img.save(filepath, "JPEG", quality=70) # Reduced quality for space
                
                self.session.add_screenshot(filename, timestamp)
                count += 1
                
                # Sleep interval (2 seconds)
                time.sleep(2)
            except Exception as e:
                logger.error(f"Screenshot failed: {e}")
                time.sleep(2)

    def analyze_session(self, session_path: str) -> str:
        """
        Uses VLM to extract meaning from the observed session.
        Strategies:
        1. Multi-frame analysis (Start -> Middle -> End)
        2. Correlate events with screen changes.
        """
        manifest_path = os.path.join(session_path, "manifest.json")
        if not os.path.exists(manifest_path):
            return "Error: Manifest not found."

        with open(manifest_path, 'r') as f:
            data = json.load(f)

        screenshots = data.get("screenshots", [])
        events = data.get("events", [])
        
        if not screenshots:
            return "Analysis Failed: No screenshots captured."

        # Strategy: Pick First, Middle, Last screenshot
        selected_indices = [0, len(screenshots)//2, -1]
        selected_indices = sorted(list(set(selected_indices))) # Remove duplicates
        
        images_base64 = []
        for idx in selected_indices:
            info = screenshots[idx]
            path = os.path.join(session_path, info['filename'])
            if os.path.exists(path):
                with open(path, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode('utf-8')
                    images_base64.append((info['timestamp'], b64))

        # Summarize Events (Textual)
        event_summary = self._summarize_events(events)

        # Construct Prompt
        # Since our current LLMService.chat_with_vision takes ONE image, 
        # we will use the LAST image (Result) as the visual context, 
        # and describe the initial state in text (or use the first image if that's more relevant).
        # ideally we stitch them, but let's stick to single image for simplicity + text context.
        
        # We'll use the LAST image to show "What was achieved".
        final_timestamp, final_b64 = images_base64[-1]
        
        prompt = f"""
        You are an AGI Observer. I have recorded a user's desktop session.
        
        **Event Log (Key Actions):**
        {event_summary}
        
        **Final Screen State:**
        (See attached image)
        
        **Task:**
        1. Analyze the event log and the final screen.
        2. Infer the user's INTENT (what were they trying to do?).
        3. Summarize the process in high-level steps.
        4. Assign a potential skill name (e.g., "send_email", "open_youtube").
        
        Output format:
        **Intent:** ...
        **Process:** ...
        **Suggested Skill Name:** ...
        """

        logger.info("Sending session data to VLM for analysis...")
        response = self.llm_service.chat_with_vision(
            system_prompt="You are an expert at understanding desktop automation and user behavior.",
            user_prompt=prompt,
            base64_image=final_b64
        )
        
        return response

    def _summarize_events(self, events: List[Dict]) -> str:
        """Compresses raw events into a readable log."""
        summary = []
        start_time = events[0]['timestamp'] if events else 0
        
        for e in events:
            t = e['timestamp'] - start_time
            etype = e['event_type']
            data = e['data']
            
            if etype == 'click':
                summary.append(f"[{t:.1f}s] Click at ({data['x']}, {data['y']}) button={data['button']}")
            elif etype == 'key_down':
                key = data['key'].replace("Key.", "")
                # Ignore frequent modifier holds to reduce noise
                summary.append(f"[{t:.1f}s] Key Press: {key}")
            elif etype == 'type':
                summary.append(f"[{t:.1f}s] Typed: '{data['text']}'")
                
        # Limit log length
        if len(summary) > 50:
            return "\n".join(summary[:20]) + f"\n... ({len(summary)-40} more events) ...\n" + "\n".join(summary[-20:])
        return "\n".join(summary)

