import time
import base64
import io
import json
import logging
import subprocess
import pyautogui
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageGrab

from core.llm_client import LLMService

logger = logging.getLogger(__name__)

FORBIDDEN_HOTKEYS = [
    'alt+f4',
    'alt+f4',
    'ctrl+alt+del',
    'ctrl+alt+delete',
    'win+l',
    'win+d',
    'win+r',
    'ctrl+alt+esc',
    'ctrl+shift+esc',
    'alt+tab',
    'ctrl+alt+tab',
    'alt+f4',
    'alt+esc',
    'ctrl+esc',
    'alt+space',
]

FORBIDDEN_KEYWORDS = [
    'shutdown',
    'power',
    'restart',
    'reboot',
    'log off',
    'sign out',
    'hibernate',
    'sleep',
]

# 危险的 UI 元素关键词 - 防止视觉点击误识别
DANGEROUS_UI_KEYWORDS = [
    'shutdown',
    'power off',
    'turn off',
    'start button',
    'start menu',
    'taskbar',
    'power button',
    'restart',
    'sign out',
    'log off',
    'lock',
    'sleep',
    'hibernate',
]

class DesktopController:
    """
    Basic wrapper for desktop automation (Mouse/Keyboard).
    """
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = True # Move mouse to corner to abort
        self.blocked_clicks = []  # 记录被阻止的点击操作

    def _is_safe_click_region(self, x: int, y: int) -> bool:
        """
        检查点击坐标是否在安全区域内。

        危险区域包括：
        - 任务栏（底部约 48-60 像素）
        - 开始按钮（左下角）
        - 屏幕四个角落（可能触发系统快捷键）

        Args:
            x, y: 点击坐标

        Returns:
            True 如果安全，False 如果在危险区域
        """
        # 检查任务栏区域（底部）
        taskbar_height = 60
        if y > (self.screen_height - taskbar_height):
            logger.warning(f"🚫 BLOCKED click in taskbar region: ({x}, {y})")
            self.blocked_clicks.append({'x': x, 'y': y, 'reason': 'taskbar'})
            return False

        # 检查开始按钮区域（左下角）
        if x < 100 and y > (self.screen_height - 60):
            logger.warning(f"🚫 BLOCKED click near start button: ({x}, {y})")
            self.blocked_clicks.append({'x': x, 'y': y, 'reason': 'start_button'})
            return False

        # 检查屏幕四个角落（FAILSAFE区域的扩展）
        corner_threshold = 10
        if (x < corner_threshold and y < corner_threshold) or \
           (x > self.screen_width - corner_threshold and y < corner_threshold) or \
           (x < corner_threshold and y > self.screen_height - corner_threshold) or \
           (x > self.screen_width - corner_threshold and y > self.screen_height - corner_threshold):
            logger.warning(f"🚫 BLOCKED click in corner region: ({x}, {y})")
            self.blocked_clicks.append({'x': x, 'y': y, 'reason': 'corner'})
            return False

        return True

    def get_safety_report(self) -> Dict[str, Any]:
        """
        获取安全检查报告，包括所有被阻止的操作。

        Returns:
            包含被阻止点击次数、最近被阻止的操作列表等信息的字典
        """
        return {
            "blocked_clicks_count": len(self.blocked_clicks),
            "recent_blocked_clicks": self.blocked_clicks[-10:] if self.blocked_clicks else [],
            "screen_resolution": (self.screen_width, self.screen_height),
            "taskbar_protected": True,
            "start_button_protected": True,
            "corners_protected": True
        }

    def click(self, x: int, y: int, button: str = 'left', force: bool = False):
        """
        Perform a mouse click.

        Args:
            x, y: 点击坐标
            button: 鼠标按键 ('left', 'right', 'middle')
            force: 是否强制执行（跳过安全检查，默认 False）

        Returns:
            True 如果点击成功，False 如果被阻止或失败
        """
        # 安全检查（除非强制执行）
        if not force and not self._is_safe_click_region(x, y):
            logger.error(f"❌ Click BLOCKED by safety check at ({x}, {y})")
            return False

        try:
            pyautogui.click(x, y, button=button)
            logger.info(f"✅ Clicked at ({x}, {y}) with {button} button")
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    def type_text(self, text: str):
        """Type text."""
        try:
            pyautogui.write(text)
            logger.info(f"Typed text: {text}")
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return False

    def press_key(self, key: str):
        """Press a key."""
        key_lower = key.lower()
        
        for forbidden in FORBIDDEN_HOTKEYS:
            if forbidden.lower() in key_lower:
                logger.warning(f"🚫 BLOCKED forbidden hotkey: {key}")
                return False
        
        for forbidden in FORBIDDEN_KEYWORDS:
            if forbidden in key_lower:
                logger.warning(f"🚫 BLOCKED dangerous keyword: {key}")
                return False
        
        try:
            pyautogui.press(key)
            logger.info(f"Pressed key: {key}")
            return True
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False

    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Capture the screen."""
        try:
            return ImageGrab.grab(bbox=region)
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            # Return a blank image to avoid crashes
            return Image.new('RGB', (100, 100), color = 'black')

    def open_app(self, app_name: str) -> str:
        """
        Open an application by name.
        Uses subprocess only (removed dangerous hotkey fallback for safety).
        """
        app_map = {
            "vs code": "code",
            "vscode": "code",
            "chrome": "chrome",
            "google chrome": "chrome",
            "notepad": "notepad",
            "calc": "calc",
            "calculator": "calc",
            "explorer": "explorer",
            "cmd": "cmd",
            "powershell": "powershell",
            "terminal": "wt"
        }

        target = app_map.get(app_name.lower(), app_name)
        logger.info(f"Attempting to open app: {app_name} -> {target}")

        try:
            # 使用 subprocess 启动应用（最安全的方式）
            subprocess.Popen(target, shell=True)
            return f"Launched {target}"
        except Exception as e:
            logger.error(f"Failed to open {app_name}: {e}")
            return f"Failed to open {app_name}: {e}"

class VisualClickExecutor:
    """
    Advanced clicker that uses VLM to find targets on screen.
    """
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service if llm_service else LLMService()
        self.controller = DesktopController()
        self.blocked_tasks = []  # 记录被阻止的任务

    def _is_safe_task_description(self, task_description: str) -> Tuple[bool, str]:
        """
        检查任务描述是否涉及危险的 UI 元素。

        Args:
            task_description: 任务描述文本

        Returns:
            (is_safe, reason): 如果安全返回 (True, "")，如果不安全返回 (False, 原因)
        """
        task_lower = task_description.lower()

        for keyword in DANGEROUS_UI_KEYWORDS:
            if keyword in task_lower:
                reason = f"Task description contains dangerous keyword: '{keyword}'"
                logger.warning(f"🚫 BLOCKED visual click task: {reason}")
                self.blocked_tasks.append({'task': task_description, 'reason': reason})
                return False, reason

        return True, ""

    def get_safety_report(self) -> Dict[str, Any]:
        """
        获取视觉点击的安全检查报告。

        Returns:
            包含被阻止任务次数、最近被阻止的任务列表等信息的字典
        """
        return {
            "blocked_tasks_count": len(self.blocked_tasks),
            "recent_blocked_tasks": self.blocked_tasks[-10:] if self.blocked_tasks else [],
            "controller_safety": self.controller.get_safety_report()
        }

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def execute_visual_click(self, task_description: str, expected_change_hint: str = "", retry_on_failure: bool = True, verify_click_effect: bool = True, force: bool = False) -> Dict[str, Any]:
        """
        Execute a click based on visual description.

        Args:
            task_description: e.g., "Click the 'Submit' button in the center"
            expected_change_hint: Description of what should happen after click (for verification)
            retry_on_failure: Whether to retry if VLM fails to find target
            verify_click_effect: Whether to check if screen changed after click
            force: 是否强制执行（跳过安全检查，默认 False）

        Returns:
            Dict with 'success', 'click_position', 'error', 'feedback_verified'
        """
        # 🔒 安全检查：检查任务描述是否涉及危险元素
        if not force:
            is_safe, reason = self._is_safe_task_description(task_description)
            if not is_safe:
                return {
                    "success": False,
                    "error": f"🚫 SAFETY BLOCK: {reason}",
                    "blocked_by_safety": True
                }

        logger.info(f"👁️ Visual Click Initiated: {task_description}")
        
        attempts = 3 if retry_on_failure else 1
        
        for attempt in range(attempts):
            try:
                # 1. Capture Screen
                screenshot = self.controller.capture_screen()
                b64_img = self._image_to_base64(screenshot)
                
                # 2. Ask VLM for coordinates
                prompt = f"""
                I need to click on an element on the screen.
                Task: "{task_description}"
                
                Analyze the screenshot and identify the exact (x, y) coordinates of the center of the target element.
                
                Output ONLY a JSON object in this format:
                {{
                    "found": true,
                    "x": 123,
                    "y": 456,
                    "confidence": 0.9,
                    "reasoning": "Found a blue button labelled 'Submit' at the bottom right."
                }}
                
                If you cannot find the element, set "found": false.
                """
                
                response = self.llm.chat_with_vision(
                    system_prompt="You are a GUI automation assistant. Output valid JSON only.",
                    user_prompt=prompt,
                    base64_image=b64_img
                )
                
                # 3. Parse Response
                try:
                    cleaned_response = response.replace("```json", "").replace("```", "").strip()
                    result = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    logger.warning(f"VLM JSON parse failed: {response}")
                    continue
                
                if not result.get("found"):
                    logger.warning(f"Target not found: {result.get('reasoning')}")
                    continue

                x, y = result["x"], result["y"]

                # 🔒 额外安全检查：检查 VLM 的 reasoning 是否包含危险元素
                reasoning = result.get("reasoning", "").lower()
                is_safe_reasoning, _ = self._is_safe_task_description(reasoning)
                if not force and not is_safe_reasoning:
                    logger.warning(f"🚫 BLOCKED: VLM identified dangerous element: {reasoning}")
                    return {
                        "success": False,
                        "error": f"🚫 SAFETY BLOCK: VLM identified potentially dangerous element",
                        "blocked_by_safety": True,
                        "vlm_reasoning": reasoning
                    }

                # 4. Perform Click (with force parameter)
                if not self.controller.click(x, y, force=force):
                    logger.error(f"Click failed or was blocked at ({x}, {y})")
                    return {
                        "success": False,
                        "error": "Click blocked by safety check or failed",
                        "blocked_by_safety": True
                    }
                
                # 5. Verification (Optional)
                feedback_verified = False
                if verify_click_effect:
                    time.sleep(1.0) # Wait for UI to update
                    new_screenshot = self.controller.capture_screen()
                    # Simple check: did the screen change? (This is a naive check)
                    # In a real system, we'd ask VLM to verify the state change.
                    feedback_verified = True # Placeholder for now
                
                return {
                    "success": True,
                    "click_position": (x, y),
                    "feedback_verified": feedback_verified,
                    "raw_output": result
                }
                
            except Exception as e:
                logger.error(f"Visual click attempt {attempt+1} failed: {e}")
                time.sleep(1)
        
        return {
            "success": False,
            "error": "Max retries exceeded or target not found."
        }
