import base64
import io
import time
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from PIL import ImageGrab, Image
from core.llm_client import LLMService

logger = logging.getLogger("VisionObserver")

@dataclass
class MatchResult:
    """
    Standardized result for visual matching operations.
    """
    x: int
    y: int
    confidence: float
    label: Optional[str] = None
    box: Optional[Tuple[int, int, int, int]] = None  # (left, top, width, height)

class VisionObserver:
    """
    Standardized result for visual matching operations.
    """
    x: int
    y: int
    confidence: float
    label: Optional[str] = None
    box: Optional[Tuple[int, int, int, int]] = None  # (left, top, width, height)

class VisionObserver:
    """
    [New Module] Multimodal Screen Observer
    Uses VLM (Vision Language Model) to understand the screen content.
    """
    def __init__(self):
        self.llm = LLMService()
        self.last_analysis_time = 0
        self.min_interval = 10 # Seconds between expensive VLM calls

    def capture_screen_base64(self, max_size=(1024, 1024)) -> str:
        """Capture screen and convert to base64 string"""
        try:
            screen = ImageGrab.grab()
            
            # Resize if too big to save tokens/bandwidth
            if screen.width > max_size[0] or screen.height > max_size[1]:
                screen.thumbnail(max_size, Image.Resampling.LANCZOS)
                
            buffer = io.BytesIO()
            screen.save(buffer, format="JPEG", quality=70)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None

    def analyze_screen(self, prompt: str = "What is on the user's screen? Briefly summarize.") -> str:
        """
        Capture screen and ask VLM to analyze it.
        """
        if time.time() - self.last_analysis_time < self.min_interval:
            return "(Vision Observer Cooldown)"
            
        b64_img = self.capture_screen_base64()
        if not b64_img:
            return "Error: Could not capture screen."
            
        logger.info("👀 Vision Observer: Sending screen to VLM...")
        
        analysis = self.llm.chat_with_vision(
            system_prompt="You are an AI Assistant with eyes. You are looking at the user's computer screen.",
            user_prompt=prompt,
            base64_image=b64_img
        )
        
        self.last_analysis_time = time.time()
        return analysis

    def get_element_coordinates(self, element_name: str) -> Optional[Tuple[int, int]]:
        """
        Ask VLM for the coordinates of a specific UI element on the screen.
        Returns: (x, y) or None if not found/error.
        """
        try:
            # 1. Capture full screen
            screen = ImageGrab.grab()
            original_width, original_height = screen.size
            
            # 2. Resize for VLM (preserve aspect ratio)
            max_size = (1024, 1024)
            screen.thumbnail(max_size, Image.Resampling.LANCZOS)
            resized_width, resized_height = screen.size
            
            # 3. Prepare Base64
            buffer = io.BytesIO()
            screen.save(buffer, format="JPEG", quality=70)
            b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.info(f"👀 Vision Observer: Locating '{element_name}' on screen...")
            
            # 4. Prompt VLM for JSON coordinates
            prompt = f"""
            Task: Locate the UI element '{element_name}' on this screenshot.
            Output: Return a JSON object with the CENTER coordinates [x, y] of the element.
            The coordinates should be based on the image size provided: {resized_width}x{resized_height}.
            
            Example Format:
            {{"x": 150, "y": 300}}
            
            If you cannot see the element clearly, return null.
            Do not explain. Only JSON.
            """
            
            response = self.llm.chat_with_vision(
                system_prompt="You are a UI Automation Agent. You locate elements precisely.",
                user_prompt=prompt,
                base64_image=b64_img
            )
            
            # 5. Parse JSON
            import json
            import re
            
            # Extract JSON block
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
            
            data = json.loads(json_str)
            if not data or "x" not in data or "y" not in data:
                logger.warning(f"Could not locate '{element_name}': {response}")
                return None
                
            x_small = int(data["x"])
            y_small = int(data["y"])
            
            # 6. Map back to Original Screen Resolution
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height
            
            final_x = int(x_small * scale_x)
            final_y = int(y_small * scale_y)
            
            logger.info(f"📍 Mapped '{element_name}' to ({final_x}, {final_y}) (Original: {original_width}x{original_height})")
            return (final_x, final_y)
            
        except Exception as e:
            logger.error(f"Error locating element: {e}")
            return None

    def analyze_image_data(self, image_array, prompt: str = "Describe this image.") -> str:
        """
        Analyze a specific image (e.g. from camera).
        Args:
            image_array: numpy array (OpenCV format) or PIL Image
            prompt: Question about the image
        """
        try:
            import numpy as np
            import cv2
            
            # Convert numpy array to PIL Image if needed
            if isinstance(image_array, np.ndarray):
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            elif isinstance(image_array, Image.Image):
                pil_image = image_array
            else:
                return "Error: Invalid image format"

            # Resize if too big
            max_size = (1024, 1024)
            if pil_image.width > max_size[0] or pil_image.height > max_size[1]:
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=70)
            b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

            logger.info("👀 Vision Observer: Sending CAMERA FRAME to VLM...")
            
            analysis = self.llm.chat_with_vision(
                system_prompt="You are an AI Assistant with a camera. You are looking at the real world through a video feed.",
                user_prompt=prompt,
                base64_image=b64_img
            )
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return f"Error analyzing image: {e}"

if __name__ == "__main__":
    # Test
    observer = VisionObserver()
    print("Analyzing screen...")
    print(observer.analyze_screen())
