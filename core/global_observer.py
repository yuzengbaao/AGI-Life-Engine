import os
import time
import logging
import win32gui
import win32process
import psutil
from PIL import ImageGrab, Image
import numpy as np
from typing import Dict, Any
import ctypes

# Configure logging
logger = logging.getLogger("GlobalObserver")

# Set DPI Awareness to capture full screen on high-res monitors
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception as e:
    logger.warning(f"Failed to set DPI awareness: {e}")

class GlobalObserver:
    def __init__(self):
        self.ocr = None
        self._ocr_initialized = False
        self.last_observation_time = 0
        self.observation_interval = 10.0 # Optimized: 10s interval to save CPU
        self.last_window_title = ""
        # 延迟初始化 OCR，避免启动时卡住
        logger.info("🔧 GlobalObserver 已创建 (OCR 将在首次使用时初始化)")

    def _init_ocr(self):
        """延迟初始化 OCR"""
        if self._ocr_initialized:
            return
        self._ocr_initialized = True
        try:
            from paddleocr import PaddleOCR
            # Suppress PaddleOCR internal logger to reduce console noise
            # logging.getLogger("ppocr").setLevel(logging.ERROR)
            
            # Initialize OCR (use_textline_orientation=True for better accuracy, lang='ch' for Chinese support)
            logger.info("正在初始化 PaddleOCR (首次使用，可能需要几秒钟)...")
            # Removed show_log=False as it caused an error in newer versions
            self.ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
            logger.info("✅ Global Vision initialized (PaddleOCR).")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR: {e}")
            self.ocr = None

    def get_active_window_info(self) -> Dict[str, Any]:
        """Get title and process name of the active window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except:
                process_name = "unknown"
                
            return {
                "hwnd": hwnd,
                "title": title,
                "process": process_name
            }
        except Exception as e:
            logger.error(f"Error getting window info: {e}")
            return {"title": "Unknown", "process": "unknown"}

    def observe(self) -> Dict[str, Any]:
        """
        Capture the screen and analyze what's happening.
        Returns a structured observation.
        """
        current_time = time.time()
        
        # Basic Info (Cheap)
        window_info = self.get_active_window_info()
        
        observation = {
            "timestamp": current_time,
            "focus": window_info,
            "vision_text": None,
            "summary": f"User is focused on '{window_info['title']}' ({window_info['process']})"
        }

        # Deep Analysis (Expensive) - Rate Limited
        if current_time - self.last_observation_time > self.observation_interval:
            self.last_observation_time = current_time
            
            # 延迟初始化 OCR
            if not self._ocr_initialized:
                self._init_ocr()
            
            # 1. Capture Screen
            try:
                screen = ImageGrab.grab()
                
                # Optimization: Resize for OCR speedup
                # OCR on 4K/2K screens is very slow. Downscale to max width 1280.
                orig_w, orig_h = screen.size
                if orig_w > 1280:
                    scale = 1280 / orig_w
                    new_h = int(orig_h * scale)
                    screen = screen.resize((1280, new_h), Image.Resampling.LANCZOS)
                    # logger.debug(f"📉 Downscaled screen for OCR: {orig_w}x{orig_h} -> 1280x{new_h}")

                # Debug: Save what we see
                screen.save("latest_vision_debug.png") 
                
                img_np = np.array(screen)
                
                # Check resolution
                height, width = img_np.shape[:2]
                # logger.info(f"Vision Capture Resolution: {width}x{height}")
                
                # 2. OCR (if available)
                if self.ocr:
                    # Run OCR
                    try:
                        result = self.ocr.ocr(img_np, cls=True)
                    except TypeError:
                        # Fallback for versions where cls argument causes issues
                        result = self.ocr.ocr(img_np)
                        
                    if result and isinstance(result, list) and len(result) > 0 and result[0]:
                        # Extract text with confidence filtering
                        valid_texts = []
                        for line in result[0]:
                            # Robust parsing for PaddleOCR output
                            try:
                                if isinstance(line, list) and len(line) >= 2:
                                    text_info = line[1] # Should be (text, score)
                                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                        text = text_info[0]
                                        confidence = text_info[1]
                                        
                                        # Filter low confidence garbage
                                        if isinstance(confidence, (int, float)) and confidence > 0.6: 
                                            valid_texts.append(str(text))
                            except Exception as parse_err:
                                # Just skip this line if malformed
                                continue
                        
                        full_text = " ".join(valid_texts[:50]) 
                        observation["vision_text"] = full_text
                        if valid_texts:
                            observation["summary"] += f". Visible text: {full_text[:100]}..."
                        else:
                             observation["summary"] += ". (Text unclear or low confidence)"
                    else:
                        # No text found or empty result
                        pass
            except Exception as e:
                logger.error(f"Vision analysis failed: {e}")
                
        return observation

if __name__ == "__main__":
    # Test run
    observer = GlobalObserver()
    print("Observing...")
    print(observer.observe())
