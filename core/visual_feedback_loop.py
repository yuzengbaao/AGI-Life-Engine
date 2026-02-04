# visual_feedback_loop.py

"""
基于视觉识别的宏操作执行器
支持录制、回放、点击失败后的自动重试与修正逻辑
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
import threading
import pyautogui
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualFeedbackLoop:
    """
    基于视觉反馈的宏操作执行器
    实现屏幕内容识别 -> 操作执行 -> 反馈验证 -> 失败修正 的闭环
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sct = mss.mss()
        self.recordings = []
        self.current_recording = []
        self.is_recording = False
        self.is_playing = False
        self.template_cache = {}
        self.confidence_threshold = 0.8
        self.max_retry_attempts = 3
        self.retry_delay = 0.5
        self.search_region = None  # (x, y, width, height)

    def capture_screen(self) -> np.ndarray:
        """捕获当前屏幕截图"""
        screenshot = self.sct.grab(self.sct.monitors[1])
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def find_template(self, template_path: str, region: tuple = None, 
                     threshold: float = None) -> Dict:
        """
        在屏幕上查找模板图像
        返回匹配位置和置信度
        """
        if threshold is None:
            threshold = self.confidence_threshold

        # 检查缓存
        cache_key = f"{template_path}_{threshold}"
        if cache_key in self.template_cache:
            template_img = self.template_cache[cache_key]
        else:
            template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template_img is None:
                logger.error(f"无法加载模板图像: {template_path}")
                return {"found": False}
            self.template_cache[cache_key] = template_img

        screen_img = self.capture_screen()

        # 如果指定了搜索区域，裁剪屏幕图像
        if region:
            x, y, w, h = region
            screen_crop = screen_img[y:y+h, x:x+w]
        else:
            screen_crop = screen_img
            region = (0, 0, self.screen_width, self.screen_height)

        # 模板匹配
        result = cv2.matchTemplate(screen_crop, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            # 计算在完整屏幕上的坐标
            abs_x = max_loc[0] + region[0]
            abs_y = max_loc[1] + region[1]
            center_x = abs_x + template_img.shape[1] // 2
            center_y = abs_y + template_img.shape[0] // 2
            
            return {
                "found": True,
                "x": center_x,
                "y": center_y,
                "confidence": max_val,
                "bbox": (abs_x, abs_y, template_img.shape[1], template_img.shape[0])
            }
        else:
            return {"found": False, "confidence": max_val}

    def click_at(self, x: int, y: int, delay: float = 0.1):
        """执行点击操作"""
        pyautogui.moveTo(x, y)
        time.sleep(0.1)
        pyautogui.click()
        time.sleep(delay)

    def type_text(self, text: str, delay: float = 0.05):
        """输入文本"""
        pyautogui.typewrite(text, interval=delay)

    def wait_for_element(self, template_path: str, timeout: float = 10.0, 
                        check_interval: float = 0.5) -> bool:
        """等待元素出现"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            match = self.find_template(template_path)
            if match["found"]:
                return True
            time.sleep(check_interval)
        return False

    def record_start(self):
        """开始录制宏操作"""
        if self.is_recording:
            logger.warning("已经在录制中")
            return

        self.is_recording = True
        self.current_recording = []
        logger.info("开始录制...")

    def record_stop(self) -> List[Dict]:
        """停止录制并返回录制的操作序列"""
        if not self.is_recording:
            logger.warning("未在录制状态")
            return []

        self.is_recording = False
        recording = self.current_recording.copy()
        logger.info(f"录制完成，共记录 {len(recording)} 个操作")
        return recording

    def record_event(self, event_type: str, **kwargs):
        """记录单个事件"""
        if not self.is_recording:
            return

        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": kwargs
        }
        self.current_recording.append(event)

    def add_click_operation(self, template_path: str, description: str = ""):
        """添加点击操作到录制序列"""
        self.record_event(
            "click",
            template_path=template_path,
            description=description
        )

    def add_input_operation(self, text: str, description: str = ""):
        """添加输入操作到录制序列"""
        self.record_event(
            "input",
            text=text,
            description=description
        )

    def add_wait_operation(self, seconds: float, description: str = ""):
        """添加等待操作到录制序列"""
        self.record_event(
            "wait",
            seconds=seconds,
            description=description
        )

    def add_verification_point(self, template_path: str, description: str = ""):
        """添加验证点"""
        self.record_event(
            "verify",
            template_path=template_path,
            description=description
        )

    def execute_click_operation(self, operation: Dict) -> bool:
        """执行点击操作，包含重试和修正逻辑"""
        template_path = operation["data"]["template_path"]
        description = operation["data"].get("description", "")

        logger.info(f"执行点击操作: {description or template_path}")

        for attempt in range(self.max_retry_attempts):
            # 查找目标元素
            match = self.find_template(template_path)
            
            if match["found"]:
                # 执行点击
                self.click_at(match["x"], match["y"])
                
                # 验证点击后是否成功（简单延迟验证）
                time.sleep(0.5)
                logger.info(f"点击成功: 尝试 {attempt + 1}")
                return True
            else:
                logger.warning(f"未找到目标元素: {description}, 置信度: {match.get('confidence', 0):.3f}, 尝试 {attempt + 1}")
                
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue

        # 所有重试都失败，尝试修正策略
        return self.recovery_strategy(operation)

    def execute_input_operation(self, operation: Dict) -> bool:
        """执行输入操作"""
        text = operation["data"]["text"]
        description = operation["data"].get("description", "")

        logger.info(f"执行输入操作: {description or text}")
        try:
            self.type_text(text)
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.error(f"输入操作失败: {e}")
            return False

    def execute_wait_operation(self, operation: Dict) -> bool:
        """执行等待操作"""
        seconds = operation["data"]["seconds"]
        description = operation["data"].get("description", "")

        logger.info(f"执行等待操作: {description or f'{seconds}秒'}")
        time.sleep(seconds)
        return True

    def execute_verify_operation(self, operation: Dict) -> bool:
        """执行验证操作"""
        template_path = operation["data"]["template_path"]
        description = operation["data"].get("description", "")
        timeout = operation["data"].get("timeout", 10.0)

        logger.info(f"执行验证操作: {description or template_path}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            match = self.find_template(template_path)
            if match["found"]:
                logger.info("验证通过")
                return True
            time.sleep(0.5)
        
        logger.error(f"验证失败: {description}")
        return False

    def recovery_strategy(self, operation: Dict) -> bool:
        """
        失败后的恢复策略
        1. 扩大搜索区域
        2. 降低匹配阈值
        3. 尝试替代元素
        4. 回到起始点重新开始
        """
        logger.info("启动恢复策略...")

        original_threshold = self.confidence_threshold
        original_region = self.search_region

        try:
            # 策略1: 降低匹配阈值
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.2)
            logger.info(f"降低匹配阈值到 {self.confidence_threshold}")

            match = self.find_template(operation["data"]["template_path"])
            if match["found"]:
                self.click_at(match["x"], match["y"])
                time.sleep(0.5)
                return True

            # 策略2: 全屏搜索
            self.search_region = None
            logger.info("切换到全屏搜索")

            match = self.find_template(operation["data"]["template_path"])
            if match["found"]:
                self.click_at(match["x"], match["y"])
                time.sleep(0.5)
                return True

            # 策略3: 尝试常见的替代元素（如确认按钮、下一步按钮等）
            alternative_templates = [
                "templates/confirm_button.png",
                "templates/ok_button.png",
                "templates/next_button.png"
            ]

            for alt_template in alternative_templates:
                if os.path.exists(alt_template):
                    match = self.find_template(alt_template)
                    if match["found"]:
                        logger.info(f"找到替代元素: {alt_template}")
                        self.click_at(match["x"], match["y"])
                        time.sleep(0.5)
                        return True

            logger.error("所有恢复策略均失败")
            return False

        finally:
            # 恢复原始设置
            self.confidence_threshold = original_threshold
            self.search_region = original_region

    def play(self, operations: List[Dict], loop_count: int = 1) -> Dict:
        """
        执行宏操作序列
        返回执行结果统计
        """
        if self.is_playing:
            logger.warning("已经在执行中")
            return {}

        self.is_playing = True
        results = {
            "success_count": 0,
            "failure_count": 0,
            "total_operations": len(operations) * loop_count,
            "start_time": time.time(),
            "details": []
        }

        try:
            for loop in range(loop_count):
                logger.info(f"开始第 {loop + 1} 轮执行")

                for i, op in enumerate(operations):
                    if not self.is_playing:
                        break

                    op_result = {
                        "operation_index": i,
                        "operation_type": op["type"],
                        "success": False,
                        "attempt": 0
                    }

                    # 执行对应类型的操作
                    if op["type"] == "click":
                        op_result["success"] = self.execute_click_operation(op)
                    elif op["type"] == "input":
                        op_result["success"] = self.execute_input_operation(op)
                    elif op["type"] == "wait":
                        op_result["success"] = self.execute_wait_operation(op)
                    elif op["type"] == "verify":
                        op_result["success"] = self.execute_verify_operation(op)
                    else:
                        logger.warning(f"未知操作类型: {op['type']}")
                        op_result["success"] = False

                    if op_result["success"]:
                        results["success_count"] += 1
                    else:
                        results["failure_count"] += 1

                    results["details"].append(op_result)

                    # 操作间最小间隔
                    time.sleep(0.1)

                if loop < loop_count - 1:
                    # 轮次间延迟
                    time.sleep(1.0)

        except Exception as e:
            logger.error(f"执行过程中发生异常: {e}")
            results["error"] = str(e)
        finally:
            self.is_playing = False
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]

        logger.info(f"执行完成: 成功 {results['success_count']}, 失败 {results['failure_count']}")
        return results

    def stop_playback(self):
        """停止正在执行的宏"""
        self.is_playing = False
        logger.info("已停止宏执行")

    def save_recording(self, operations: List[Dict], filename: str = None) -> str:
        """保存录制的宏到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"macro_{timestamp}.json"

        data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "operations": operations
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"宏已保存到: {filename}")
        return filename

    def load_recording(self, filename: str) -> List[Dict]:
        """从文件加载宏"""
        if not os.path.exists(filename):
            logger.error(f"文件不存在: {filename}")
            return []

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"宏已加载: {filename}")
            return data.get("operations", [])
        except Exception as e:
            logger.error(f"加载宏文件失败: {e}")
            return []

    def set_confidence_threshold(self, threshold: float):
        """设置匹配置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"设置匹配阈值: {self.confidence_threshold}")

    def set_max_retry_attempts(self, attempts: int):
        """设置最大重试次数"""
        self.max_retry_attempts = max(1, attempts)
        logger.info(f"设置最大重试次数: {self.max_retry_attempts}")

    def set_search_region(self, x: int, y: int, width: int, height: int):
        """设置搜索区域"""
        self.search_region = (x, y, width, height)
        logger.info(f"设置搜索区域: {self.search_region}")

    def clear_search_region(self):
        """清除搜索区域限制"""
        self.search_region = None
        logger.info("清除搜索区域限制")


# 使用示例（不在类中）
def example_usage():
    """使用示例"""
    vfl = VisualFeedbackLoop()

    # 开始录制
    vfl.record_start()

    # 添加操作
    vfl.add_click_operation("templates/login_button.png", "点击登录按钮")
    vfl.add_input_operation("username123", "输入用户名")
    vfl.add_click_operation("templates/submit_button.png", "点击提交")
    vfl.add_verify_operation("templates/dashboard.png", "验证进入仪表板")

    # 停止录制
    operations = vfl.record_stop()

    # 保存录制
    filename = vfl.save_recording(operations)

    # 加载并执行
    loaded_ops = vfl.load_recording(filename)
    results = vfl.play(loaded_ops, loop_count=1)

    print(f"执行结果: {results}")


if __name__ == "__main__":
    example_usage()