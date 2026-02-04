重构 core/desktop_automation.py 中的 execute_visual_click 函数，目标是提升坐标映射精度，集成视觉语言模型（VLM）实时反馈校正机制，并通过单元测试验证点击成功率。

一、需求分析

原函数 execute_visual_click 主要功能为：基于图像识别在桌面界面上定位目标元素并执行鼠标点击。存在的问题包括：
- 坐标映射受屏幕缩放、DPI设置影响，导致点击偏移；
- 图像匹配失败或误匹配时缺乏纠正机制；
- 缺少对点击是否成功的验证。

改进方向：
1. 提升坐标映射精度：引入系统DPI和缩放因子补偿；
2. 集成VLM实时反馈：在点击后截屏，由VLM判断目标状态是否变化，若未成功则进行位置校正重试；
3. 添加点击结果验证与重试逻辑；
4. 编写单元测试模拟多种场景，评估点击成功率。

二、重构后的 execute_visual_click 函数实现

文件路径：core/desktop_automation.py

import os
import time
import cv2
import numpy as np
from PIL import ImageGrab
from pynput.mouse import Controller, Button
from typing import Tuple, Optional
import logging

# 模拟VLM接口（实际项目中替换为真实API调用）
def query_vlm_feedback(screenshot: np.ndarray, task_instruction: str) -> dict:
    """
    模拟调用VLM获取反馈
    返回示例: {"success": True, "corrected_box": [x1, y1, x2, y2], "reason": "Target was clicked"}
    """
    # 此处为模拟逻辑，实际应调用VLM API
    # 简化实现：假设VLM能检测目标是否存在并建议修正区域
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return {
            "success": False,  # 模拟首次未成功
            "corrected_box": [x + w // 2, y + h // 2, x + w // 2 + 10, y + h // 2 + 10],
            "reason": "Target still present, click again"
        }
    else:
        return {"success": True, "corrected_box": None, "reason": "Target disappeared"}

# 获取系统缩放比例（Windows 示例）
def get_system_scale_factor() -> float:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        scale = user32.GetDpiForSystem() / 96.0  # 基准 DPI 为 96
        return scale
    except Exception as e:
        logging.warning(f"无法获取系统缩放因子，使用默认值1.0: {e}")
        return 1.0

# 图像模板匹配并返回中心坐标
def find_template_on_screen(template_path: str) -> Optional[Tuple[int, int]]:
    if not os.path.exists(template_path):
        logging.error(f"模板图片不存在: {template_path}")
        return None

    screen_pil = ImageGrab.grab()
    screen_cv = np.array(screen_pil)
    screen_gray = cv2.cvtColor(screen_cv, cv2.COLOR_BGR2GRAY)

    template = cv2.imread(template_path, 0)
    if template is None:
        logging.error(f"无法加载模板图像: {template_path}")
        return None

    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 设定匹配阈值
    if max_val < 0.7:
        logging.info(f"图像匹配度低: {max_val:.3f}")
        return None

    h, w = template.shape
    center_x = max_loc[0] + w // 2
    center_y = max_loc[1] + h // 2

    scale = get_system_scale_factor()
    adjusted_x = int(center_x * scale)
    adjusted_y = int(center_y * scale)

    return (adjusted_x, adjusted_y)

# 重构后的 execute_visual_click 函数
def execute_visual_click(template_path: str, max_retries: int = 3) -> bool:
    """
    执行视觉点击操作，集成坐标映射校正与VLM反馈机制
    :param template_path: 目标按钮/图标的模板图像路径
    :param max_retries: 最大重试次数
    :return: 是否成功完成点击并确认效果
    """
    mouse = Controller()
    scale_factor = get_system_scale_factor()
    logging.info(f"系统缩放因子: {scale_factor:.2f}")

    for attempt in range(max_retries):
        logging.info(f"尝试第 {attempt + 1} 次点击")

        # 步骤1: 图像识别定位
        pos = find_template_on_screen(template_path)
        if not pos:
            logging.warning("未找到目标元素")
            continue

        # 步骤2: 执行点击
        mouse.position = pos
        time.sleep(0.3)
        mouse.click(Button.left, 1)
        time.sleep(1.0)  # 等待界面响应

        # 步骤3: 截屏并请求VLM反馈
        post_click_screenshot = np.array(ImageGrab.grab())
        instruction = f"Check if the element matching '{template_path}' has been successfully clicked (e.g., disappeared, changed state). If not, suggest a corrected click region."
        vlm_response = query_vlm_feedback(post_click_screenshot, instruction)

        # 步骤4: 判断是否成功
        if vlm_response["success"]:
            logging.info("VLM确认点击成功")
            return True

        # 步骤5: 若失败且有建议区域，则移动到新位置再试
        corrected_box = vlm_response.get("corrected_box")
        if corrected_box and len(corrected_box) == 4:
            new_x = int((corrected_box[0] + corrected_box[2]) / 2 / scale_factor)
            new_y = int((corrected_box[1] + corrected_box[3]) / 2 / scale_factor)
            # 更新鼠标位置（可选：添加微小偏移避免重复点）
            jitter = np.random.randint(-5, 5)
            mouse.position = (new_x + jitter, new_y + jitter)
            time.sleep(0.5)
            mouse.click(Button.left, 1)
            time.sleep(1.0)

            # 再次验证
            final_screenshot = np.array(ImageGrab.grab())
            final_vlm = query_vlm_feedback(final_screenshot, instruction)
            if final_vlm["success"]:
                logging.info("二次点击经VLM确认成功")
                return True

    logging.error("达到最大重试次数，点击失败")
    return False

三、单元测试实现

文件路径：tests/test_desktop_automation.py

import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from core.desktop_automation import execute_visual_click, find_template_on_screen, query_vlm_feedback

class TestDesktopAutomation(unittest.TestCase):

    @patch('core.desktop_automation.ImageGrab.grab')
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch('cv2.matchTemplate')
    @patch('cv2.minMaxLoc')
    def test_find_template_on_screen_success(self, mock_minMaxLoc, mock_matchTemplate, mock_cvtColor, mock_imread, mock_grab):
        # 模拟屏幕截图
        mock_screen = np.zeros((1080, 1920), dtype=np.uint8)
        mock_grab.return_value = mock_screen

        # 模拟模板图像
        mock_template = np.ones((50, 50), dtype=np.uint8)
        mock_imread.return_value = mock_template

        mock_cvtColor.side_effect = lambda img, _: img

        # 模拟匹配结果
        mock_matchTemplate.return_value = np.array([[0.1, 0.8], [0.2, 0.7]])
        mock_minMaxLoc.return_value = (0, 0, 0, (1, 0))  # max_loc at (1,0)

        with patch('os.path.exists', return_value=True):
            result = find_template_on_screen("dummy.png")

        self.assertEqual(result, (25, 25))  # center of (1,0) block

    @patch('core.desktop_automation.find_template_on_screen')
    @patch('core.desktop_automation.query_vlm_feedback')
    @patch('PIL.ImageGrab.grab')
    @patch('pynput.mouse.Controller')
    def test_execute_visual_click_success_on_first_try(self, mock_controller, mock_grab, mock_vlm, mock_find):
        mock_find.return_value = (100, 200)
        mock_mouse = MagicMock()
        mock_controller.return_value = mock_mouse

        mock_grab.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_vlm.return_value = {"success": True, "corrected_box": None}

        success = execute_visual_click("button.png", max_retries=2)

        self.assertTrue(success)
        mock_mouse.click.assert_called()

    @patch('core.desktop_automation.find_template_on_screen')
    @patch('core.desktop_automation.query_vlm_feedback')
    @patch('PIL.ImageGrab.grab')
    @patch('pynput.mouse.Controller')
    def test_execute_visual_click_with_vlm_correction(self, mock_controller, mock_grab, mock_vlm, mock_find):
        mock_find.return_value = (100, 200)
        mock_mouse = MagicMock()
        mock_controller.return_value = mock_mouse

        screenshot = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_grab.return_value = screenshot

        # 第一次失败，第二次成功
        mock_vlm.side_effect = [
            {"success": False, "corrected_box": [105, 205, 115, 215]},
            {"success": True, "corrected_box": None}
        ]

        with patch('core.desktop_automation.get_system_scale_factor', return_value=1.0):
            success = execute_visual_click("button.png", max_retries=2)

        self.assertTrue(success)
        self.assertEqual(mock_mouse.click.call_count, 2)

    @patch('core.desktop_automation.find_template_on_screen', return_value=None)
    def test_execute_visual_click_fails_to_locate(self, mock_find):
        success = execute_visual_click("missing.png", max_retries=3)
        self.assertFalse(success)

if __name__ == '__main__':
    unittest.main()

四、测试说明

1. 测试覆盖场景：
   - 图像识别成功定位
   - VLM首次反馈失败但二次成功
   - 完全无法识别模板
   - 点击后状态被VLM确认改变

2. 模拟方式：
   - 使用 unittest.mock 模拟屏幕捕获、图像处理、鼠标控制
   - VLM接口行为通过 side_effect 控制多轮响应

3. 成功率验证方法：
   在真实环境中运行以下脚本统计成功率：

   def run_click_success_rate_test(template_path, trials=50):
       successes = 0
       for i in range(trials):
           if execute_visual_click(template_path, max_retries=3):
               successes += 1
           time.sleep(2)
       rate = successes / trials
       print(f"点击成功率: {rate:.2%} ({successes}/{trials})")
       return rate

五、总结

本次重构实现了：
- 坐标映射精度提升：通过系统DPI补偿，减少因缩放导致的偏移；
- 引入VLM作为外部感知反馈源，实现点击结果验证与动态校正；
- 增加重试与自适应调整机制，提高鲁棒性；
- 单元测试覆盖核心逻辑，确保功能稳定性；
- 日志输出便于调试与监控。

后续可扩展方向：
- 将VLM替换为真实API（如GPT-4V、LLaVA等）；
- 支持多模态指令理解（自然语言描述目标）；
- 加入OCR辅助识别文本按钮；
- 构建自动化测试仪表盘统计长期成功率趋势。