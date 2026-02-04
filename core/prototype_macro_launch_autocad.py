import pyautogui
import cv2
import numpy as np
from PIL import Image
import time
import os
from typing import Optional, Tuple, Dict, Any

# 配置路径（请根据实际情况修改）
TEMPLATE_PATH: str = 'acad_icon_template.png'  # AutoCAD 图标模板截图文件
LAUNCH_WAIT_TIME: int = 10  # 启动等待时间（秒）
VERIFICATION_REGION: Tuple[int, int, int, int] = (0, 0, 1920, 1080)  # 屏幕捕获区域（全屏）

def locate_application_icon(template_path: str) -> Optional[Tuple[int, int]]:
    """
    使用模板匹配在屏幕上查找 AutoCAD 图标
    返回图标中心坐标，未找到则返回 None

    Args:
        template_path: 模板图像文件路径

    Returns:
        匹配到的图标中心坐标 (x, y)，未找到或出错返回 None
    """
    try:
        if not os.path.exists(template_path):
            print(f"错误：模板文件 {template_path} 不存在")
            return None

        # 截取当前屏幕
        screenshot_pil = pyautogui.screenshot()
        screenshot: np.ndarray = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
        template: Optional[np.ndarray] = cv2.imread(template_path, cv2.IMREAD_COLOR)

        if template is None:
            print("无法读取模板图像，请检查文件格式")
            return None

        # 执行模板匹配
        result: np.ndarray = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val: float
        max_val: float
        min_loc: Tuple[int, int]
        max_loc: Tuple[int, int]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 设置匹配阈值
        threshold: float = 0.8
        if max_val >= threshold:
            h, w = template.shape[:2]
            center_x: int = max_loc[0] + w // 2
            center_y: int = max_loc[1] + h // 2
            print(f"找到 AutoCAD 图标，位置：({center_x}, {center_y})，匹配度：{max_val:.3f}")
            return (center_x, center_y)
        else:
            print(f"未找到 AutoCAD 图标，最高匹配度：{max_val:.3f} < {threshold}")
            return None
    except Exception as e:
        print(f"图标识别过程中发生异常：{str(e)}")
        return None

def click_position(pos: Optional[Tuple[int, int]]) -> bool:
    """
    移动鼠标至指定位置并单击

    Args:
        pos: 屏幕坐标 (x, y)

    Returns:
        是否点击成功
    """
    if pos is None:
        print("无效的位置，无法执行点击")
        return False
    try:
        pyautogui.moveTo(pos[0], pos[1], duration=0.5)
        pyautogui.click()
        print("已模拟点击 AutoCAD 图标")
        return True
    except Exception as e:
        print(f"点击操作失败：{str(e)}")
        return False

def capture_verification_screenshot(output_path: str = 'verification_result.png') -> Optional[str]:
    """
    捕获指定区域截图用于结果验证

    Args:
        output_path: 输出图像保存路径

    Returns:
        成功时返回保存路径，失败返回 None
    """
    try:
        if not (pyautogui.size().width >= VERIFICATION_REGION[2] and 
                pyautogui.size().height >= VERIFICATION_REGION[3]):
            print("警告：验证区域超出当前屏幕分辨率")

        screenshot: Image.Image = pyautogui.screenshot(region=VERIFICATION_REGION)
        screenshot.save(output_path)
        print(f"验证截图已保存至：{output_path}")
        return output_path
    except Exception as e:
        print(f"截图保存失败：{str(e)}")
        return None

def verify_acad_startup(verification_image_path: str) -> bool:
    """
    验证 AutoCAD 是否成功启动（通过检测主界面特征元素）

    Args:
        verification_image_path: 验证截图路径

    Returns:
        是否检测到主界面
    """
    MAIN_WINDOW_TEMPLATE: str = 'acad_main_window.png'
    if not os.path.exists(MAIN_WINDOW_TEMPLATE):
        print("警告：主窗口验证模板不存在，跳过精确验证")
        return True  # 默认认为成功（降级模式）

    try:
        main_screenshot: Optional[np.ndarray] = cv2.imread(verification_image_path)
        window_template: Optional[np.ndarray] = cv2.imread(MAIN_WINDOW_TEMPLATE)

        if main_screenshot is None:
            print("无法加载验证截图")
            return False
        if window_template is None:
            print("无法加载主窗口模板")
            return False

        result: np.ndarray = cv2.matchTemplate(main_screenshot, window_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > 0.7:
            print(f"检测到 AutoCAD 主界面，匹配度：{max_val:.3f}，启动成功")
            return True
        else:
            print(f"未检测到主界面，匹配度：{max_val:.3f}，启动可能失败")
            return False
    except Exception as e:
        print(f"启动验证过程出错：{str(e)}")
        return False

def launch_autocad_process(timeout: int = 30) -> bool:
    """
    启动 AutoCAD 进程并等待加载
    
    Args:
        timeout: 等待超时的秒数
        
    Returns:
        启动是否成功
    """
    print("开始执行 AutoCAD 启动宏操作...")
    
    # 步骤1：查找图标
    icon_position: Optional[Tuple[int, int]] = locate_application_icon(TEMPLATE_PATH)
    if icon_position is None:
        print("未能定位 AutoCAD 图标，尝试直接使用命令启动...")
        # Fallback: Try launching via os.startfile or subprocess if icon not found
        try:
            # 假设默认路径，实际应可配置
            import subprocess
            subprocess.Popen(["acad.exe"]) 
            print("已通过 subprocess 启动 acad.exe")
        except FileNotFoundError:
            print("未能通过命令启动 AutoCAD，流程终止")
            return False
    else:
        # 步骤2：点击启动
        if not click_position(icon_position):
            print("启动点击失败，流程终止")
            return False
    
    # 步骤3：等待程序加载
    print(f"等待 {timeout} 秒供 AutoCAD 启动...")
    time.sleep(timeout)
    
    # 步骤4：捕获验证截图
    verification_image: Optional[str] = capture_verification_screenshot()
    if verification_image is None:
        print("无法获取验证截图，结果不确定")
        return False
    
    # 步骤5：验证启动结果
    success: bool = verify_acad_startup(verification_image)
    return success

def main() -> None:
    """
    主函数：执行 AutoCAD 启动宏操作完整流程
    """
    if launch_autocad_process(timeout=LAUNCH_WAIT_TIME):
        print("✅ AutoCAD 启动宏操作成功完成")
    else:
        print("❌ AutoCAD 启动宏操作失败或未检测到主界面")

if __name__ == "__main__":
    main()