import cv2
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

try:
    import easyocr  # Optional: fallback to Tesseract if not available
except ImportError:
    easyocr = None

# Type definitions
@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

@dataclass
class TextDetection:
    text: str
    region: BoundingBox

@dataclass
class ROI:
    image: np.ndarray
    offset_x: int
    offset_y: int

MacroOperation = Tuple[str, ...]
MacroLog = List[str]


class SimulatorInterface:
    """Stub for simulator interface to support type hints."""
    def move_cursor(self, x: int, y: int) -> None: ...
    def click(self) -> None: ...
    def capture_screen(self, save_path: Optional[str] = None) -> np.ndarray: ...


def extract_element_from_text(vlm_description: str) -> List[str]:
    """
    Extract UI element keywords using rule-based matching.
    Optimized with precompiled patterns and early exits.
    """
    keywords = []
    description_lower = vlm_description.lower()
    element_indicators = ["按钮", "文本框", "输入框", "菜单", "图标", "链接", "选项卡"]
    
    for indicator in element_indicators:
        if indicator in description_lower:
            keywords.append(indicator)
    
    # Add color or action-specific terms
    if "蓝色" in description_lower:
        keywords.append("蓝色")
    if "登录" in description_lower:
        keywords.append("登录")
    if "注册" in description_lower:
        keywords.append("注册")

    return keywords


def extract_location_hint(vlm_description: str) -> str:
    """Extract spatial hint with priority mapping."""
    loc_hints = {
        "左上": "top_left",
        "上方": "top",
        "右上": "top_right",
        "左侧": "left",
        "中央": "center",
        "右侧": "right",
        "左下": "bottom_left",
        "下方": "bottom",
        "右下": "bottom_right"
    }
    desc_lower = vlm_description.lower()
    for key, value in loc_hints.items():
        if key in desc_lower:
            return value
    return "full"  # default to full screen


def get_roi_by_spatial_hint(image: np.ndarray, spatial_hint: str) -> ROI:
    """
    Divide image into 3x3 grid and select region based on spatial hint.
    Returns cropped ROI with global offset info.
    """
    height, width = image.shape[:2]
    cx, cy = width // 3, height // 3
    regions = {
        "top_left": (0, 0, cx, cy),
        "top": (cx, 0, cx, cy),
        "top_right": (2*cx, 0, cx, cy),
        "left": (0, cy, cx, cy),
        "center": (cx, cy, cx, cy),
        "right": (2*cx, cy, cx, cy),
        "bottom_left": (0, 2*cy, cx, cy),
        "bottom": (cx, 2*cy, cx, cy),
        "bottom_right": (2*cx, 2*cy, cx, cy),
        "full": (0, 0, width, height)
    }
    
    x, y, w, h = regions.get(spatial_hint, (0, 0, width, height))
    x, y = max(0, x), max(0, y)
    w, h = min(w, width - x), min(h, height - y)
    
    roi_img = image[y:y+h, x:x+w]
    return ROI(image=roi_img, offset_x=x, offset_y=y)


def extract_color_region(image: np.ndarray, color: str) -> Optional[BoundingBox]:
    """
    Extract dominant region of specified color using HSV thresholding.
    Supports only 'blue' for now; extendable.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if color == "blue":
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    else:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 50:  # noise filter
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    return BoundingBox(x=x, y=y, w=w, h=h)


def ocr_detect_text(image: np.ndarray) -> List[TextDetection]:
    """
    Perform OCR on image using EasyOCR if available, else mock or skip.
    Caches reader to avoid reinitialization overhead.
    """
    try:
        if not hasattr(ocr_detect_text, "reader"):
            ocr_detect_text.reader = easyocr.Reader(['ch_sim', 'en']) if easyocr else None
        if not ocr_detect_text.reader:
            return []

        results = ocr_detect_text.reader.readtext(image)
        detections = []
        for (bbox, text, prob) in results:
            x_min = int(min(point[0] for point in bbox))
            y_min = int(min(point[1] for point in bbox))
            x_max = int(max(point[0] for point in bbox))
            y_max = int(max(point[1] for point in bbox))
            w, h = x_max - x_min, y_max - y_min
            detections.append(
                TextDetection(text=text, region=BoundingBox(x=x_min, y=y_min, w=w, h=h, confidence=prob))
            )
        return detections
    except Exception as e:
        print(f"OCR failed: {e}")
        return []


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x_left = max(box1.x, box2.x)
    y_top = max(box1.y, box2.y)
    x_right = min(box1.x + box1.w, box2.x + box2.w)
    y_bottom = min(box1.y + box1.h, box2.y + box2.h)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1.w * box1.h
    box2_area = box2.w * box2.h
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def fuse_candidates(candidates: List[BoundingBox], img_shape: Tuple[int, int]) -> Optional[BoundingBox]:
    """
    Fuse multiple candidate regions using IoU clustering heuristic.
    Selects the largest high-overlap cluster.
    """
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Cluster by IoU > 0.1
    clusters: List[List[BoundingBox]] = []
    for cand in candidates:
        matched = False
        for cluster in clusters:
            avg_box = BoundingBox(
                x=sum(b.x for b in cluster) // len(cluster),
                y=sum(b.y for b in cluster) // len(cluster),
                w=sum(b.w for b in cluster) // len(cluster),
                h=sum(b.h for b in cluster) // len(cluster)
            )
            if compute_iou(avg_box, cand) > 0.1:
                cluster.append(cand)
                matched = True
                break
        if not matched:
            clusters.append([cand])

    if not clusters:
        return None

    # Choose cluster with highest total confidence and area
    best_cluster = max(clusters, key=lambda cl: sum(b.confidence * b.w * b.h for b in cl))
    final_box = BoundingBox(
        x=int(np.mean([b.x for b in best_cluster])),
        y=int(np.mean([b.y for b in best_cluster])),
        w=int(np.mean([b.w for b in best_cluster])),
        h=int(np.mean([b.h for b in best_cluster]))
    )
    return final_box


def integrate_vision_and_macro(
    vlm_description: str,
    screenshot_path: str,
    simulator: SimulatorInterface
) -> Tuple[bool, Optional[Tuple[int, int]], MacroLog]:
    """
    集成视觉理解与宏操作执行的原型函数
    
    参数:
        vlm_description: VLM对目标UI元素的自然语言描述
        screenshot_path: 当前界面截图文件路径
        simulator: 支持鼠标/点击模拟的环境对象（如PyAutoGUI封装）

    返回:
        success: 是否成功定位并点击
        coordinates: 识别出的目标坐标(x, y)
        macro_log: 执行的操作日志
    """
    macro_log: MacroLog = []
    
    # Step 1: Parse VLM description
    try:
        element_keywords = extract_element_from_text(vlm_description)
        spatial_hint = extract_location_hint(vlm_description)
        macro_log.append(f"解析关键词: {element_keywords}, 位置提示: {spatial_hint}")
    except Exception as e:
        macro_log.append(f"解析失败: {str(e)}")
        return False, None, macro_log

    # Step 2: Load screenshot
    try:
        image = cv2.imread(screenshot_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {screenshot_path}")
        height, width = image.shape[:2]
        macro_log.append(f"截图尺寸: {width}x{height}")
    except Exception as e:
        macro_log.append(f"图像加载失败: {str(e)}")
        return False, None, macro_log

    # Step 3: Get ROI based on spatial hint
    try:
        roi = get_roi_by_spatial_hint(image, spatial_hint)
        roi_image = roi.image
        macro_log.append(f"应用ROI: 偏移({roi.offset_x}, {roi.offset_y}), 尺寸{roi_image.shape[1]}x{roi_image.shape[0]}")
    except Exception as e:
        macro_log.append(f"ROI提取失败: {str(e)}")
        return False, None, macro_log

    # Step 4: Multi-modal candidate detection
    candidate_regions: List[BoundingBox] = []

    try:
        # Color detection
        if "蓝色" in vlm_description:
            blue_box = extract_color_region(roi_image, "blue")
            if blue_box:
                candidate_regions.append(blue_box)
                macro_log.append("颜色匹配: 蓝色区域检测到")

        # Text detection via OCR
        if any(kw in vlm_description for kw in ["登录", "注册", "按钮"]):
            text_detections = ocr_detect_text(roi_image)
            login_texts = [det for det in text_detections if "登录" in det.text or "注册" in det.text]
            for lt in login_texts:
                candidate_regions.append(lt.region)
                macro_log.append(f"OCR匹配文本: '{lt.text}' @ ({lt.region.x}, {lt.region.y})")
    except Exception as e:
        macro_log.append(f"特征检测异常: {str(e)}")

    # Step 5: Fuse candidates
    try:
        final_box = fuse_candidates(candidate_regions, roi_image.shape)
        if final_box is None:
            macro_log.append("未找到匹配的UI元素")
            return False, None, macro_log

        # Convert to global screen coordinates
        x_center = roi.offset_x + final_box.x + final_box.w // 2
        y_center = roi.offset_y + final_box.y + final_box.h // 2
        macro_log.append(f"目标坐标确定: ({x_center}, {y_center})")
    except Exception as e:
        macro_log.append(f"候选融合失败: {str(e)}")
        return False, None, macro_log

    # Step 6: Generate macro sequence
    macro_sequence: List[MacroOperation] = [
        ("move_cursor", x_center, y_center),
        ("click",)
    ]

    # Step 7: Execute in simulator
    try:
        for op in macro_sequence:
            action = op[0]
            if action == "move_cursor":
                simulator.move_cursor(op[1], op[2])
                macro_log.append(f"移动光标至 ({op[1]}, {op[2]})")
            elif action == "click":
                simulator.click()
                macro_log.append("执行点击")
        
        time.sleep(1)  # Allow UI to respond

        # Step 8: Validate result
        post_click_screenshot_path = screenshot_path.replace(".png", "_post_click.png")
        try:
            new_screenshot = simulator.capture_screen(post_click_screenshot_path)
            post_ocr_result = " ".join([det.text for det in ocr_detect_text(new_screenshot)])
            success_keywords = ["欢迎", "主页", "成功", "进入", "加载"]
            success = any(kw in post_ocr_result for kw in success_keywords)
            macro_log.append(f"验证通过: {'是' if success else '否'}，检测到文本: {post_ocr_result[:50]}...")
        except Exception as e:
            macro_log.append(f"验证阶段出错: {str(e)}")
            success = False

        return success, (x_center, y_center), macro_log

    except Exception as e:
        error_msg = f"宏执行失败: {str(e)}"
        macro_log.append(error_msg)
        return False, (x_center, y_center), macro_log