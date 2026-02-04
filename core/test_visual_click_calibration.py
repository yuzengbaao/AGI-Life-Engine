import time
import json
from typing import Dict, Tuple, List, Optional
import random

# 模拟不同分辨率下的屏幕点击与视觉验证测试脚本
# 使用 execute_visual_click 执行点击，通过 vision_observer.py 验证实际点击位置
# 记录偏差数据用于后续动态校准模型训练


def execute_visual_click(x: float, y: float, resolution: Tuple[int, int]) -> Tuple[int, int]:
    """
    模拟在指定分辨率下执行视觉点击。

    Args:
        x (float): 归一化横坐标（0.0 ~ 1.0）
        y (float): 归一化纵坐标（0.0 ~ 1.0）
        resolution (Tuple[int, int]): 屏幕分辨率 (width, height)

    Returns:
        Tuple[int, int]: 实际点击的像素坐标 (actual_x, actual_y)

    Raises:
        ValueError: 当归一化坐标超出范围时抛出异常
    """
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"归一化坐标必须在 [0.0, 1.0] 范围内，但得到 ({x}, {y})")

    width, height = resolution
    actual_x = int(x * width)
    actual_y = int(y * height)
    print(f"[点击模拟] 分辨率: {resolution}, 归一化坐标: ({x:.3f}, {y:.3f}) -> 实际坐标: ({actual_x}, {actual_y})")
    # 此处调用实际的点击接口（如ADB、PyAutoGUI等）
    # 模拟点击延迟
    time.sleep(0.5)
    return actual_x, actual_y


def get_observed_position_via_vision() -> Tuple[int, int]:
    """
    调用 vision_observer.py 获取实际观测到的点击中心位置。
    这里模拟返回一个带有随机偏移的坐标。

    Returns:
        Tuple[int, int]: 视觉系统检测到的点击坐标 (observed_x, observed_y)
    """
    try:
        # 模拟视觉检测返回的坐标（含误差）
        offset_x = random.randint(-15, 15)
        offset_y = random.randint(-15, 15)
        observed_x = 540 + offset_x  # 假设目标在 540,960 周围
        observed_y = 960 + offset_y
        print(f"[视觉观测] 检测到点击位置: ({observed_x}, {observed_y}), 偏移: ({offset_x}, {offset_y})")
        return observed_x, observed_y
    except Exception as e:
        print(f"[错误] 视觉观测失败: {e}")
        # 返回默认值或重试机制可在此扩展
        return 540, 960


def run_calibration_test(
    resolution: Tuple[int, int],
    test_points: List[Tuple[float, float]],
    rounds: int = 3
) -> List[Dict[str, object]]:
    """
    在指定分辨率下运行多轮点击测试，收集偏差数据。

    Args:
        resolution (Tuple[int, int]): 屏幕分辨率 (width, height)
        test_points (List[Tuple[float, float]]): 测试点列表，每个为归一化坐标 (nx, ny)
        rounds (int): 每个测试点重复的轮数，默认为 3

    Returns:
        List[Dict[str, object]]: 包含所有测试记录的列表

    Raises:
        ValueError: 如果分辨率无效或测试点为空
    """
    if len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError(f"无效分辨率: {resolution}")

    if not test_points:
        raise ValueError("测试点列表不能为空")

    calibration_data: List[Dict[str, object]] = []

    for round_num in range(rounds):
        print(f"[第 {round_num + 1}/{rounds} 轮] 开始本轮测试...")
        for nx, ny in test_points:
            try:
                # 执行点击
                expected_x, expected_y = execute_visual_click(nx, ny, resolution)

                # 观察实际点击位置
                time.sleep(0.3)
                observed_x, observed_y = get_observed_position_via_vision()

                # 计算偏差
                delta_x = observed_x - expected_x
                delta_y = observed_y - expected_y

                record: Dict[str, object] = {
                    "resolution": resolution,
                    "normalized_x": nx,
                    "normalized_y": ny,
                    "expected_x": expected_x,
                    "expected_y": expected_y,
                    "observed_x": observed_x,
                    "observed_y": observed_y,
                    "delta_x": delta_x,
                    "delta_y": delta_y
                }
                calibration_data.append(record)
                print(f"[记录] 偏差: Δx={delta_x}, Δy={delta_y}")
                time.sleep(0.5)  # 每次点击间隔

            except Exception as e:
                print(f"[异常] 处理测试点 ({nx}, {ny}) 时发生错误: {e}")
                continue  # 出错时跳过当前点，继续下一个

    return calibration_data


def save_calibration_data(data: List[Dict[str, object]], filename: str) -> None:
    """
    将校准数据保存为 JSON 文件。

    Args:
        data (List[Dict[str, object]]): 待保存的数据
        filename (str): 输出文件名

    Raises:
        IOError: 文件写入失败时抛出
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[保存成功] 数据已写入: {filename}")
    except IOError as e:
        print(f"[保存失败] 无法写入文件 {filename}: {e}")
        raise


def main() -> None:
    """
    主函数：遍历多种分辨率，执行测试，保存偏差数据
    """
    # 测试使用的分辨率列表
    resolutions: List[Tuple[int, int]] = [
        (720, 1280),
        (1080, 1920),
        (1440, 2560),
        (1080, 2340),
        (750, 1334)
    ]

    # 测试点：归一化坐标（覆盖中心、四角、边缘）
    test_points: List[Tuple[float, float]] = [
        (0.5, 0.5),   # 中心
        (0.1, 0.1),   # 左上
        (0.9, 0.1),   # 右上
        (0.1, 0.9),   # 左下
        (0.9, 0.9),   # 右下
        (0.5, 0.1),   # 上中
        (0.5, 0.9),   # 下中
        (0.1, 0.5),   # 左中
        (0.9, 0.5)    # 右中
    ]

    all_data: List[Dict[str, object]] = []

    start_time = time.time()
    try:
        for res in resolutions:
            print(f"\n[开始测试] 分辨率: {res}")
            try:
                data = run_calibration_test(res, test_points, rounds=2)
                all_data.extend(data)
            except Exception as e:
                print(f"[跳过] 分辨率 {res} 测试失败: {e}")
                continue

        # 生成带时间戳的文件名并保存
        timestamp = int(start_time)
        filename = f"calibration_data_{timestamp}.json"
        save_calibration_data(all_data, filename)

        total_records = len(all_data)
        duration = time.time() - start_time
        print(f"\n[完成] 所有测试结束，共收集 {total_records} 条记录，耗时 {duration:.2f} 秒")
        print(f"偏差数据已保存至: {filename}")
        print("下一步建议：使用该数据训练回归模型以预测点击偏移，实现动态校准")

    except KeyboardInterrupt:
        print("\n[中断] 用户终止了测试流程")
    except Exception as e:
        print(f"[严重错误] 主流程异常终止: {e}")


if __name__ == "__main__":
    main()