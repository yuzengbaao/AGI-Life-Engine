#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单任务环境：GridWorld（网格世界）
为统一AGI系统提供真实的学习和测试环境

功能：
1. 8x8网格世界
2. 智能体从起点移动到终点
3. 障碍物和陷阱
4. 明确的奖励信号

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """动作类型"""
    UP = 0      # 向上
    DOWN = 1    # 向下
    LEFT = 2    # 向左
    RIGHT = 3   # 向右


@dataclass
class GridWorldState:
    """GridWorld状态"""
    agent_pos: Tuple[int, int]  # 智能体位置 (x, y)
    goal_pos: Tuple[int, int]   # 目标位置 (x, y)
    grid_size: int = 8          # 网格大小
    obstacles: List[Tuple[int, int]] = None  # 障碍物列表
    traps: List[Tuple[int, int]] = None         # 陷阱列表

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []
        if self.traps is None:
            self.traps = []


class GridWorld:
    """
    网格世界环境

    8x8网格，智能体需要从起点移动到终点
    - 到达终点：+10奖励
    - 遇到障碍：-1奖励（不能移动）
    - 遇到陷阱：-5奖励
    - 每步：-0.1奖励（鼓励快速到达）
    """

    def __init__(
        self,
        grid_size: int = 8,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (7, 7),
        obstacles: List[Tuple[int, int]] = None,
        traps: List[Tuple[int, int]] = None,
        stochastic: bool = False  # 是否启用随机性
    ):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles or []
        self.traps = traps or []
        self.stochastic = stochastic

        # 当前状态
        self.agent_pos = start_pos
        self.done = False
        self.steps = 0
        self.max_steps = 100

        # 状态表示（64维向量，适配系统）
        self.state_dim = 64

        logger.info(f"[GridWorld] 初始化: {grid_size}x{grid_size}网格, "
                   f"起点{start_pos}, 终点{goal_pos}")

    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        self.agent_pos = self.start_pos
        self.done = False
        self.steps = 0

        state = self._get_state_vector()
        logger.debug(f"[GridWorld] 重置: 位置={self.agent_pos}")
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作

        Args:
            action: 动作 (0=上, 1=下, 2=左, 3=右)

        Returns:
            state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.done:
            logger.warning("[GridWorld] 环境已结束，请先reset")
            return self._get_state_vector(), 0.0, True, {}

        self.steps += 1
        reward = 0.0
        info = {}

        # 当前位置
        x, y = self.agent_pos

        # 执行动作
        action_type = ActionType(action)
        new_x, new_y = x, y

        if action_type == ActionType.UP:
            new_y = max(0, y - 1)
        elif action_type == ActionType.DOWN:
            new_y = min(self.grid_size - 1, y + 1)
        elif action_type == ActionType.LEFT:
            new_x = max(0, x - 1)
        elif action_type == ActionType.RIGHT:
            new_x = min(self.grid_size - 1, x + 1)

        # 检查障碍物
        if (new_x, new_y) in self.obstacles:
            # 遇到障碍物，不能移动
            reward = -1.0
            info['event'] = 'obstacle'
            logger.debug(f"[GridWorld] 遇到障碍物: ({new_x}, {new_y})")
        else:
            # 移动到新位置
            self.agent_pos = (new_x, new_y)

            # 检查事件
            if (new_x, new_y) == self.goal_pos:
                # 到达终点
                reward = 10.0
                self.done = True
                info['event'] = 'goal'
                info['success'] = True
                logger.info(f"[GridWorld] [GOAL] 到达终点! 步数={self.steps}")
            elif (new_x, new_y) in self.traps:
                # 遇到陷阱
                reward = -5.0
                info['event'] = 'trap'
                logger.debug(f"[GridWorld] [TRAP] 遇到陷阱: ({new_x}, {new_y})")
            else:
                # 普通移动
                reward = -0.1  # 每步小惩罚，鼓励快速到达

        # 检查是否超时
        if self.steps >= self.max_steps:
            self.done = True
            info['event'] = 'timeout'
            logger.debug(f"[GridWorld] [TIMEOUT] 超时: {self.steps}步")

        # 更新状态
        state = self._get_state_vector()
        info['steps'] = self.steps
        info['position'] = self.agent_pos

        return state, reward, self.done, info

    def _get_state_vector(self) -> np.ndarray:
        """
        获取状态向量（64维）

        编码（紧凑型，适合64维）：
        - 位置0-1：智能体位置归一化 (x/7, y/7)
        - 位置2-3：目标位置归一化 (x/7, y/7)
        - 位置4-5：相对方向 (dx/7, dy/7)
        - 位置6：曼哈顿距离归一化
        - 位置7：步数进度
        - 位置8-63：保留用于扩展（初始化为0）
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # 智能体位置归一化（0-1）
        state[0] = self.agent_pos[0] / (self.grid_size - 1)
        state[1] = self.agent_pos[1] / (self.grid_size - 1)

        # 目标位置归一化（0-1）
        state[2] = self.goal_pos[0] / (self.grid_size - 1)
        state[3] = self.goal_pos[1] / (self.grid_size - 1)

        # 相对方向（归一化到[-1, 1]）
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]
        state[4] = dx / (self.grid_size - 1)
        state[5] = dy / (self.grid_size - 1)

        # 曼哈顿距离归一化（0-1）
        manhattan = abs(dx) + abs(dy)
        max_manhattan = 2 * (self.grid_size - 1)  # 最大可能的曼哈顿距离
        state[6] = manhattan / max_manhattan

        # 步数进度（0-1）
        state[7] = self.steps / self.max_steps

        # 位置8-63保留用于扩展（当前为0）

        return state

    def get_manhattan_distance(self) -> int:
        """获取曼哈顿距离"""
        return abs(self.goal_pos[0] - self.agent_pos[0]) + abs(self.goal_pos[1] - self.agent_pos[1])

    def render(self) -> str:
        """渲染当前状态（ASCII艺术）"""
        grid = []
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                if (x, y) == self.agent_pos:
                    row += "@  "  # 智能体
                elif (x, y) == self.goal_pos:
                    row += "G  "  # 目标
                elif (x, y) in self.obstacles:
                    row += "#  "  # 障碍物
                elif (x, y) in self.traps:
                    row += "!  "  # 陷阱
                else:
                    row += ".  "  # 空地
            grid.append(row)

        # 添加信息
        info = f"\n步数: {self.steps}/{self.max_steps} | 曼哈顿距离: {self.get_manhattan_distance()}"
        grid.append(info)

        return "\n".join(grid)

    def get_optimal_action(self) -> int:
        """获取最优动作（用于评估）"""
        x, y = self.agent_pos
        gx, gy = self.goal_pos

        # 计算每个动作的曼哈顿距离
        distances = {}
        distances[ActionType.UP] = abs(x - gx) + abs(max(0, y - 1) - gy)
        distances[ActionType.DOWN] = abs(x - gx) + abs(min(self.grid_size - 1, y + 1) - gy)
        distances[ActionType.LEFT] = abs(max(0, x - 1) - gx) + abs(y - gy)
        distances[ActionType.RIGHT] = abs(min(self.grid_size - 1, x + 1) - gx) + abs(y - gy)

        # 返回距离最小的动作
        best_action = min(distances, key=distances.get)
        return best_action.value


# 预设场景
def create_simple_gridworld() -> GridWorld:
    """创建简单场景（无障碍物）"""
    return GridWorld(
        grid_size=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=[],
        traps=[],
        stochastic=False
    )


def create_medium_gridworld() -> GridWorld:
    """创建中等场景（有障碍物）"""
    obstacles = [(3, 3), (3, 4), (4, 3), (4, 4)]  # 中央障碍
    traps = [(1, 1), (6, 6)]  # 陷阱

    return GridWorld(
        grid_size=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
        traps=traps,
        stochastic=False
    )


def create_complex_gridworld() -> GridWorld:
    """创建复杂场景（随机障碍物）"""
    np.random.seed(42)

    # 随机生成障碍物（避开起点和终点）
    obstacles = []
    for _ in range(15):
        ox = np.random.randint(0, 8)
        oy = np.random.randint(0, 8)
        if (ox, oy) not in [(0, 0), (7, 7)]:  # 避开起点和终点
            obstacles.append((ox, oy))

    return GridWorld(
        grid_size=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
        traps=[],
        stochastic=True
    )


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("[测试] GridWorld环境")
    print("="*60)

    # 测试简单场景
    print("\n[测试1] 简单场景")
    env = create_simple_gridworld()

    print("\n初始状态:")
    print(env.render())

    print("\n执行10步随机动作:")
    for i in range(10):
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        print(f"  步{i+1}: 动作={action.name}, 奖励={reward:.1f}, "
              f"位置={info['position']}, 事件={info.get('event', 'move')}")
        if done:
            break

    print(f"\n最终状态:")
    print(env.render())

    # 测试中等场景
    print("\n" + "="*60)
    print("[测试2] 中等场景")
    env = create_medium_gridworld()

    print("\n初始状态:")
    print(env.render())

    print("\n执行20步随机动作:")
    total_reward = 0.0
    for i in range(20):
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"  步{i+1}: 完成！事件={info.get('event')}")
            break

    print(f"\n最终状态:")
    print(env.render())
    print(f"累计奖励: {total_reward:.1f}")

    print("\n" + "="*60)
    print("[成功] GridWorld测试完成")
    print("="*60)
