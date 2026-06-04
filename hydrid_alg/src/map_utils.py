import random
from typing import List, Tuple

import numpy as np

from src.config import (
    TRAIN_BIG_SIZE,
    TRAIN_GOAL,
    MOVE_DELTA,
    MOVE_FRACTION,
    TRAIN_NUM_OBSTACLES,
    TRAIN_RECT_H_MAX,
    TRAIN_RECT_H_MIN,
    TRAIN_RECT_W_MAX,
    TRAIN_RECT_W_MIN,
    TRAIN_START,
)

def generate_random_map(
    size: int = TRAIN_BIG_SIZE,
    num_obstacles: int = TRAIN_NUM_OBSTACLES,
    start: Tuple[int, int] = TRAIN_START,
    goal: Tuple[int, int] = TRAIN_GOAL,
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)

    for _ in range(num_obstacles):
        ox = random.randint(0, size - 10)
        oy = random.randint(0, size - 10)
        shape_type = random.choice(["rect", "cross", "dot"])

        if shape_type == "rect":
            w = random.randint(TRAIN_RECT_W_MIN, TRAIN_RECT_W_MAX)
            h = random.randint(TRAIN_RECT_H_MIN, TRAIN_RECT_H_MAX)
            grid[ox : ox + w, oy : oy + h] = 1

        elif shape_type == "cross":
            if 2 <= ox < size - 2 and 2 <= oy < size - 2:
                grid[ox, oy] = 1
                grid[ox + 2, oy] = 1
                grid[ox - 2, oy] = 1
                grid[ox, oy + 2] = 1
                grid[ox, oy - 2] = 1
        else:
            grid[ox, oy] = 1

    grid[start] = 2
    grid[goal] = 3
    return grid


def get_start_goal_from_map(grid: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    start = tuple(np.argwhere(grid == 2)[0])
    goal = tuple(np.argwhere(grid == 3)[0])
    return start, goal


def get_local_window(grid: np.ndarray, robot_pos, window_size: int) -> np.ndarray:
    half = window_size // 2
    x, y = robot_pos

    x_min = max(0, x - half)
    x_max = min(grid.shape[0], x + half)

    y_min = max(0, y - half)
    y_max = min(grid.shape[1], y + half)

    window = np.zeros((window_size, window_size), dtype=grid.dtype)
    sub = grid[x_min:x_max, y_min:y_max]

    wx = half - (x - x_min)
    wy = half - (y - y_min)

    window[wx : wx + sub.shape[0], wy : wy + sub.shape[1]] = sub
    return window


def get_local_obs(window: np.ndarray, radius: int) -> List[Tuple[int, int]]:
    center_x = window.shape[0] // 2
    center_y = window.shape[1] // 2

    local_obs = []
    for i in range(center_x - radius, center_x + radius + 1):
        for j in range(center_y - radius, center_y + radius + 1):
            if 0 <= i < window.shape[0] and 0 <= j < window.shape[1]:
                if window[i, j] == 1:
                    local_obs.append((i, j))
    return local_obs


def get_local_map_features(window: np.ndarray, radius: int) -> np.ndarray:
    center = window.shape[0] // 2
    local = window[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    local = (local == 1).astype(np.float32)
    return local.flatten()


def move_obstacles(grid: np.ndarray, start, goal, robot_pos=None) -> np.ndarray:
    new_grid = grid.copy()
    obstacles = np.argwhere(grid == 1)

    if len(obstacles) == 0:
        return new_grid

    move_count = max(1, int(len(obstacles) * MOVE_FRACTION))
    idx = np.random.choice(len(obstacles), move_count, replace=False)

    robot_pos_tuple = tuple(robot_pos) if robot_pos is not None else None

    for i in idx:
        ox, oy = obstacles[i]

        dx = random.randint(-MOVE_DELTA, MOVE_DELTA)
        dy = random.randint(-MOVE_DELTA, MOVE_DELTA)

        nx = ox + dx
        ny = oy + dy

        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if (
                new_grid[nx, ny] == 0
                and (nx, ny) != start
                and (nx, ny) != goal
                and (robot_pos_tuple is None or (nx, ny) != robot_pos_tuple)
            ):
                new_grid[nx, ny] = 1
                new_grid[ox, oy] = 0

    return new_grid