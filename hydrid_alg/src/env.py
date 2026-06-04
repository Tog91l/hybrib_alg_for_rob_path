from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.apf import normalize_force, total_force
from src.config import (
    ACTION_COUNT,
    COLLISION_PENALTY,
    GOAL_REWARD,
    LOCAL_FEATURE_RADIUS,
    LOCAL_OBS_RADIUS,
    MAX_STEPS,
    MOVE_EVERY_N_STEPS,
    OBS_SIZE,
    PROGRESS_SCALE,
    STAY_PENALTY,
    STEP_PENALTY,
    TRUNCATION_PENALTY,
    WINDOW_SIZE,
)
from src.map_utils import (
    get_local_map_features,
    get_local_obs,
    get_local_window,
    move_obstacles,
)


class GridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid: np.ndarray, start, goal, max_steps: Optional[int] = None):
        super().__init__()

        self.original_grid = grid.copy()
        self.big_grid = grid.copy()

        self.start = tuple(start)
        self.goal = tuple(goal)

        self.size = grid.shape[0]
        self.max_steps = max_steps if max_steps is not None else MAX_STEPS

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_COUNT)

        self.success_count = 0
        self.episode_count = 0

        self.metrics = {
            "J_path": [],
            "J_time": [],
            "J_saf": [],
            "J_total": [],
            "steps": [],
        }

        self.current_J_path = 0.0
        self.current_J_time = 0.0
        self.current_J_saf = 0.0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.steps > 0:
            j_total = self.current_J_path + self.current_J_time + self.current_J_saf
            self.metrics["J_path"].append(self.current_J_path)
            self.metrics["J_time"].append(self.current_J_time)
            self.metrics["J_saf"].append(self.current_J_saf)
            self.metrics["J_total"].append(j_total)
            self.metrics["steps"].append(self.steps)

        self.big_grid = self.original_grid.copy()
        self.robot_pos = [self.start[0], self.start[1]]
        self.steps = 0

        self.current_J_path = 0.0
        self.current_J_time = 0.0
        self.current_J_saf = 0.0

        self.episode_count += 1
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        local_map = get_local_window(self.big_grid, self.robot_pos, WINDOW_SIZE)
        local_obs = get_local_obs(local_map, radius=LOCAL_OBS_RADIUS)

        obs_global = []
        half_window = local_map.shape[0] // 2  # важная правка

        for ox_local, oy_local in local_obs:
            gx = self.robot_pos[0] - half_window + ox_local
            gy = self.robot_pos[1] - half_window + oy_local
            obs_global.append((gx, gy))

        force = total_force(self.robot_pos, self.goal, obs_global)
        direction = normalize_force(force)

        dx = (self.goal[0] - self.robot_pos[0]) / self.size
        dy = (self.goal[1] - self.robot_pos[1]) / self.size

        local_features = get_local_map_features(local_map, radius=LOCAL_FEATURE_RADIUS)

        obs = np.concatenate(
            [
                np.array(
                    [
                        (self.robot_pos[0] / self.size) * 2 - 1,
                        (self.robot_pos[1] / self.size) * 2 - 1,
                        dx,
                        dy,
                        direction[0],
                        direction[1],
                    ],
                    dtype=np.float32,
                ),
                local_features,
            ]
        )
        return obs.astype(np.float32)

    def step(self, action: int):
        if self.steps > 0 and self.steps % MOVE_EVERY_N_STEPS == 0:
            self.big_grid = move_obstacles(
                self.big_grid,
                self.start,
                self.goal,
                robot_pos=self.robot_pos,
            )

        x, y = self.robot_pos
        prev_dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal))

        moves = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        dx, dy = moves[action]

        nx = x + dx
        ny = y + dy

        self.steps += 1
        done = False
        truncated = False

        reward = -STEP_PENALTY
        self.current_J_time += 1

        if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size or self.big_grid[nx, ny] == 1:
            reward -= COLLISION_PENALTY
            nx, ny = x, y
            self.current_J_saf += 1

        if (nx, ny) == (x, y):
            reward -= STAY_PENALTY
        else:
            self.current_J_path += float(np.linalg.norm([dx, dy]))
            new_dist = np.linalg.norm(np.array([nx, ny]) - np.array(self.goal))
            progress = prev_dist - new_dist
            reward += progress * PROGRESS_SCALE

            if (nx, ny) == self.goal:
                reward += GOAL_REWARD
                done = True
                self.success_count += 1

        if self.steps > self.max_steps:
            truncated = True
            reward -= TRUNCATION_PENALTY

        self.robot_pos = [nx, ny]
        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def get_success_rate(self) -> float:
        if self.episode_count == 0:
            return 0.0
        return self.success_count / self.episode_count

    def get_metrics(self):
        if len(self.metrics["steps"]) == 0:
            return None

        return {key: float(np.mean(values)) for key, values in self.metrics.items()}