import random

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from src.config import (
    MODEL_PATH,
    PATHS_DIR,
    TEST_MAP_PATH,
    TEST_BIG_SIZE,
    TEST_NUM_OBSTACLES,
    TEST_START,
    TEST_GOAL,
)
from src.env import GridEnv
from src.map_utils import get_start_goal_from_map


def generate_test_map( size=TEST_BIG_SIZE, num_obstacles=TEST_NUM_OBSTACLES, start=TEST_START, goal=TEST_GOAL):
    grid = np.zeros((size, size), dtype=np.uint8)

    for _ in range(num_obstacles):
        ox = random.randint(0, size - 10)
        oy = random.randint(0, size - 10)
        shape_type = random.choice(["rect", "cross", "dot"])

        if shape_type == "rect":
            w = random.randint(2, 25)
            h = random.randint(1, 35)
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


def main():

    test_map = generate_test_map()
    np.save(TEST_MAP_PATH, test_map)

    plt.figure(figsize=(8, 8))
    plt.imshow(test_map, origin="lower")
    plt.scatter(TEST_START[1], TEST_START[0], c="green", s=50)
    plt.scatter(TEST_GOAL[1], TEST_GOAL[0], c="blue", s=50)
    plt.title("Generated test map")
    plt.show()

    start, goal = get_start_goal_from_map(test_map)

    test_env = GridEnv(test_map, start, goal, max_steps=23500)
    model = DQN.load(MODEL_PATH)

    obs, _ = test_env.reset()
    done = False
    truncated = False
    path = []
    grids = []

    while not done and not truncated:
        grids.append(test_env.big_grid.copy().astype(np.uint8))
        path.append(tuple(test_env.robot_pos))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = test_env.step(action)

    grids.append(test_env.big_grid.copy().astype(np.uint8))
    path.append(tuple(test_env.robot_pos))

    np.save(PATHS_DIR / "test_path.npy", np.array(path, dtype=np.int32))
    np.save(PATHS_DIR / "test_grids.npy", np.array(grids, dtype=np.uint8))

    print("Steps:", len(path))
    print("Success rate:", test_env.get_success_rate())
    print("Final position:", test_env.robot_pos)
    print("Goal:", goal)
    print("Reached goal:", tuple(test_env.robot_pos) == goal)

    goal_dist = np.linalg.norm(np.array(test_env.robot_pos) - np.array(goal))
    print("Distance to goal:", goal_dist)
    print("Steps used:", test_env.steps)


if __name__ == "__main__":
    main()