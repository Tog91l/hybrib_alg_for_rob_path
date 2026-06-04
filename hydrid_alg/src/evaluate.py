import numpy as np
from stable_baselines3 import DQN

from src.config import MODEL_PATH, PATHS_DIR, TRAIN_MAP_PATH, ensure_dirs
from src.env import GridEnv
from src.map_utils import get_start_goal_from_map


def rollout_episode(env: GridEnv, model: DQN):
    grids = []
    path = []

    obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        grids.append(env.big_grid.copy().astype(np.uint8))
        path.append(tuple(env.robot_pos))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)

    grids.append(env.big_grid.copy().astype(np.uint8))
    path.append(tuple(env.robot_pos))
    return grids, path


def compute_metrics_dynamic(path, grids, alpha1=1.0, alpha2=0.5, alpha3=2.0):
    path = np.array(path)

    path_length = 0.0
    for i in range(len(path) - 1):
        path_length += np.linalg.norm(path[i + 1] - path[i])

    J_path = alpha1 * path_length
    steps = len(path)
    J_time = alpha2 * steps

    saf_list = []
    for t in range(len(path)):
        p = path[t]
        grid = grids[t]

        obstacles = np.argwhere(grid == 1)
        if len(obstacles) == 0:
            continue

        dists = np.linalg.norm(obstacles - p, axis=1)
        saf_list.append(np.min(dists))

    J_saf = 0.0 if len(saf_list) == 0 else alpha3 * np.mean(saf_list)
    J_total = J_path + J_time - J_saf

    return {
        "J_total": J_total,
        "J_path": J_path,
        "J_time": J_time,
        "J_saf": J_saf,
    }


def main():
    ensure_dirs()

    nav_map = np.load(TRAIN_MAP_PATH)
    start, goal = get_start_goal_from_map(nav_map)

    env = GridEnv(nav_map, start, goal)
    model = DQN.load(MODEL_PATH)

    grids, path = rollout_episode(env, model)
    metrics = compute_metrics_dynamic(path, grids)

    np.save(PATHS_DIR / "train_eval_path.npy", np.array(path, dtype=np.int32))
    np.save(PATHS_DIR / "train_eval_grids.npy", np.array(grids, dtype=np.uint8))

    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("Steps:", len(path))
    print("Success rate:", env.get_success_rate())


if __name__ == "__main__":
    main()