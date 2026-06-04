import numpy as np
from stable_baselines3 import DQN

from src.config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    EXPLORATION_FINAL_EPS,
    EXPLORATION_FRACTION,
    GAMMA,
    LEARNING_RATE,
    LEARNING_STARTS,
    MODEL_PATH,
    TARGET_UPDATE_INTERVAL,
    TOTAL_TIMESTEPS,
    TRAIN_FREQ,
    TRAIN_MAP_PATH,
)
from src.env import GridEnv
from src.map_utils import get_start_goal_from_map


def main():

    nav_map = np.load(TRAIN_MAP_PATH)
    start, goal = get_start_goal_from_map(nav_map)

    env = GridEnv(nav_map, start, goal)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        train_freq=TRAIN_FREQ,
        gamma=GAMMA,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        tensorboard_log=str((MODEL_PATH.parent.parent / "outputs" / "logs").resolve()),
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_PATH)

    metrics = env.get_metrics()
    print("TRAINING METRICS")
    if metrics is not None:
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

    print("Success rate:", env.get_success_rate())


if __name__ == "__main__":
    main()