import numpy as np

from src.config import ETA, K_ATT, REP_RADIUS


def attract_force(pos, goal, k: float = K_ATT) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    return k * (goal - pos)


def rep_force(pos, obstacles, eta: float = ETA, rep_radius: float = REP_RADIUS) -> np.ndarray:
    if len(obstacles) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    pos = np.asarray(pos, dtype=np.float32)
    obs = np.asarray(obstacles, dtype=np.float32)

    diffs = pos - obs
    dists = np.linalg.norm(diffs, axis=1)

    mask = (dists < rep_radius) & (dists > 1e-5)
    if not np.any(mask):
        return np.array([0.0, 0.0], dtype=np.float32)

    diffs = diffs[mask]
    dists = dists[mask]

    directions = diffs / dists[:, None]
    magnitudes = eta * (1.0 / dists - 1.0 / rep_radius) / (dists ** 2)

    forces = magnitudes[:, None] * directions
    return np.sum(forces, axis=0).astype(np.float32)


def total_force(pos, goal, obstacles) -> np.ndarray:
    return attract_force(pos, goal) + rep_force(pos, obstacles)


def normalize_force(force: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(force)
    if norm > 0:
        return (force / norm).astype(np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)