from pathlib import Path

# ---------------- Paths ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
MAPS_DIR = ROOT_DIR / "maps"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PATHS_DIR = OUTPUTS_DIR / "paths"
#ANIMATIONS_DIR = OUTPUTS_DIR / "animations"
LOGS_DIR = OUTPUTS_DIR / "logs"

MODEL_PATH = MODEL_DIR / "dqn_apf_hybrid_right_s.zip"
TRAIN_MAP_PATH = MAPS_DIR / "train_map.npy"
TEST_MAP_PATH = MAPS_DIR / "test_map.npy"

# ---------------- Train map ----------------
TRAIN_BIG_SIZE = 1000
TRAIN_NUM_OBSTACLES = 450
TRAIN_RECT_W_MIN = 2
TRAIN_RECT_W_MAX = 55
TRAIN_RECT_H_MIN = 1
TRAIN_RECT_H_MAX = 45
TRAIN_START = (3, 1)
TRAIN_GOAL = (999, 998)

# ---------------- Test map ----------------
TEST_BIG_SIZE = 500
TEST_NUM_OBSTACLES = 500
TEST_RECT_W_MIN = 2
TEST_RECT_W_MAX = 25
TEST_RECT_H_MIN = 1
TEST_RECT_H_MAX = 35
TEST_START = (3, 1)
TEST_GOAL = (499, 498)

# ---------------- Local perception ----------------
WINDOW_SIZE = 100
LOCAL_OBS_RADIUS = 35
LOCAL_FEATURE_RADIUS = 25

# 6 scalar features + (2*LOCAL_FEATURE_RADIUS+1)^2 local map cells
OBS_SIZE = 6 + (2 * LOCAL_FEATURE_RADIUS + 1) ** 2

# ---------------- APF ----------------
K_ATT = 2.75
ETA = 90.0
REP_RADIUS = 3.0

# ---------------- Dynamic obstacles ----------------
MOVE_EVERY_N_STEPS = 25
MOVE_FRACTION = 0.15
MOVE_DELTA = 3

# ---------------- RL env ----------------
MAX_STEPS = 6500
ACTION_COUNT = 8

STEP_PENALTY = 0.05
COLLISION_PENALTY = 1.0
STAY_PENALTY = 0.05
PROGRESS_SCALE = 4.0
GOAL_REWARD = 300.0
TRUNCATION_PENALTY = 10.0

# ---------------- DQN ----------------
LEARNING_RATE = 3e-4
BUFFER_SIZE = 400_000
LEARNING_STARTS = 5_000
BATCH_SIZE = 128
TRAIN_FREQ = 8
GAMMA = 0.97
EXPLORATION_FRACTION = 0.35
EXPLORATION_FINAL_EPS = 0.05
TARGET_UPDATE_INTERVAL = 4000
TOTAL_TIMESTEPS = 3_000_000


def ensure_dirs() -> None:
    for p in [MODEL_DIR, MAPS_DIR, OUTPUTS_DIR, PATHS_DIR, LOGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)