"""Centralized configuration settings for the GestureFlow project."""
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
MP_TASK_PATH: Path = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"
MODEL_PATH: Path = PROJECT_ROOT / "modelos" / "lstm_gestos.keras"
GESTOS_DIR: Path = PROJECT_ROOT / "gestos"

# ── Data Collection & Processing Settings ──────────────────────────────────────
SEQUENCE_LENGTH: int = 30
NUM_FEATURES: int = 126  # 21 landmarks * 3 coordinates (x,y,z) * 2 hands
NUM_SEQUENCES: int = 200
SAVE_EVERY: int = 5
COUNTDOWN_SECS: int = 5
FLASH_DURATION: int = 15

# ── Model Training Settings ────────────────────────────────────────────────────
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
EPOCHS: int = 100
BATCH_SIZE: int = 32

# ── Inference Settings ─────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.8

# ── Step 8: System Gesture Control ────────────────────────────────────────────
PINCH_THRESHOLD: float   = 0.06
PINCH_MIN_FRAMES: int    = 3
SWIPE_VELOCITY: float    = 0.035
SWIPE_FRAMES: int        = 8
SWIPE_COOLDOWN: float    = 1.5
CURSOR_SMOOTHING: float  = 0.25
