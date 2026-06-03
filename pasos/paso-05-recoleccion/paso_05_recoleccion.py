import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Path resolution — mirrors the pattern used in paso_04_vocales.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30   # Number of frames per recorded sequence (temporal window)
NUM_FEATURES   = 63    # 21 hand landmarks * 3 coordinates (x, y, z)
NUM_SEQUENCES  = 30    # How many example sequences to record per gesture


# ---------------------------------------------------------------------------
# Step 2.1 — MediaPipe configuration (IMAGE mode for synchronous processing)
# ---------------------------------------------------------------------------
# We use RunningMode.IMAGE instead of LIVE_STREAM so every detect() call
# blocks until MediaPipe returns the result. This guarantees we capture
# exactly SEQUENCE_LENGTH frames without any being dropped asynchronously.
def build_landmarker() -> vision.HandLandmarker:
    """Create and return a HandLandmarker configured for synchronous IMAGE mode."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo no encontrado en: {MODEL_PATH}\n"
            "Asegúrate de haber descargado 'hand_landmarker.task' en la carpeta 'prueba/'."
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Synchronous — no callback needed
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)
