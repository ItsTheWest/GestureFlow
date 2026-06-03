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


# ---------------------------------------------------------------------------
# Step 2.2 — Keypoint extraction helper
# ---------------------------------------------------------------------------
def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """
    Flatten the landmarks of the first detected hand into a 1-D array of 63 values.

    Returns:
        np.ndarray of shape (63,): [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        or np.zeros(63) when no hand is detected — keeping data shape consistent.
    """
    if results.hand_landmarks:
        # Only use the first hand detected (index 0)
        hand = results.hand_landmarks[0]
        # Build a flat list: x, y, z for each of the 21 landmarks
        keypoints = []
        for landmark in hand:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints, dtype=np.float32)   # shape: (63,)
    else:
        # No hand visible → return zeros so every frame stays shape (63,)
        return np.zeros(NUM_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 2.3 (helper) — Overlay text on frame for user guidance
# ---------------------------------------------------------------------------
def draw_hud(frame: np.ndarray, gesture: str, sequence: int, frame_num: int, waiting: bool) -> None:
    """Render recording state info directly onto the camera feed."""
    h = frame.shape[0]
    color_accent = (0, 255, 200)
    color_warn   = (0, 200, 255)
    color_label  = (255, 255, 255)

    # Gesture name at the top
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_label, 2)

    # Sequence / total counter
    cv2.putText(frame, f"Secuencia {sequence + 1}/{NUM_SEQUENCES}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_accent, 2)

    if waiting:
        # Pause state: ask user to get ready
        cv2.putText(frame, "PREPARATE...", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_warn, 3)
    else:
        # Active recording state: show frame progress
        cv2.putText(frame, f"Grabando frame {frame_num + 1}/{SEQUENCE_LENGTH}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_accent, 2)
