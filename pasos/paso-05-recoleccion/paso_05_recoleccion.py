from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Path resolution — same pattern as previous steps
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH   = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH  = 30   # Frames per sequence (~1 second at 30 fps)
NUM_FEATURES     = 126  # 42 landmarks (21 per hand) × 3 coordinates (x, y, z)
NUM_SEQUENCES    = 30   # How many examples we collect per gesture

# How often (in frames) we auto-save a sequence.
# With SEQUENCE_LENGTH=30 and SAVE_EVERY=15 there is 50% overlap:
# this generates more diversity in the training data.
SAVE_EVERY       = 15

# Seconds of countdown before automatic recording begins
COUNTDOWN_SECS   = 3

# Frames the confirmation flash lasts in the HUD (~0.5 s at 30 fps)
FLASH_DURATION   = 15


# ---------------------------------------------------------------------------
# Step 1 — MediaPipe configuration (IMAGE mode = synchronous, no callback)
# ---------------------------------------------------------------------------
def build_landmarker() -> vision.HandLandmarker:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH)) 

    options = vision.HandLandmarkerOptions(
       base_options=base_options,
       running_mode=vision.RunningMode.IMAGE,  
       num_hands=2,                         
    )

    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Step 2 — Keypoint extraction helper
# ---------------------------------------------------------------------------
# def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """
    Flatten the landmarks of the first detected hand into a 1-D array of 63 values.

    Returns:
        np.ndarray of shape (63,): [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        or np.zeros(63) if no hand is visible — keeps shape constant across frames.
    """
    # TODO: Check if results.hand_landmarks has any entries
    # TODO: If yes, grab the first hand and iterate its 21 landmarks
    # TODO: Collect [x, y, z] for each landmark into a flat list
    # TODO: Return as np.float32 array of shape (63,)
    # TODO: If no hand detected, return np.zeros(NUM_FEATURES, dtype=np.float32)
    pass  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 3a — HUD rendering for the waiting phase (before SPACE)
# ---------------------------------------------------------------------------
def draw_waiting(frame: np.ndarray, gesture: str, saved: int) -> None:
    """Show the waiting state: camera active but not collecting yet."""
    # TODO: Get frame dimensions (h, w)
    # TODO: Draw gesture name at top-left
    # TODO: Draw saved count (saved/NUM_SEQUENCES)
    # TODO: Draw centered instruction "Press SPACE to start"
    # TODO: Draw "Q: quit" at bottom-left
    pass


# ---------------------------------------------------------------------------
# Step 3b — HUD rendering for the countdown phase
# ---------------------------------------------------------------------------
def draw_countdown(frame: np.ndarray, gesture: str, seconds_left: int) -> None:
    """Show the gesture name and the countdown number centered on screen."""
    # TODO: Create a semi-transparent dark overlay for readability
    # TODO: Draw gesture name at top-left
    # TODO: Draw the countdown number large and centered
    # TODO: Draw "Prepare your gesture..." at the bottom
    pass


# ---------------------------------------------------------------------------
# Step 4 — HUD rendering during automatic recording
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    gesture: str,
    saved: int,
    buffer_len: int,
    frame_counter: int,
    hand_detected: bool,
    flash_timer: int,
) -> None:
    """
    Show recording status overlay on the camera frame.

    Elements:
      · Gesture name and saved sequence counter (top)
      · Hand detection indicator
      · Rolling buffer progress bar with next auto-save marker
      · Confirmation flash when a sequence is saved
    """
    # TODO: Get frame dimensions (h, w), define color constants
    # TODO: Draw gesture name and saved count at top
    # TODO: Draw hand detection indicator (green if detected, red if not)
    # TODO: Draw a progress bar showing buffer fill percentage
    # TODO: When buffer is full, draw a vertical marker showing SAVE_EVERY cycle
    # TODO: Draw flash message or default instruction at bottom
    pass


# ---------------------------------------------------------------------------
# Step 5 — Output folder creation with resume support
# ---------------------------------------------------------------------------
#def pedir_nombre_gesto() -> tuple[str, int] | tuple[None, None]:
    """
    Ask for the gesture name, create the folder, and detect how many sequences
    already exist so we can resume from the correct index without overwriting.

    Returns:
        (normalized_name, next_index) or (None, None) on error.
    """
    # TODO: Read gesture name from input(), normalize (strip + lowercase)
    # TODO: Validate non-empty
    # TODO: Create the folder at PROJECT_ROOT / "gestos" / normalized
    # TODO: Count existing .npy files to determine the resume index
    # TODO: Print resume info if files exist, else print "folder ready"
    # TODO: Return (normalized, next_index) or (None, None) on error
    pass  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 6 — Main loop: wait → countdown → automatic recording
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, start_index: int, landmarker: vision.HandLandmarker) -> None:
    """
    Phase 0 — Wait: camera is active but does NOT collect until SPACE is pressed.
    Phase 1 — Countdown: displays 3-2-1 so the user can position their hand.
    Phase 2 — Automatic recording: the rolling buffer saves a sequence every
              SAVE_EVERY frames when a hand is detected, no further input needed.

    File naming convention: gestos/<gesture>/<sequence_index>.npy
    """
    # TODO: Setup — compute output_dir, sequences_saved, sequences_needed
    # TODO: Open camera (cv2.VideoCapture), set resolution, create window

    # TODO: Phase 0 — Wait loop: read frames, draw_waiting(), show frame
    #       Break on SPACE, quit on Q

    # TODO: Phase 1 — Countdown loop: for each second, draw_countdown()
    #       Quit on Q during countdown

    # TODO: Phase 2 — Recording loop:
    #       Create deque(maxlen=SEQUENCE_LENGTH) as rolling buffer
    #       Initialize frame_counter and flash_timer
    #       Loop while sequences_saved < NUM_SEQUENCES:
    #         - Read frame, flip, convert BGR→RGB, run landmarker.detect()
    #         - Extract keypoints, append to buffer, increment frame_counter
    #         - Auto-save condition: buffer full + hand visible + frame_counter % SAVE_EVERY == 0
    #         - Save as .npy, increment sequences_saved, trigger flash
    #         - Draw HUD, show frame, handle Q key

    # TODO: Cleanup — release camera, destroy windows, print summary
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
#if __name__ == "__main__":
    # Step 1: Ask for gesture name and detect resume index
    #gesto_creado, next_index = pedir_nombre_gesto()
    #if gesto_creado is None or next_index is None:
        #exit(1)

    # Step 2: Build the MediaPipe HandLandmarker (IMAGE mode)
    #with build_landmarker() as landmarker:
        # Step 3: Wait → countdown → automatic recording
        #grabar_gesto(gesto_creado, next_index, landmarker)
