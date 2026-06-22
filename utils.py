"""Shared utilities for GestureFlow training and inference scripts.

Functions here are used by more than one pipeline stage (data collection,
training, real-time detection).  Any change here propagates automatically
to all callers — do NOT copy-paste these functions into individual scripts.
"""
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import vision


def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """Extract exactly 126 coordinates (63 left + 63 right) from a detection result.

    Hands absent from the frame are represented as zero vectors, guaranteeing
    a fixed-length output regardless of how many hands MediaPipe detected.
    This shape contract must remain identical between data collection (paso_05)
    and real-time inference (paso_07) — any divergence silently corrupts
    predictions.

    Args:
        results: Result object returned by HandLandmarker.detect().

    Returns:
        np.ndarray of shape (126,) and dtype float32.
    """
    left_hand  = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.hand_landmarks and results.handedness:
        for idx, hand_info in enumerate(results.handedness):
            hand_label = hand_info[0].category_name
            landmarks  = results.hand_landmarks[idx]

            flat_coords: list[float] = []
            for lm in landmarks:
                flat_coords.extend([lm.x, lm.y, lm.z])

            if hand_label == "Left":
                left_hand = np.array(flat_coords, dtype=np.float32)
            elif hand_label == "Right":
                right_hand = np.array(flat_coords, dtype=np.float32)

    return np.concatenate([left_hand, right_hand])


def get_gesture_names(base_path: Path) -> list[str]:
    """Return gesture class names sorted alphabetically by folder name.

    The sort order defines the label-to-index mapping used by the model.
    Callers must use this function (not a hand-rolled loop) to guarantee
    consistent ordering between training and inference.

    Args:
        base_path: Directory whose immediate subdirectories are gesture classes.

    Returns:
        Sorted list of gesture folder names.

    Raises:
        FileNotFoundError: If base_path does not exist.
        ValueError: If fewer than 2 gesture classes are found.
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Gesture directory not found: {base_path}")

    names = sorted(p.name for p in base_path.iterdir() if p.is_dir())

    if len(names) < 2:
        raise ValueError(
            f"At least 2 gesture classes required, found {len(names)} in {base_path}"
        )

    return names
