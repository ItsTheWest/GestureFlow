"""Shared utilities for the gesture detection project."""
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import vision


HAND_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
])


def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """Extract exactly 126 coordinates (63 left + 63 right) from a detection result.

    Missing hands in the frame are represented as zero vectors, guaranteeing
    a fixed-length output regardless of how many hands MediaPipe detects.
    This shape contract must be identical between data collection (paso_05)
    and real-time inference (paso_07) — any divergence silently corrupts predictions.

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

            wrist_x: float = flat_coords[0]
            wrist_y: float = flat_coords[1]
            wrist_z: float = flat_coords[2]

            relative_coords: list[float] = []
            for i in range(0, len(flat_coords), 3):
                relative_coords.extend([
                    flat_coords[i] - wrist_x,
                    flat_coords[i+1] - wrist_y,
                    flat_coords[i+2] - wrist_z
                ])

            if hand_label == "Left":
                left_hand = np.array(relative_coords, dtype=np.float32)
            elif hand_label == "Right":
                right_hand = np.array(relative_coords, dtype=np.float32)

    return np.concatenate([left_hand, right_hand])


def get_gesture_names(base_path: Path) -> list[str]:
    """Return gesture class names sorted alphabetically by folder name.

    The sort order defines the label-to-index mapping used by the model.
    Callers must use this function (not a manual loop) to guarantee
    a consistent order between training and inference.

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
            f"At least 2 gesture classes are required, found {len(names)} in {base_path}"
        )

    return names
