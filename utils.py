"""Shared utilities for GestureFlow training and inference scripts.

Functions here are used by more than one pipeline stage (data collection,
training, real-time detection).  Any change here propagates automatically
to all callers — do NOT copy-paste these functions into individual scripts.
"""
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import vision


def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """extrae exactamente 126 coordenadas (63 izquierda + 63 derecha) de un resultado de detección.

    Las manos ausentes en el frame se representan como vectores cero, garantizando
    una salida de longitud fija sin importar cuántas manos detecte MediaPipe.
    Este contrato de forma debe ser idéntico entre la recopilación de datos (paso_05)
    y la inferencia en tiempo real (paso_07) — cualquier divergencia corrompe silenciosamente
    las predicciones.

    Args:
        results: Objeto de resultado devuelto por HandLandmarker.detect().

    Returns:
        np.ndarray de forma (126,) y dtype float32. 
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
    """retorna nombres de clases de gestos ordenados alfabéticamente por nombre de carpeta.

    El orden de clasificación define el mapeo de etiquetas a índices utilizado por el modelo.
    Los llamadores deben usar esta función (no un bucle manual) para garantizar
    un orden consistente entre el entrenamiento y la inferencia.

    Args:
        base_path: Directorio cuyas subcarpetas inmediatas son clases de gestos.

    Returns:
        Lista ordenada de nombres de carpetas de gestos.

    Raises:
        FileNotFoundError: Si base_path no existe.
        ValueError: Si se encuentran menos de 2 clases de gestos.
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Directorio de gestos no encontrado: {base_path}")

    names = sorted(p.name for p in base_path.iterdir() if p.is_dir())

    if len(names) < 2:
        raise ValueError(
            f"Se requieren al menos 2 clases de gestos, se encontraron {len(names)} en {base_path}"
        )

    return names
