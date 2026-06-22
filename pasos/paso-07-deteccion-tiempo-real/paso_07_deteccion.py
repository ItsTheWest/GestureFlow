from pathlib import Path
import numpy as np


PROJECT_ROOT         = Path(__file__).resolve().parent.parent.parent
MP_TASK_PATH         = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"
MODEL_PATH           = PROJECT_ROOT / "modelos" / "lstm_gestos.keras"
GESTOS_DIR           = PROJECT_ROOT / "gestos"

SEQUENCE_LENGTH      = 30 # cantidad de frames a procesar
CONFIDENCE_THRESHOLD = 0.8 # confianza minima para mostrar una prediccion

HAND_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
])

def cargar_modelo(modelo:Path):
    pass