# ── Standard Library ──────────────────────────────────────────────────────────
from collections import deque
from pathlib import Path

# ── Third-Party ────────────────────────────────────────────────────────────────
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
from keras.models import load_model


# ── Local ──────────────────────────────────────────────────────────────────────
from utils import extract_keypoints, get_gesture_names

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

def cargar_modelo(model_path:Path):
    try:
        model = load_model(model_path)
        print(f"Modelo cargado exitosamente desde {model_path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"No se encontro el modelo en {model_path} o {e}")

def setup_landmarker() -> vision.HandLandmarker: #permite hacer inferencias con mediapipe para detectar puntos clave de la mano
    base_options = mp.BaseOptions(model_asset_path=str(MP_TASK_PATH)) # Path al modelo de mediapipe

    options = vision.HandLandmarkerOptions(
        base_options=base_options, # Opciones base del modelo
        running_mode=vision.RunningMode.IMAGE, # Modo de ejecucion síncrono para imágenes individuales
        num_hands=2, # Numero de manos a detectar
        min_hand_detection_confidence=0.5, # Confianza minima para detectar una mano
        min_hand_presence_confidence=0.5, # Confianza minima para detectar la presencia de una mano
        min_tracking_confidence=0.5, # Confianza minima para rastrear una mano
    )
    return vision.HandLandmarker.create_from_options(options)


def dibujar_landmarks(frame: np.ndarray, results: vision.HandLandmarkerResult) -> None:
    if not results.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand_landmarks_list in results.hand_landmarks:
       # Guarda las coordenadas de los puntos clave en una lista 
        coords = [
            (int(lm.x * w), int(lm.y * h)) 
            for lm in hand_landmarks_list
        ]
        # Dibuja las conexiones usando las coordenadas precalculadas
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 0), 2)
            
        # Dibuja los puntos clave
        for coord in coords:
            cv2.circle(frame, coord, 3, (0, 0, 255), -1)

def main()-> None:
    # PASO 1: Cargar el modelo
    model = cargar_modelo(MODEL_PATH)
    model.summary()  # muestra resumen del modelo
    #PASO 2: Cargar los gestos
    gestures = get_gesture_names(GESTOS_DIR)
    print(f"Loaded {len(gestures)} gestures: {gestures}") # muestra los gestos cargados
    sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    # PASO 3: Inicializar la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara de video.")
    
    # PASO 4: Inicializar el detector de puntos clave
    with setup_landmarker() as landmarker:
        print("Sistema listo. Iniciando bucle de detección...")
        
        # --- Bucle principal de captura ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el cuadro de la cámara.")
                break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
