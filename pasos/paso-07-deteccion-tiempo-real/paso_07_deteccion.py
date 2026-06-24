# ── Standard Library ──────────────────────────────────────────────────────────
from collections import deque
from pathlib import Path
import threading
import time
from typing import Callable

# ── Third-Party ────────────────────────────────────────────────────────────────
import cv2
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np

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

def cargar_modelo(model_path: Path):
    try:
        model = load_model(model_path)
        print(f"Modelo cargado exitosamente desde {model_path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"No se encontro el modelo en {model_path} o {e}")

def setup_landmarker() -> vision.HandLandmarker:
    """Permite hacer inferencias con mediapipe para detectar puntos clave de la mano."""
    base_options = BaseOptions(model_asset_path=str(MP_TASK_PATH))

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # Cambiado a VIDEO para flujos en tiempo real
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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


def predecir_gesto_async(
    model,
    input_sequence: np.ndarray,
    gestures: list[str],
    callback: Callable[[int, float], None]
) -> None:
    """Ejecuta la predicción del modelo LSTM en un hilo secundario de manera asíncrona."""
    input_data = np.expand_dims(input_sequence, axis=0)
    prediction = model(input_data, training=False).numpy()[0]
    gesture_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    callback(gesture_index, confidence)


def main() -> None:
    # PASO 1: Cargar el modelo
    model = cargar_modelo(MODEL_PATH)
    model.summary()  # muestra resumen del modelo
    
    # PASO 2: Cargar los gestos
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
        
        # Guardar tiempo inicial para calcular timestamp_ms incremental
        start_time: float = time.time()
        frame_count: int = 0
        current_gesture: str = ""
        current_confidence: float = 0.0
        prediction_in_progress: bool = False

        def on_prediction_complete(gesture_index: int, confidence: float) -> None:
            nonlocal current_gesture, current_confidence, prediction_in_progress
            print(f"Pred: {gestures[gesture_index]} ({confidence:.4f})")
            if confidence > CONFIDENCE_THRESHOLD:
                current_gesture = gestures[gesture_index]
            else:
                current_gesture = ""
            current_confidence = confidence
            prediction_in_progress = False
        
        # --- Bucle principal de captura ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el cuadro de la cámara.")
                break
                
            # 7: Voltear horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            
            # 8: Convertir BGR -> RGB y envolver en mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 9: Calcular timestamp_ms y realizar detección en modo VIDEO
            timestamp_ms = int((time.time() - start_time) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # 10: Dibujar esqueleto de la mano para visualización
            dibujar_landmarks(frame, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            frame_count += 1
            
            # 12: Implementar la predicción de forma asíncrona cuando la secuencia esté completa
            if len(sequence) == SEQUENCE_LENGTH and not prediction_in_progress:
                prediction_in_progress = True
                # Copiar secuencia para evitar race conditions en hilos concurrentes
                sequence_snapshot = np.array(sequence, dtype=np.float32)
                threading.Thread(
                    target=predecir_gesto_async,
                    args=(model, sequence_snapshot, gestures, on_prediction_complete)
                ).start()
            
            # Mostrar el último gesto detectado persistente en pantalla
            if current_gesture and current_confidence > CONFIDENCE_THRESHOLD:
                cv2.putText(frame, f"{current_gesture} ({current_confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 13: Mostrar frame y esperar tecla de salida ESC (código 27)
            cv2.imshow("GestureFlow Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    # 14: Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
