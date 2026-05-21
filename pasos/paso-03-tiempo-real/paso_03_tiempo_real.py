# --- Librerías ---
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# Ancho para inferencia (más bajo = menos retardo en CPU). Landmarks son 0-1, se dibujan bien en el frame completo.
ANCHO_INFERENCIA = 320

ultimo_resultado = None
listo_para_inferir = True  # Evita encolar frames: solo enviamos cuando el callback terminó


def on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global ultimo_resultado, listo_para_inferir
    ultimo_resultado = result
    listo_para_inferir = True


def dibujar_manos(frame, results):
    if not results or not results.hand_landmarks:
        return False

    for hand_landmarks in results.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
    return True


def frame_para_inferencia(frame_bgr):
    """Redimensiona solo para el modelo; coordenadas normalizadas siguen valiendo en el frame grande."""
    h, w = frame_bgr.shape[:2]
    if w <= ANCHO_INFERENCIA:
        return frame_bgr
    escala = ANCHO_INFERENCIA / w
    nuevo_h = int(h * escala)
    return cv2.resize(frame_bgr, (ANCHO_INFERENCIA, nuevo_h), interpolation=cv2.INTER_AREA)


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit(1)

# Menos píxeles desde la cámara = captura y conversión más rápidas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

inicio = time.perf_counter()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    print("Deteccion en tiempo real | Q = salir")
    print("(Sin cola de frames: se procesa el frame mas reciente cuando el modelo termina)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if ultimo_resultado is not None:
            dibujar_manos(display, ultimo_resultado)

        cv2.putText(
            display,"Tiempo real | Q: salir",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,
        )
        # Mostrar el frame en la ventana
        cv2.imshow("Paso 03 - Tiempo real", display)
        # Salir del bucle si se pulsa la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Timestamp real (ms): LIVE_STREAM exige que suba; evita desfase por frame_index fijo
        timestamp_ms = int((time.perf_counter() - inicio) * 1000)

        if listo_para_inferir:
            listo_para_inferir = False
            pequeno = frame_para_inferencia(frame)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB),
            )
            landmarker.detect_async(mp_image, timestamp_ms)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
