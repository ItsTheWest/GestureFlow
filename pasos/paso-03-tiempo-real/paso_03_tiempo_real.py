# --- Libraries ---
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
mp_hands = mp.tasks.vision.HandLandmarksConnections

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"

# Inference width (lower = less CPU lag). Landmarks are 0-1 and draw well on the full frame.
ANCHO_INFERENCIA = 320

ultimo_resultado = None
listo_para_inferir = True  # Avoids frame queuing: we only send when the callback has finished


def on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global ultimo_resultado, listo_para_inferir
    ultimo_resultado = result
    listo_para_inferir = True


def dibujar_manos(frame, results):
    if not results or not results.hand_landmarks:
        return False

    for hand_landmarks in results.hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return True


def frame_para_inferencia(frame_bgr):
    """Downscales only for the model; normalized coordinates still apply to the full frame."""
    h, w = frame_bgr.shape[:2]
    if w <= ANCHO_INFERENCIA:
        return frame_bgr
    escala = ANCHO_INFERENCIA / w
    nuevo_h = int(h * escala)
    return cv2.resize(frame_bgr, (ANCHO_INFERENCIA, nuevo_h), interpolation=cv2.INTER_AREA)


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera")
    exit(1)

# Create the window with normal GUI to avoid the Qt toolbar
cv2.namedWindow("Paso 03 - Tiempo real", cv2.WINDOW_GUI_NORMAL)

# Fewer pixels from the camera = faster capture and conversion
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
    print("Real-time detection | Q = quit")
    print("(No frame queue: the most recent frame is processed when the model finishes)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if ultimo_resultado is not None:
            dibujar_manos(display, ultimo_resultado)

        cv2.putText(
            display,"Tiempo real | Q: salir",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,
        )
        # Show the frame in the window
        cv2.imshow("Paso 03 - Tiempo real", display)
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Real timestamp (ms): LIVE_STREAM requires it to increase; avoids drift from a fixed frame_index
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
