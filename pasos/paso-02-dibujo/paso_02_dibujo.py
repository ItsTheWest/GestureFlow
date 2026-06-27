# --- Vision and processing libraries ---
import cv2  # OpenCV: camera capture (BGR), windows and color conversion
import mediapipe as mp  # MediaPipe: image for inference and drawing utilities
from mediapipe.tasks import python  # BaseOptions: path to the .task model
from mediapipe.tasks.python import vision  # HandLandmarker, RunningMode and detect()
from pathlib import Path  # Absolute paths without depending on the execution directory

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
mp_hands = mp.tasks.vision.HandLandmarksConnections

# --- Paths relative to the script (same pattern as prueba.py and future steps) ---
SCRIPT_DIR = Path(__file__).resolve().parent  # Folder of this step: pasos/paso-02-dibujo/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # GestureFlow/ root (two levels up from paso-02-dibujo)
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"  # Shared model


def dibujar_manos(frame, results):
    """
    Dibuja landmarks y conexiones de cada mano detectada sobre `frame` (in-place).
    Devuelve True si hubo al menos una mano; False si no hay detecciones.
    """
    if not results.hand_landmarks:
        return False

    for hand_landmarks in results.hand_landmarks:
        # Circles on each point + skeleton lines (HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return True


# --- Verify the model exists before opening the camera ---
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# --- Webcam setup (inherited from step 01) ---
cap = cv2.VideoCapture(0)  # 0 = default camera; try 1 if it doesn't open
if not cap.isOpened():
    print("Error: Could not open the camera")
    exit(1)

# --- HandLandmarker detector configuration (same idea as prueba.py) ---
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  # One frame at a time (not LIVE_STREAM yet)
    num_hands=2,  # Maximum number of hands to detect in each SPACE capture
)

# The context manager releases the model when leaving the with block
with vision.HandLandmarker.create_from_options(options) as landmarker:
    print("ESPACIO = detectar y dibujar manos | Q = salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the frame")
            break

        frame = cv2.flip(frame, 1)  # Horizontal mirror (same orientation as step 01)
        preview = frame.copy()  # Copy for the live feed without drawing on top yet

        # Instructions in the preview window
        cv2.putText(
            preview,
            "ESPACIO: detectar | Q: salir",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Paso 02 - Dibujo", preview)

        key = cv2.waitKey(1) & 0xFF  # Key press in ~1 ms; keeps the loop smooth
        if key == ord("q"):
            break

        if key == ord(" "):
            # Freeze the current frame for inference and drawing (not the preview)
            snapshot = frame.copy()
            # MediaPipe expects RGB; OpenCV delivers BGR → mandatory conversion
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB),
            )
            results = landmarker.detect(mp_image)

            if dibujar_manos(snapshot, results):
                print(f"Hands detected: {len(results.hand_landmarks)}")
            else:
                print("No hands detected")

            # Show the snapshot with drawn hands until another key is pressed
            cv2.imshow("Paso 02 - Dibujo", snapshot)
            cv2.waitKey(0)  # Pause until a key is pressed (same as prueba.py after detecting)

# --- Release camera and close windows (always on loop exit) ---
cap.release()
cv2.destroyAllWindows()

