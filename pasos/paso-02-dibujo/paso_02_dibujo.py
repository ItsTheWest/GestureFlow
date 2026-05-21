# --- Librerías de visión y procesamiento ---
import cv2  # OpenCV: captura de cámara (BGR), ventanas y conversión de color
import mediapipe as mp  # MediaPipe: imagen para inferencia y utilidades de dibujo
from mediapipe.tasks import python  # BaseOptions: ruta al modelo .task
from mediapipe.tasks.python import vision  # HandLandmarker, RunningMode y detect()
from mediapipe.framework.formats import landmark_pb2  # Formato protobuf que usa draw_landmarks
from pathlib import Path  # Rutas absolutas sin depender del directorio desde el que ejecutas

# --- Rutas relativas al script (mismo patrón que prueba.py y pasos futuros) ---
SCRIPT_DIR = Path(__file__).resolve().parent  # Carpeta de este paso: pasos/paso-02-dibujo/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Raíz GestureFlow/ (sube dos niveles desde paso-02-dibujo)
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"  # Modelo compartido con Fase 0


def dibujar_manos(frame, results):
    """
    Dibuja landmarks y conexiones de cada mano detectada sobre `frame` (in-place).
    Devuelve True si hubo al menos una mano; False si no hay detecciones.
    """
    if not results.hand_landmarks:
        return False

    for hand_landmarks in results.hand_landmarks:
        # La API Tasks devuelve objetos simples; draw_landmarks necesita protobuf
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks  # 21 puntos por mano (muñeca, nudillos, yemas, etc.)
        ])
        # Círculos en cada punto + líneas del esqueleto (HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
    return True


# --- Comprobar que el modelo existe antes de abrir la cámara ---
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

# --- Apertura de la webcam (heredado del paso 01) ---
cap = cv2.VideoCapture(0)  # 0 = cámara por defecto; prueba 1 si no abre
if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit(1)

# --- Configuración del detector HandLandmarker (misma idea que prueba.py) ---
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  # Un frame a la vez (no LIVE_STREAM aún)
    num_hands=2,  # Máximo de manos a buscar en cada captura con ESPACIO
)

# El context manager libera el modelo al salir del bloque with
with vision.HandLandmarker.create_from_options(options) as landmarker:
    print("ESPACIO = detectar y dibujar manos | Q = salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame")
            break

        frame = cv2.flip(frame, 1)  # Espejo horizontal (misma orientación que paso 01)
        preview = frame.copy()  # Copia para el vídeo en vivo sin dibujar encima todavía

        # Instrucciones en la ventana de previsualización
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

        key = cv2.waitKey(1) & 0xFF  # Tecla en ~1 ms; permite bucle fluido
        if key == ord("q"):
            break

        if key == ord(" "):
            # Congelar el frame actual para inferencia y dibujo (no el preview)
            snapshot = frame.copy()
            # MediaPipe espera RGB; OpenCV entrega BGR → conversión obligatoria
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB),
            )
            results = landmarker.detect(mp_image)

            if dibujar_manos(snapshot, results):
                print(f"Manos detectadas: {len(results.hand_landmarks)}")
            else:
                print("No se detectaron manos")

            # Mostrar el snapshot con manos dibujadas hasta pulsar otra tecla
            cv2.imshow("Paso 02 - Dibujo", snapshot)
            cv2.waitKey(0)  # Pausa hasta tecla (igual que prueba.py tras detectar)

# --- Liberar cámara y cerrar ventanas (siempre al salir del bucle) ---
cap.release()
cv2.destroyAllWindows()
