
# --- Librerías de visión y procesamiento ---
import cv2  # OpenCV: leer imágenes (BGR) y mostrar ventanas
import mediapipe as mp  # MediaPipe: detección de manos y utilidades de dibujo
from mediapipe.tasks import python  # Configuración base del modelo (.task)
from mediapipe.tasks.python import vision  # HandLandmarker y modos de ejecución
from pathlib import Path  # Rutas multiplataforma sin depender del directorio actual

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
mp_hands = mp.tasks.vision.HandLandmarksConnections


# --- Rutas relativas al script (funciona aunque ejecutes desde otra carpeta) ---

SCRIPT_DIR = Path(__file__).resolve().parent  # Carpeta donde está este .py (prueba/)
PROJECT_ROOT = SCRIPT_DIR.parent  # Raíz del proyecto (GestureFlow/)

image_path = PROJECT_ROOT / "assets" / "img_prueba" /  "image.png"  # Imagen de entrada
model_path = SCRIPT_DIR / "hand_landmarker.task"  # Modelo entrenado de MediaPipe

# --- Carga de la imagen en memoria (formato BGR de OpenCV) ---

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

# --- Configuración del detector HandLandmarker ---
base_options = python.BaseOptions(
    model_asset_path=str(model_path)  # Ruta al archivo .task en disco
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  # Una sola imagen (no video ni cámara en vivo)
    num_hands=2,  # Máximo de manos a buscar en el frame

)

# El context manager libera recursos del modelo al salir del bloque
with vision.HandLandmarker.create_from_options(options) as landmarker:
    # MediaPipe espera RGB; OpenCV guarda en BGR → conversión obligatoria

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),

    )
    # Inferencia: devuelve landmarks normalizados (x,y,z entre 0 y 1 respecto al tamaño de imagen)
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        # Una entrada por cada mano detectada (hasta num_hands)
        for hand_landmarks in results.hand_landmarks:
            # Dibuja círculos en cada landmark y líneas según HAND_CONNECTIONS
            mp_drawing.draw_landmarks(
                image,  # Se modifica in-place (misma matriz que imread)
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  # Qué puntos unir (esqueleto de la mano)
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

# --- Visualización del resultado ---
# Crear la ventana con GUI normal para evitar la barra de herramientas de Qt
cv2.namedWindow("Result", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("Result", image)  # Ventana con la imagen y las manos dibujadas
cv2.waitKey(0)  # Pausa hasta que pulses una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV 

