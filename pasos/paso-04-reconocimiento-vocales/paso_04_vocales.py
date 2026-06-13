# --- Librerías ---
import math
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections as mp_hands

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# Ancho para inferencia (más bajo = menos retardo en CPU). Landmarks son 0-1, se dibujan bien en el frame completo.
ANCHO_INFERENCIA = 320

ultimo_resultado = None
listo_para_inferir = True  # Evita encolar frames: solo enviamos cuando el callback terminó

# Variables para validar que el gesto se mantenga por mano (Left/Right)
TIEMPO_CONFIRMACION = 1.0  # Segundos que debe mantenerse el gesto

estado_manos = {
    "Left": {
        "vocal_detectada": None,
        "tiempo_inicio": 0.0,
        "confirmada": False
    },
    "Right": {
        "vocal_detectada": None,
        "tiempo_inicio": 0.0,
        "confirmada": False
    }
}


def on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """Callback asíncrono para recibir los resultados del landmarker."""
    global ultimo_resultado, listo_para_inferir
    ultimo_resultado = result
    listo_para_inferir = True


def dibujar_manos(frame: np.ndarray, results: vision.HandLandmarkerResult) -> bool:
    """
    Dibuja los landmarks y conexiones de la mano en el frame actual.

    Args:
        frame: Lienzo BGR sobre el que se dibujarán las marcas.
        results: Resultados entregados por el landmarker.

    Returns:
        bool: True si se detectó y dibujó al menos una mano, False en caso contrario.
    """
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


def frame_para_inferencia(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Redimensiona el fotograma para acelerar la inferencia en CPU.

    Args:
        frame_bgr: Imagen original a resolución completa.

    Returns:
        np.ndarray: Imagen redimensionada si excede el ancho establecido.
    """
    h, w = frame_bgr.shape[:2]
    if w <= ANCHO_INFERENCIA:
        return frame_bgr
    escala = ANCHO_INFERENCIA / w
    nuevo_h = int(h * escala)
    return cv2.resize(frame_bgr, (ANCHO_INFERENCIA, nuevo_h), interpolation=cv2.INTER_AREA)


def get_vowel(hand_landmarks: list) -> str | None:
    """
    Clasifica de manera heurística si el gesto de la mano corresponde a una vocal.

    Args:
        hand_landmarks: Lista de 21 landmarks de una mano.

    Returns:
        str | None: Carácter de la vocal ('A', 'E', 'I', 'O', 'U') o None.
    """
    # En la API Tasks de MediaPipe, hand_landmarks es directamente una lista.
    # 1. Comprobar que los 4 dedos estén cerrados (la punta está por debajo de la articulación PIP)
    is_index_closed = hand_landmarks[8].y > hand_landmarks[6].y # y = eje vertical, de arriba hacia abajo dedo 8 es la punta y el dedo 6 es la articulación PIP
    is_middle_closed = hand_landmarks[12].y > hand_landmarks[10].y # y = eje vertical, de arriba hacia abajo dedo 12 es la punta y el dedo 10 es la articulación PIP
    is_ring_closed = hand_landmarks[16].y > hand_landmarks[14].y # y = eje vertical, de arriba hacia abajo dedo 16 es la punta y el dedo 14 es la articulación PIP
    is_pinky_closed = hand_landmarks[20].y > hand_landmarks[18].y # y = eje vertical, de arriba hacia abajo dedo 20 es la punta y el dedo 18 es la articulación PIP
    is_ring_semi_closed = hand_landmarks[15].y > hand_landmarks[14].y and hand_landmarks[16].y > hand_landmarks[15].y # y = eje vertical, de arriba hacia abajo dedo 15 es la articulación PIP y el dedo 16 es la punta
    is_pinky_semi_closed = hand_landmarks[19].y > hand_landmarks[18].y and hand_landmarks[20].y > hand_landmarks[19].y # y = eje vertical, de arriba hacia abajo dedo 19 es la articulación PIP y el dedo 20 es la punta
    
    
    # 2. Lógica específica del pulgar para la letra 'A' (Pulgar apoyado al lado del índice)
    # El pulgar debe estar apuntando hacia arriba (punta más arriba que su articulación)
    is_thumb_up = (hand_landmarks[4].y < hand_landmarks[3].y) and (hand_landmarks[4].y < hand_landmarks[6].y)
    
    # El pulgar NO debe cruzar los otros dedos (eso sería una 'S'). 
    # Debe estar "por fuera" del dedo índice. Comparamos la distancia en el eje X
    # hacia el meñique para asegurarnos de que está al lado exterior de la mano.
    dist_thumb_to_pinky = abs(hand_landmarks[4].x - hand_landmarks[17].x)
    dist_index_to_pinky = abs(hand_landmarks[5].x - hand_landmarks[17].x)
    is_thumb_on_side = dist_thumb_to_pinky > dist_index_to_pinky    

    is_thumb_down = hand_landmarks[4].x > hand_landmarks[3].x 

    # Regla estricta para 'A'
    if is_index_closed and is_middle_closed and is_ring_closed and is_pinky_closed and is_thumb_up and is_thumb_on_side:
        return 'A'
    # Regla estricta para 'E'
    if is_index_closed and is_middle_closed and is_ring_closed and is_pinky_closed and is_thumb_down:
        return 'E'
    # Regla estricta para 'I'
    if  is_index_closed and is_middle_closed and is_ring_closed and not is_pinky_closed and is_thumb_up:
        return 'I'
    # Regla estricta para 'U'
    if not is_index_closed and not is_middle_closed and is_ring_closed and is_pinky_closed and is_thumb_down:
        return 'U'
        
    # logica para 'O' en la que los dedos de la mano se tocan y forman un circulo
    def get_distance(lm1, lm2):
        return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
    
    dist_thumb_index = get_distance(hand_landmarks[4], hand_landmarks[8])
    dist_thumb_middle = get_distance(hand_landmarks[4], hand_landmarks[12])
    are_tips_touching = dist_thumb_index < 0.05 and dist_thumb_middle < 0.05

    # Regla estricta para 'O'
    if are_tips_touching and is_ring_semi_closed and is_pinky_semi_closed:
        return 'O'
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit(1)

# Crear la ventana con GUI normal para evitar la barra de herramientas de Qt
cv2.namedWindow("Paso 03 - Tiempo real", cv2.WINDOW_GUI_NORMAL)

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
            
            # Detectar vocal por mano si hay manos detectadas
            manos_presentes = set()
            if ultimo_resultado.hand_landmarks and ultimo_resultado.handedness:
                for idx, landmarks in enumerate(ultimo_resultado.hand_landmarks):
                    hand_label = ultimo_resultado.handedness[idx][0].category_name
                    manos_presentes.add(hand_label)
                    
                    vocal = get_vowel(landmarks)
                    
                    if vocal:
                        if vocal == estado_manos[hand_label]["vocal_detectada"]:
                            # La vocal se mantiene, comprobamos el tiempo
                            tiempo_transcurrido = time.time() - estado_manos[hand_label]["tiempo_inicio"]
                            if tiempo_transcurrido >= TIEMPO_CONFIRMACION:
                                estado_manos[hand_label]["confirmada"] = True
                            else:
                                estado_manos[hand_label]["confirmada"] = False
                        else:
                            # Cambio de vocal (o nueva), empezamos a contar
                            estado_manos[hand_label]["vocal_detectada"] = vocal
                            estado_manos[hand_label]["tiempo_inicio"] = time.time()
                            estado_manos[hand_label]["confirmada"] = False
                    else:
                        estado_manos[hand_label]["vocal_detectada"] = None
                        estado_manos[hand_label]["confirmada"] = False
            
            # Resetear el estado de las manos que no aparecieron en el frame
            for hand_label in ["Left", "Right"]:
                if hand_label not in manos_presentes:
                    estado_manos[hand_label]["vocal_detectada"] = None
                    estado_manos[hand_label]["confirmada"] = False

            # Dibujar el estado de las vocales en pantalla
            y_offset = 80
            for hand_label in ["Left", "Right"]:
                vocal = estado_manos[hand_label]["vocal_detectada"]
                if vocal:
                    lado = "Izquierda" if hand_label == "Left" else "Derecha"
                    if estado_manos[hand_label]["confirmada"]:
                        texto = f"Mano {lado} - Confirmada: {vocal}"
                        cv2.putText(display, texto, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    else:
                        tiempo_transcurrido = time.time() - estado_manos[hand_label]["tiempo_inicio"]
                        texto = f"Mano {lado} - Validando {vocal}... {tiempo_transcurrido:.1f}s"
                        cv2.putText(display, texto, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    y_offset += 40

        cv2.putText(
            display, "Tiempo real | Q: salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        # Mostrar el frame en la ventana
        cv2.imshow("Paso 03 - Tiempo real", display)

        # Control de teclado unificado (única llamada waitKey por bucle)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
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

    cap.release()
    cv2.destroyAllWindows()
