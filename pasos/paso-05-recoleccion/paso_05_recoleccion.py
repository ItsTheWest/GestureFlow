import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Resolución de rutas — mismo patrón que paso_04_vocales.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Parámetros de grabación
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30   # Número de frames que componen una sola secuencia de movimiento
NUM_FEATURES   = 63    # 21 puntos clave de la mano * 3 coordenadas (x, y, z)
NUM_SEQUENCES  = 30    # Cuántas secuencias de ejemplo grabaremos por gesto


# ---------------------------------------------------------------------------
# Paso 2.1 — Configuración de MediaPipe (modo IMAGE para procesamiento síncrono)
# ---------------------------------------------------------------------------
# Usamos RunningMode.IMAGE en lugar de LIVE_STREAM para que cada llamada a detect()
# bloquee hasta que MediaPipe devuelva el resultado. Esto garantiza que capturamos
# exactamente SEQUENCE_LENGTH frames sin que ninguno se pierda de forma asíncrona.
def build_landmarker() -> vision.HandLandmarker:
    """Crea y devuelve un HandLandmarker configurado en modo IMAGE (síncrono)."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo no encontrado en: {MODEL_PATH}\n"
            "Asegúrate de haber descargado 'hand_landmarker.task' en la carpeta 'prueba/'."
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Síncrono — no se necesita callback
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Paso 2.2 — Función auxiliar de extracción de puntos clave
# ---------------------------------------------------------------------------
def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """
    Aplana los landmarks de la primera mano detectada en un array 1-D de 63 valores.

    Retorna:
        np.ndarray de forma (63,): [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        o np.zeros(63) si no se detectó ninguna mano — mantiene la forma constante.
    """
    if results.hand_landmarks:
        # Usamos únicamente la primera mano detectada (índice 0)
        hand = results.hand_landmarks[0]
        # Construimos una lista plana: x, y, z por cada uno de los 21 landmarks
        keypoints = []
        for landmark in hand:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints, dtype=np.float32)   # forma: (63,)
    else:
        # No hay mano visible → retornamos ceros para que cada frame mantenga forma (63,)
        return np.zeros(NUM_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Paso 2.3 (auxiliar) — Superponer texto en el frame para guiar al usuario
# ---------------------------------------------------------------------------
def draw_hud(frame: np.ndarray, gesture: str, sequence: int, frame_num: int, waiting: bool) -> None:
    """Renderiza el estado de la grabación directamente sobre la imagen de la cámara."""
    h = frame.shape[0]
    color_acento = (0, 255, 200)   # Verde-cian para información de progreso
    color_aviso  = (0, 200, 255)   # Amarillo-naranja para el estado de espera
    color_label  = (255, 255, 255) # Blanco para el nombre del gesto

    # Nombre del gesto en la parte superior
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_label, 2)

    # Contador de secuencia actual vs. total
    cv2.putText(frame, f"Secuencia {sequence + 1}/{NUM_SEQUENCES}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_acento, 2)

    if waiting:
        # Estado de pausa: pedimos al usuario que se prepare
        cv2.putText(frame, "PREPARATE...", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_aviso, 3)
    else:
        # Estado activo de grabación: mostramos el progreso de frames
        cv2.putText(frame, f"Grabando frame {frame_num + 1}/{SEQUENCE_LENGTH}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_acento, 2)


# ---------------------------------------------------------------------------
# Creación de carpeta — Paso 1 del flujo de trabajo
# ---------------------------------------------------------------------------
def pedir_nombre_gesto() -> str | None:
    """
    Solicita al usuario el nombre del gesto, crea la carpeta de salida y retorna
    el nombre normalizado. Retorna None si la carpeta no se pudo crear.
    """
    raw_name = input("Nombre del gesto a grabar: ")
    normalized = raw_name.strip().lower()  # Quitamos espacios y convertimos a minúsculas

    if not normalized:
        print("Error: El nombre no puede estar vacío.")
        return None

    # Ruta de destino: gestos/<nombre_normalizado>/
    gesture_folder = PROJECT_ROOT / "gestos" / normalized
    # parents=True crea carpetas intermedias; exist_ok=True no falla si ya existe
    gesture_folder.mkdir(parents=True, exist_ok=True)

    if not gesture_folder.exists():
        print(f"Error: No se pudo crear la carpeta: {gesture_folder}")
        return None

    print(f"Carpeta lista en: {gesture_folder}")
    return normalized


# ---------------------------------------------------------------------------
# Pasos 2.3 + 2.4 — Bucle principal de grabación
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, landmarker: vision.HandLandmarker) -> None:
    """
    Graba NUM_SEQUENCES secuencias de SEQUENCE_LENGTH frames cada una, extrae
    los puntos clave por frame y guarda cada secuencia como archivo .npy.

    Convención de nombres: gestos/<gesto>/<índice_secuencia>.npy
    """
    output_dir = PROJECT_ROOT / "gestos" / gesture_name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nIniciando grabación para el gesto '{gesture_name}'")
    print(f"  · Secuencias: {NUM_SEQUENCES}")
    print(f"  · Frames por secuencia: {SEQUENCE_LENGTH}")
    print("  · Presiona Q en cualquier momento para cancelar.\n")

    # -----------------------------------------------------------------------
    # Bucle externo — cada iteración = una secuencia completa grabada
    # -----------------------------------------------------------------------
    for sequence in range(NUM_SEQUENCES):
        sequence_data = []   # Lista temporal que acumula SEQUENCE_LENGTH arrays de forma (63,)

        # -------------------------------------------------------------------
        # Bucle interno — cada iteración = un frame capturado
        # -------------------------------------------------------------------
        for frame_num in range(SEQUENCE_LENGTH):

            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara.")
                break

            # Espejamos la imagen para que el usuario vea su mano de forma natural
            frame = cv2.flip(frame, 1)

            # UX: En el primer frame de cada secuencia pausamos 2 segundos
            # para que el usuario tenga tiempo de reposicionar la mano.
            is_waiting = (frame_num == 0)
            if is_waiting:
                draw_hud(frame, gesture_name, sequence, frame_num, waiting=True)
                cv2.imshow("GestureFlow - Recolección de Datos", frame)
                # Pausa de 2 segundos; Q sigue funcionando para cancelar
                if cv2.waitKey(2000) & 0xFF == ord("q"):
                    print("\nGrabación cancelada por el usuario.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                # Re-leemos para mostrar un frame fresco después de la pausa
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

            # ---------------------------------------------------------------
            # Convertimos BGR → RGB y encapsulamos en mp.Image para la detección síncrona
            # ---------------------------------------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Llamada síncrona — bloquea hasta que MediaPipe devuelve el resultado
            results = landmarker.detect(mp_image)

            # Extraemos los 63 puntos clave (ceros si no se detectó ninguna mano)
            keypoints = extract_keypoints(results)
            sequence_data.append(keypoints)

            # Mostramos el progreso de grabación al usuario
            draw_hud(frame, gesture_name, sequence, frame_num, waiting=False)
            cv2.imshow("GestureFlow - Recolección de Datos", frame)

            # Espera de 1 ms para procesar eventos de la UI; permite salir con Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nGrabación cancelada por el usuario.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # -------------------------------------------------------------------
        # Paso 2.4 — Guardamos la secuencia completa como archivo .npy
        # -------------------------------------------------------------------
        # sequence_data es una lista de SEQUENCE_LENGTH arrays, cada uno de forma (63,)
        # np.array() lo convierte a forma (SEQUENCE_LENGTH, NUM_FEATURES) = (30, 63)
        npy_array = np.array(sequence_data, dtype=np.float32)   # forma: (30, 63)
        save_path = output_dir / f"{sequence}.npy"
        np.save(str(save_path), npy_array)
        print(f"  [✓] Secuencia {sequence + 1:02d}/{NUM_SEQUENCES} guardada → {save_path.name}  forma={npy_array.shape}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Recolección completa. {NUM_SEQUENCES} secuencias guardadas en: {output_dir}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Paso 1: Pedimos el nombre del gesto y creamos la carpeta de salida
    gesto_creado = pedir_nombre_gesto()
    if not gesto_creado:
        exit(1)

    # Paso 2: Construimos el landmarker de MediaPipe (modo IMAGE)
    with build_landmarker() as landmarker:
        # Paso 3: Ejecutamos el bucle doble de grabación
        grabar_gesto(gesto_creado, landmarker)
