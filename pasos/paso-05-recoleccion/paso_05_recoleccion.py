import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Resolución de rutas — mismo patrón que los pasos anteriores
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH   = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Parámetros de grabación
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30   # Frames que componen una secuencia de gesto (~1 segundo a 30 fps)
NUM_FEATURES    = 63   # 21 puntos clave de la mano × 3 coordenadas (x, y, z)
NUM_SEQUENCES   = 30   # Cuántos ejemplos recolectamos por gesto

# Duración del mensaje flash en frames (~1.5 s a 30 fps)
FLASH_DURATION  = 45


# ---------------------------------------------------------------------------
# Paso 1 — Configuración de MediaPipe (modo IMAGE = síncrono, sin callback)
# ---------------------------------------------------------------------------
def build_landmarker() -> vision.HandLandmarker:
    """Crea y devuelve un HandLandmarker configurado en modo IMAGE síncrono."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo no encontrado en: {MODEL_PATH}\n"
            "Asegúrate de haber descargado 'hand_landmarker.task' en la carpeta 'prueba/'."
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Síncrono — bloquea hasta recibir el resultado
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Paso 2 — Función auxiliar de extracción de puntos clave
# ---------------------------------------------------------------------------
def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """
    Aplana los landmarks de la primera mano detectada en un array 1-D de 63 valores.

    Retorna:
        np.ndarray de forma (63,): [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        o np.zeros(63) si no hay mano visible — mantiene la forma constante entre frames.
    """
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]   # Solo la primera mano detectada
        keypoints = []
        for lm in hand:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints, dtype=np.float32)   # forma: (63,)
    else:
        # No hay mano visible → devolvemos ceros para que cada frame mantenga forma (63,)
        return np.zeros(NUM_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Paso 3 — Renderizado del HUD (modo continuo)
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    gesture: str,
    saved: int,
    buffer_len: int,
    flash_msg: str | None,
) -> None:
    """
    Superpone el estado de la grabación directamente sobre el frame de la cámara.

    Disposición:
      Arriba  — nombre del gesto + contador de secuencias guardadas
      Abajo   — barra de progreso del buffer + instrucción o mensaje flash
    """
    h, w = frame.shape[:2]

    COLOR_BLANCO  = (255, 255, 255)
    COLOR_CIAN    = (0, 255, 200)    # Relleno de la barra y texto de progreso
    COLOR_FLASH   = (80, 255, 80)    # Verde brillante para confirmación de guardado
    COLOR_AVISO   = (0, 165, 255)    # Naranja para advertencia de buffer incompleto
    COLOR_TENUE   = (140, 140, 140)  # Gris para la instrucción en reposo
    COLOR_FONDO   = (50, 50, 50)     # Fondo oscuro de la pista de la barra

    # --- Sección superior ----------------------------------------------------
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLANCO, 2)

    cv2.putText(frame, f"Guardadas: {saved}/{NUM_SEQUENCES}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_CIAN, 2)

    # --- Barra de progreso del buffer rodante --------------------------------
    # Pista de fondo
    BAR_TOP, BAR_BOT = h - 65, h - 45
    cv2.rectangle(frame, (10, BAR_TOP), (w - 10, BAR_BOT), COLOR_FONDO, -1)

    # Relleno proporcional al contenido actual del buffer
    fill_w = int((buffer_len / SEQUENCE_LENGTH) * (w - 20))
    if fill_w > 0:
        cv2.rectangle(frame, (10, BAR_TOP), (10 + fill_w, BAR_BOT), COLOR_CIAN, -1)

    # Etiqueta de porcentaje sobre la barra
    pct = int(buffer_len / SEQUENCE_LENGTH * 100)
    cv2.putText(frame, f"Buffer: {pct}%", (10, BAR_TOP - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TENUE, 1)

    # --- Línea inferior — mensaje flash o instrucción permanente -------------
    if flash_msg:
        # El color depende del tipo de mensaje: éxito o advertencia
        color = COLOR_FLASH if "Guardado" in flash_msg else COLOR_AVISO
        cv2.putText(frame, flash_msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "ESPACIO: guardar ultimo gesto  |  Q: salir", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TENUE, 1)


# ---------------------------------------------------------------------------
# Paso 4 — Creación de la carpeta de salida
# ---------------------------------------------------------------------------
def pedir_nombre_gesto() -> str | None:
    """
    Solicita al usuario el nombre del gesto, crea la carpeta de salida y retorna
    el nombre normalizado. Retorna None si la carpeta no se pudo crear.
    """
    raw_name   = input("Nombre del gesto a grabar: ")
    normalized = raw_name.strip().lower()   # Eliminamos espacios y convertimos a minúsculas

    if not normalized:
        print("Error: El nombre no puede estar vacío.")
        return None

    gesture_folder = PROJECT_ROOT / "gestos" / normalized
    gesture_folder.mkdir(parents=True, exist_ok=True)   # Crea carpetas intermedias sin error

    if not gesture_folder.exists():
        print(f"Error: No se pudo crear la carpeta: {gesture_folder}")
        return None

    print(f"Carpeta lista en: {gesture_folder}")
    return normalized


# ---------------------------------------------------------------------------
# Paso 5 — Bucle principal de grabación (deque continuo + disparador ESPACIO)
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, landmarker: vision.HandLandmarker) -> None:
    """
    Lee frames de la cámara de forma continua y los acumula en un buffer rodante (deque).
    Cada frame es procesado por MediaPipe y sus 63 puntos clave se agregan al buffer;
    el frame más antiguo se descarta automáticamente cuando el buffer está lleno.

    Al presionar ESPACIO, el contenido actual del buffer (últimos SEQUENCE_LENGTH frames)
    se guarda como archivo .npy, sin interrumpir el flujo de la cámara.

    Convención de nombres: gestos/<gesto>/<índice_secuencia>.npy
    """
    output_dir = PROJECT_ROOT / "gestos" / gesture_name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nIniciando grabación continua para '{gesture_name}'")
    print(f"  · Realiza el gesto libremente frente a la cámara")
    print(f"  · Cuando el gesto esté completo, presiona ESPACIO para guardar los últimos {SEQUENCE_LENGTH} frames")
    print(f"  · Repite hasta completar {NUM_SEQUENCES} secuencias")
    print("  · Presiona Q en cualquier momento para cancelar.\n")

    # Buffer rodante: descarta automáticamente el frame más antiguo cuando está lleno
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    sequences_saved = 0
    flash_msg: str | None = None
    flash_timer: int = 0

    while sequences_saved < NUM_SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        # Espejamos la imagen para que el usuario vea su mano de forma natural
        frame = cv2.flip(frame, 1)

        # --- Inferencia de MediaPipe (síncrona) ------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = landmarker.detect(mp_image)

        # Agregamos el vector de 63 puntos clave al buffer rodante
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)

        # --- Cuenta regresiva del mensaje flash ------------------------------
        if flash_timer > 0:
            flash_timer -= 1
        else:
            flash_msg = None   # Limpiamos el flash al expirar el temporizador

        # --- HUD y visualización ---------------------------------------------
        draw_hud(frame, gesture_name, sequences_saved, len(buffer), flash_msg)
        cv2.imshow("GestureFlow - Recolección Continua", frame)

        # --- Manejo de teclas ------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nGrabación cancelada por el usuario.")
            break

        if key == ord(" "):
            if len(buffer) == SEQUENCE_LENGTH:
                # Guardamos una instantánea del buffer rodante actual
                npy_array = np.array(buffer, dtype=np.float32)   # forma: (30, 63)
                save_path = output_dir / f"{sequences_saved}.npy"
                np.save(str(save_path), npy_array)

                sequences_saved += 1
                print(f"  [✓] Secuencia {sequences_saved:02d}/{NUM_SEQUENCES} guardada → {save_path.name}  forma={npy_array.shape}")

                # Activamos el flash verde de confirmación en el HUD
                flash_msg   = f"¡Guardado! ({sequences_saved}/{NUM_SEQUENCES})"
                flash_timer = FLASH_DURATION

            else:
                # El buffer aún no está lleno (primer segundo de grabación)
                flash_msg   = f"Buffer incompleto: {len(buffer)}/{SEQUENCE_LENGTH} frames — espera un momento"
                flash_timer = FLASH_DURATION

    cap.release()
    cv2.destroyAllWindows()

    if sequences_saved == NUM_SEQUENCES:
        print(f"\n✅ Recolección completa. {NUM_SEQUENCES} secuencias guardadas en: {output_dir}")
    else:
        print(f"\n⚠️ Recolección incompleta: {sequences_saved}/{NUM_SEQUENCES} secuencias guardadas.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Paso 1: Pedimos el nombre del gesto y creamos la carpeta de salida
    gesto_creado = pedir_nombre_gesto()
    if not gesto_creado:
        exit(1)

    # Paso 2: Construimos el HandLandmarker de MediaPipe (modo IMAGE)
    with build_landmarker() as landmarker:
        # Paso 3: Ejecutamos el bucle de captura continua
        grabar_gesto(gesto_creado, landmarker)
