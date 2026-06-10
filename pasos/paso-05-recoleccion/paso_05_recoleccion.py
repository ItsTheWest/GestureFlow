from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
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
SEQUENCE_LENGTH  = 30   # Frames que componen una secuencia (~1 segundo a 30 fps)
NUM_FEATURES     = 63   # 21 puntos clave de la mano × 3 coordenadas (x, y, z)
NUM_SEQUENCES    = 30   # Cuántos ejemplos recolectamos por gesto

# Cada cuántos frames guardamos una secuencia automáticamente.
# Con SEQUENCE_LENGTH=30 y SAVE_EVERY=15 hay un solapamiento del 50%:
# eso genera mayor diversidad en los datos de entrenamiento.
SAVE_EVERY       = 15

# Segundos de cuenta regresiva antes de iniciar la grabación automática
COUNTDOWN_SECS   = 3

# Frames que dura el destello de confirmación en el HUD (~0.5 s a 30 fps)
FLASH_DURATION   = 15


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
        # No hay mano visible → devolvemos ceros para mantener forma constante (63,)
        return np.zeros(NUM_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Paso 3a — Renderizado del HUD en fase de espera (antes de ESPACIO)
# ---------------------------------------------------------------------------
def draw_waiting(frame: np.ndarray, gesture: str, saved: int) -> None:
    """Muestra el estado de espera: cámara activa pero sin recolectar todavía."""
    h, w = frame.shape[:2]

    # Nombre del gesto
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(frame, f"Guardadas: {saved}/{NUM_SEQUENCES}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)

    # Instrucción central grande
    msg = "Presiona ESPACIO para iniciar"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(frame, msg, ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)

    cv2.putText(frame, "Q: salir", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)


# ---------------------------------------------------------------------------
# Paso 3b — Renderizado del HUD en fase de cuenta regresiva
# ---------------------------------------------------------------------------
def draw_countdown(frame: np.ndarray, gesture: str, seconds_left: int) -> None:
    """Muestra el nombre del gesto y el número de cuenta regresiva centrado en pantalla."""
    h, w = frame.shape[:2]

    # Fondo semitransparente para legibilidad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Nombre del gesto
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Número grande centrado
    label = str(seconds_left)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 6.0, 8)
    cx, cy = (w - tw) // 2, (h + th) // 2
    cv2.putText(frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 255, 200), 8)

    # Instrucción inferior
    cv2.putText(frame, "Prepara el gesto...", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)


# ---------------------------------------------------------------------------
# Paso 4 — Renderizado del HUD en fase de grabación automática
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    gesture: str,
    saved: int,
    buffer_len: int,
    frame_counter: int,
    hand_detected: bool,
    flash_timer: int,
) -> None:
    """
    Muestra el estado de la grabación automática sobre el frame de la cámara.

    Elementos:
      · Nombre del gesto y contador de secuencias guardadas (arriba)
      · Indicador de mano detectada
      · Barra de buffer rodante con marcador de próximo guardado automático
      · Flash de confirmación al guardar
    """
    h, w = frame.shape[:2]

    COLOR_BLANCO = (255, 255, 255)
    COLOR_CIAN   = (0, 255, 200)
    COLOR_VERDE  = (80, 255, 80)
    COLOR_ROJO   = (80, 80, 255)
    COLOR_TENUE  = (140, 140, 140)
    COLOR_FONDO  = (50, 50, 50)

    # --- Sección superior ----------------------------------------------------
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLANCO, 2)

    cv2.putText(frame, f"Guardadas: {saved}/{NUM_SEQUENCES}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_CIAN, 2)

    # Indicador de detección de mano
    hand_label = "Mano detectada ✓" if hand_detected else "Sin mano — muestrate en camara"
    hand_color = COLOR_VERDE if hand_detected else COLOR_ROJO
    cv2.putText(frame, hand_label, (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    # --- Barra de buffer rodante ---------------------------------------------
    BAR_TOP, BAR_BOT = h - 65, h - 45
    cv2.rectangle(frame, (10, BAR_TOP), (w - 10, BAR_BOT), COLOR_FONDO, -1)

    # Relleno proporcional al contenido del buffer
    fill_w = int((buffer_len / SEQUENCE_LENGTH) * (w - 20))
    if fill_w > 0:
        cv2.rectangle(frame, (10, BAR_TOP), (10 + fill_w, BAR_BOT), COLOR_CIAN, -1)

    # Marcador vertical que indica cuándo ocurrirá el próximo guardado automático
    if buffer_len == SEQUENCE_LENGTH:
        # Calculamos la posición relativa dentro del ciclo SAVE_EVERY
        progress_in_cycle = (frame_counter % SAVE_EVERY) / SAVE_EVERY
        marker_x = 10 + int(progress_in_cycle * (w - 20))
        cv2.line(frame, (marker_x, BAR_TOP - 5), (marker_x, BAR_BOT + 5), (255, 255, 0), 2)

    pct = int(buffer_len / SEQUENCE_LENGTH * 100)
    cv2.putText(frame, f"Buffer: {pct}%", (10, BAR_TOP - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TENUE, 1)

    # --- Línea inferior — flash o instrucción --------------------------------
    if flash_timer > 0:
        cv2.putText(frame, f"¡Guardado automaticamente! ({saved}/{NUM_SEQUENCES})",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_VERDE, 2)
    else:
        cv2.putText(frame, "Grabacion automatica activa  |  Q: salir",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TENUE, 1)


# ---------------------------------------------------------------------------
# Paso 5 — Creación de la carpeta de salida con soporte de reanudación
# ---------------------------------------------------------------------------
def pedir_nombre_gesto() -> tuple[str, int] | tuple[None, None]:
    """
    Solicita el nombre del gesto, crea la carpeta y detecta cuántas secuencias
    ya existen para reanudar desde el índice correcto sin sobrescribir datos previos.

    Retorna:
        (nombre_normalizado, próximo_índice) o (None, None) si hay error.
    """
    raw_name   = input("Nombre del gesto a grabar: ")
    normalized = raw_name.strip().lower()

    if not normalized:
        print("Error: El nombre no puede estar vacío.")
        return None, None

    gesture_folder = PROJECT_ROOT / "gestos" / normalized
    gesture_folder.mkdir(parents=True, exist_ok=True)

    if not gesture_folder.exists():
        print(f"Error: No se pudo crear la carpeta: {gesture_folder}")
        return None, None

    # Detectamos el índice de reanudación contando archivos .npy existentes
    existing = sorted(gesture_folder.glob("*.npy"))
    next_index = len(existing)

    if next_index > 0:
        print(f"Carpeta existente: {gesture_folder}")
        print(f"  → Se encontraron {next_index} secuencias ya guardadas. Continuando desde {next_index}.npy")
    else:
        print(f"Carpeta lista en: {gesture_folder}")

    return normalized, next_index


# ---------------------------------------------------------------------------
# Paso 6 — Bucle principal: cuenta regresiva + grabación automática
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, start_index: int, landmarker: vision.HandLandmarker) -> None:
    """
    Fase 0 — Espera: la cámara está activa pero NO recolecta hasta que el usuario
              presione ESPACIO. Esto evita capturar datos accidentalmente al abrir la ventana.
    Fase 1 — Cuenta regresiva: muestra 3-2-1 para que el usuario se coloque.
    Fase 2 — Grabación automática: el buffer rodante guarda una secuencia cada
              SAVE_EVERY frames cuando hay una mano detectada, sin más intervención.

    Convención de nombres: gestos/<gesto>/<índice_secuencia>.npy
    """
    output_dir       = PROJECT_ROOT / "gestos" / gesture_name
    sequences_saved  = start_index        # Reanudamos desde donde quedamos
    sequences_needed = NUM_SEQUENCES - start_index

    if sequences_needed <= 0:
        print(f"✅ El gesto '{gesture_name}' ya tiene {NUM_SEQUENCES} secuencias completas.")
        return

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Crear la ventana con GUI normal para evitar la barra de herramientas de Qt
    cv2.namedWindow("GestureFlow - Recolección Automática", cv2.WINDOW_GUI_NORMAL)

    print(f"\nCámara lista para '{gesture_name}'")
    print(f"  · Presiona ESPACIO cuando estés listo para iniciar la cuenta regresiva")
    print(f"  · Faltan {sequences_needed} secuencias para completar {NUM_SEQUENCES}")
    print("  · Presiona Q en cualquier momento para cancelar.\n")

    # -----------------------------------------------------------------------
    # Fase 0 — Espera activa: cámara encendida, sin recolección
    # -----------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        draw_waiting(frame, gesture_name, sequences_saved)
        cv2.imshow("GestureFlow - Recolección Automática", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("\nCancelado en espera.")
            return
        if key == ord(" "):
            print("  → ESPACIO presionado. Iniciando cuenta regresiva...")
            break   # Salimos de la espera e iniciamos el conteo

    # -----------------------------------------------------------------------
    # Fase 1 — Cuenta regresiva (3-2-1)
    # -----------------------------------------------------------------------
    for i in range(COUNTDOWN_SECS, 0, -1):
        deadline = time.time() + 1.0           # Cada número dura exactamente 1 segundo
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            draw_countdown(frame, gesture_name, i)
            cv2.imshow("GestureFlow - Recolección Automática", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("\nCancelado durante la cuenta regresiva.")
                return

    # -----------------------------------------------------------------------
    # Fase 2 — Grabación automática con buffer rodante
    # -----------------------------------------------------------------------
    # deque(maxlen=30): al agregar el frame 31, el frame 0 se descarta automáticamente
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    frame_counter = 0    # Contador global de frames para el ciclo SAVE_EVERY
    flash_timer   = 0    # Frames restantes de destello de confirmación

    while sequences_saved < NUM_SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)

        # --- Inferencia de MediaPipe (síncrona) ------------------------------
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results      = landmarker.detect(mp_image)
        hand_visible = bool(results.hand_landmarks)

        # Agregamos el vector de 63 puntos al buffer rodante
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)
        frame_counter += 1

        # --- Guardado automático ---------------------------------------------
        # Condiciones: buffer lleno + mano visible + se cumple el ciclo SAVE_EVERY
        if (len(buffer) == SEQUENCE_LENGTH
                and hand_visible
                and frame_counter % SAVE_EVERY == 0):

            npy_array = np.array(buffer, dtype=np.float32)   # forma: (30, 63)
            save_path = output_dir / f"{sequences_saved}.npy"
            np.save(str(save_path), npy_array)

            sequences_saved += 1
            flash_timer = FLASH_DURATION
            print(f"  [✓] Secuencia {sequences_saved:02d}/{NUM_SEQUENCES} guardada → {save_path.name}")

        # --- Cuenta regresiva del flash --------------------------------------
        if flash_timer > 0:
            flash_timer -= 1

        # --- HUD y visualización ---------------------------------------------
        draw_hud(frame, gesture_name, sequences_saved,
                 len(buffer), frame_counter, hand_visible, flash_timer)
        cv2.imshow("GestureFlow - Recolección Automática", frame)

        # Procesamos eventos de UI; Q cancela en cualquier momento
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nGrabación cancelada por el usuario.")
            break

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
    # Paso 1: Pedimos el nombre y detectamos el índice de reanudación
    gesto_creado, next_index = pedir_nombre_gesto()
    if gesto_creado is None or next_index is None:
        exit(1)

    # Paso 2: Construimos el HandLandmarker de MediaPipe (modo IMAGE)
    with build_landmarker() as landmarker:
        # Paso 3: Cuenta regresiva + grabación automática
        grabar_gesto(gesto_creado, next_index, landmarker)
