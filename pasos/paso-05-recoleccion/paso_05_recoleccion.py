from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import extract_keypoints

# ---------------------------------------------------------------------------
# Path resolution — same pattern as previous steps
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH   = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Recording parameters 
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH  = 30   # Frames per sequence (~1 second at 30 fps)
NUM_FEATURES     = 126  # 42 landmarks (21 per hand) × 3 coordinates (x, y, z)
NUM_SEQUENCES    = 30   # How many examples we collect per gesture

# How often (in frames) we auto-save a sequence.
# With SEQUENCE_LENGTH=30 and SAVE_EVERY=15 there is 50% overlap:
# this generates more diversity in the training data.
SAVE_EVERY       = 15

# Seconds of countdown before automatic recording begins
COUNTDOWN_SECS   = 3

# Frames the confirmation flash lasts in the HUD (~0.5 s at 30 fps)
FLASH_DURATION   = 15


# ---------------------------------------------------------------------------
# Step 1 — MediaPipe configuration (IMAGE mode = synchronous, no callback)
# ---------------------------------------------------------------------------
def build_landmarker() -> vision.HandLandmarker:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH)) 

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Step 3a — HUD rendering for the waiting phase (before SPACE)
# ---------------------------------------------------------------------------
def draw_waiting(frame: np.ndarray, gesture: str, saved: int) -> None:
    """Muestra el estado de espera: cámara activa pero sin recopilar datos aún."""
    h,w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_hud = 0.8
    font_scale_instruction = 1.0
    thickness_hud = 2
    thickness_instruction = 2
    
    color_green = (0, 255, 0)
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)
    
    gesture_text = f"Gesto: {gesture.upper()}"
    progress_text = f"Progreso: {saved}/{NUM_SEQUENCES}"
    instruction_text = "PRESIONA ESPACIO PARA EMPEZAR"
    quit_text = "Q: Salir"

    #Dibujamos el texto en el frame
    cv2.putText(frame, gesture_text, (20, 40), font, font_scale_hud, color_green, thickness_hud)
    cv2.putText(frame, progress_text, (20, 80), font, font_scale_hud, color_white, thickness_hud) # 
    cv2.putText(frame, quit_text, (20, 120), font, font_scale_hud, color_white, thickness_hud)

    #Obtenemos las dimensiones del texto de la instrucción
    (text_w, text_h), _ = cv2.getTextSize(instruction_text, font, font_scale_instruction, thickness_instruction)
    
    #Calculamos el centro del frame para posicionar el texto
    text_x = (w - text_w) // 2
    text_y = (h + text_h) // 2

    # Dibujamos el texto de la instrucción 
    cv2.putText(frame, instruction_text, (text_x, text_y), font, font_scale_instruction, color_green, thickness_instruction)

# ---------------------------------------------------------------------------
# Step 3b — HUD rendering for the countdown phase
# ---------------------------------------------------------------------------
def draw_countdown(frame: np.ndarray, gesture: str, seconds_left: int) -> None:
    """Muestra el nombre del gesto y el número de la cuenta atrás centrado en la pantalla."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create a semi-transparent dark overlay for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw gesture name at top-left
    gesture_text = f"Gesto: {gesture.upper()}"
    cv2.putText(frame, gesture_text, (20, 40), font, 0.8, (0, 255, 0), 2)
    
    # Draw "Q: quit" instruction at top-left (below gesture)
    quit_text = "Q: Salir"
    cv2.putText(frame, quit_text, (20, 80), font, 0.8, (255, 255, 255), 2)

    # Draw the countdown number large and centered
    number_text = str(seconds_left)
    font_scale_num = 6.0
    thickness_num = 12
    (num_w, num_h), _ = cv2.getTextSize(number_text, font, font_scale_num, thickness_num)
    num_x = (w - num_w) // 2
    num_y = (h + num_h) // 2
    cv2.putText(frame, number_text, (num_x, num_y), font, font_scale_num, (0, 255, 0), thickness_num)

    # Draw "Prepare your gesture..." at the bottom
    prepare_text = "PREPARA TU GESTO..."
    font_scale_prep = 0.8
    thickness_prep = 2
    (prep_w, prep_h), _ = cv2.getTextSize(prepare_text, font, font_scale_prep, thickness_prep)
    prep_x = (w - prep_w) // 2
    prep_y = h - 60
    cv2.putText(frame, prepare_text, (prep_x, prep_y), font, font_scale_prep, (255, 255, 255), thickness_prep)


# ---------------------------------------------------------------------------
# Step 4 — HUD rendering during automatic recording
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
    Muestra la interfaz HUD de grabación sobre el cuadro de la cámara.

    Elementos:
      · Nombre del gesto y contador de secuencias guardadas (arriba)
      · Indicador de detección de mano
      · Barra de progreso del búfer circular con marcador de guardado
      · Flash de confirmación cuando se guarda una secuencia
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Definiciones de colores en BGR
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_white = (255, 255, 255)
    color_gray = (50, 50, 50)

    # 1. Barra superior del HUD: Indicador REC (izquierda) y contador de guardados (derecha)
    rec_text = f"REC: {gesture.upper()}"
    cv2.putText(frame, rec_text, (20, 40), font, 0.8, color_red, 2)

    progress_text = f"GUARDADO: {saved}/{NUM_SEQUENCES}"
    (prog_w, _), _ = cv2.getTextSize(progress_text, font, 0.8, 2)
    cv2.putText(frame, progress_text, (w - prog_w - 20, 40), font, 0.8, color_white, 2)

    # 2. Estado de detección de la mano
    if hand_detected:
        hand_text = "MANO: DETECTADA"
        hand_color = color_green
    else:
        hand_text = "MANO: NO DETECTADA"
        hand_color = color_red
    cv2.putText(frame, hand_text, (20, 80), font, 0.7, hand_color, 2)

    # 3. Barra de progreso del búfer circular (en la parte inferior)
    bar_w, bar_h = 400, 20
    bar_x = (w - bar_w) // 2
    bar_y = h - 50

    # Dibujar fondo de la barra de progreso (gris oscuro)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_gray, -1)

    # Dibujar porción llena (verde)
    if buffer_len > 0:
        fill_w = int((buffer_len / SEQUENCE_LENGTH) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_green, -1)

    # Dibujar borde de la barra de progreso (contorno blanco)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_white, 1)

    # 4. Marcador de guardado: cuando el búfer esté lleno, dibujar línea de paso en la posición de SAVE_EVERY
    if buffer_len == SEQUENCE_LENGTH:
        marker_x = bar_x + int((SAVE_EVERY / SEQUENCE_LENGTH) * bar_w)
        cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_h), color_white, 2)

    # 5. Flash de confirmación o mensaje de indicación justo arriba de la barra de progreso
    if flash_timer > 0:
        flash_text = "¡SECUENCIA GUARDADA!"
        flash_color = color_green
    else:
        flash_text = "Mueve la mano para registrar el gesto..."
        flash_color = color_white

    (flash_w, _), _ = cv2.getTextSize(flash_text, font, 0.7, 2)
    flash_x = (w - flash_w) // 2
    cv2.putText(frame, flash_text, (flash_x, bar_y - 15), font, 0.7, flash_color, 2)


# ---------------------------------------------------------------------------
# Step 5 — Output folder creation with resume support
# ---------------------------------------------------------------------------
def pedir_nombre_gesto() -> tuple[str | None, int | None]:
    """
    Solicita el nombre del gesto, crea la carpeta correspondiente y detecta cuántas
    secuencias ya existen para reanudar el registro desde el índice correcto sin sobrescribir.

    Retorna:
        (nombre_normalizado, siguiente_indice) o (None, None) en caso de error.
    """
    try:
        gesture_name = input("Ingrese el nombre del gesto a registrar: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nOperación cancelada por el usuario.")
        return None, None

    if not gesture_name:
        print("Error: El nombre del gesto no puede estar vacío.")
        return None, None

    # Resolver el directorio de salida
    output_dir = PROJECT_ROOT / "gestos" / gesture_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Contar los archivos .npy existentes para reanudar el registro secuencialmente
    existing_files = list(output_dir.glob("*.npy"))
    next_index = len(existing_files)

    if next_index > 0:
        print(f"Carpeta existente detectada. Reanudando desde la secuencia {next_index}.")
    else:
        print(f"Carpeta creada. Iniciando desde la secuencia 0.")

    return gesture_name, next_index


# ---------------------------------------------------------------------------
# Step 6 — Main loop: wait → countdown → automatic recording
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, start_index: int, landmarker: vision.HandLandmarker) -> None:
    """
    Fase 0 — Esperar: la cámara está activa pero NO recopila datos hasta presionar ESPACIO.
    Fase 1 — Cuenta atrás: muestra 3-2-1 para que el usuario posicione la mano.
    Fase 2 — Registro automático: el búfer circular guarda una secuencia cada
             SAVE_EVERY fotogramas si se detecta una mano, sin entrada adicional.

    Convención de nombres: gestos/<gesto>/<indice_secuencia>.npy
    """
    output_dir = PROJECT_ROOT / "gestos" / gesture_name
    sequences_saved = start_index

    # Inicializar la captura de video (el índice 0 suele ser la webcam por defecto)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Configurar resolución de captura estándar
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "GestureFlow - Recolección de Datos"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    # -----------------------------------------------------------------------
    # Fase 0 — Bucle de Espera
    # -----------------------------------------------------------------------
    print("Fase 0: Esperando por ESPACIO para comenzar a grabar...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro de la cámara.")
            break

        frame = cv2.flip(frame, 1)  # Espejar fotograma
        draw_waiting(frame, gesture_name, sequences_saved) 
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF # Espera que se presione una tecla
        if key == ord(' '): # si se presiona la barra espaciadora
            break
        elif key == ord('q'): # si se presiona la tecla q
            print("Grabación cancelada en fase de espera.")
            cap.release()
            cv2.destroyAllWindows()
            return

    # -----------------------------------------------------------------------
    # Fase 1 — Bucle de Cuenta Atrás
    # -----------------------------------------------------------------------
    print("Fase 1: Iniciando cuenta atrás...")
    for sec in range(COUNTDOWN_SECS, 0, -1):
        deadline = time.time() + 1.0 # Espera 1 segundo
        while time.time() < deadline: # Mientras no pase 1 segundo
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            draw_countdown(frame, gesture_name, sec)
            cv2.imshow(window_name, frame) 

            key = cv2.waitKey(1) & 0xFF # Espera que se presione una tecla
            if key == ord('q'): # si se presiona la tecla q
                print("Grabación cancelada durante cuenta atrás.")
                cap.release()
                cv2.destroyAllWindows()
                return

    # -----------------------------------------------------------------------
    # Fase 2 — Bucle de Grabación Automática
    # -----------------------------------------------------------------------
    print("Fase 2: Grabando secuencias automáticamente...")
    buffer: deque = deque(maxlen=SEQUENCE_LENGTH) # Búfer circular de longitud fija
    frame_counter = 0
    flash_timer = 0
    start_time = time.time()

    while sequences_saved < NUM_SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro de la cámara.")
            break

        frame = cv2.flip(frame, 1)

        # Preprocesar fotograma para MediaPipe HandLandmarker
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Ejecutar inferencia síncrona en modo VIDEO con timestamp
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extraer características (126 coordenadas) y añadir al búfer circular
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)
        frame_counter += 1 

        hand_detected = bool(results.hand_landmarks)

        # Guardado automático: búfer lleno + mano visible + intervalo de fotogramas cumplido
        #validacion que permite que el modelo detecte la mano
        if len(buffer) == SEQUENCE_LENGTH and hand_detected and (frame_counter % SAVE_EVERY == 0):
            sequence_data = np.array(buffer, dtype=np.float32) # Convierte el búfer a un array de NumPy y lo guarda como un archivo .npy
            file_path = output_dir / f"{sequences_saved}.npy" # Crea la ruta del archivo .npy con el índice de la secuencia
            np.save(str(file_path), sequence_data) # Guarda el array de NumPy en el archivo .npy
            print(f"Secuencia {sequences_saved} guardada exitosamente.") # Imprime el mensaje de que la secuencia se guardó exitosamente
            sequences_saved += 1 # Incrementa el contador de secuencias guardadas
            flash_timer = FLASH_DURATION # Reinicia el temporizador de flash

        # Renderizar capas del HUD
        draw_hud(
            frame,
            gesture_name,
            sequences_saved,
            len(buffer),
            frame_counter,
            hand_detected,
            flash_timer
        )

        if flash_timer > 0:
            flash_timer -= 1

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Grabación interrumpida por el usuario.")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print(f"Grabación terminada. Total de secuencias guardadas: {sequences_saved}/{NUM_SEQUENCES}")


# ---------------------------------------------------------------------------
# Punto de Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Paso 1: Solicitar nombre de gesto y detectar índice de reanudación
    gesto_creado, next_index = pedir_nombre_gesto()
    if gesto_creado is None or next_index is None:
        exit(1)

    # Paso 2: Construir el HandLandmarker de MediaPipe (modo IMAGE)
    with build_landmarker() as landmarker:
        # Paso 3: Esperar → cuenta atrás → registro automático
        grabar_gesto(gesto_creado, next_index, landmarker)
