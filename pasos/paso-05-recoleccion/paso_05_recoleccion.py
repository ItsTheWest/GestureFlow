from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Path resolution — same pattern as previous steps
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH   = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

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
       running_mode=vision.RunningMode.IMAGE,  
       num_hands=2,                         
    )

    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Step 2 — Extracción de Keypoints
# ---------------------------------------------------------------------------
def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    left_hand = np.zeros(63, dtype=np.float32) # Inicializa la mano izquierda en ceros
    right_hand = np.zeros(63, dtype=np.float32) # Inicializa la mano derecha en ceros

    if results.hand_landmarks and results.handedness: # Si se detectan manos
        for idx, hand_info in enumerate(results.handedness): # Recorre las manos detectadas
            # Extracción de la mano izquierda o derecha
            hand_label = hand_info[0].category_name
            
            # Extracción de los landmarks y aplanamiento
            landmarks = results.hand_landmarks[idx]
            flat_coords = [] # Lista para almacenar las coordenadas de los landmarks
            for lm in landmarks:
                flat_coords.extend([lm.x, lm.y, lm.z]) # Se añade cada coordenada
            
            # Colocación en el slot correcto
            if hand_label == "Left":
                left_hand = np.array(flat_coords, dtype=np.float32)
            elif hand_label == "Right":
                right_hand = np.array(flat_coords, dtype=np.float32)

    return np.concatenate([left_hand, right_hand])


# ---------------------------------------------------------------------------
# Step 3a — HUD rendering for the waiting phase (before SPACE)
# ---------------------------------------------------------------------------
def draw_waiting(frame: np.ndarray, gesture: str, saved: int) -> None:
    """Show the waiting state: camera active but not collecting yet."""
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
    """Show the gesture name and the countdown number centered on screen."""
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
    Show recording status overlay on the camera frame.

    Elements:
      · Gesture name and saved sequence counter (top)
      · Hand detection indicator
      · Rolling buffer progress bar with next auto-save marker
      · Confirmation flash when a sequence is saved
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # BGR Color definitions
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_white = (255, 255, 255)
    color_gray = (50, 50, 50)

    # 1. Top HUD bar: REC Indicator (left) & Saved counter (right)
    rec_text = f"REC: {gesture.upper()}"
    cv2.putText(frame, rec_text, (20, 40), font, 0.8, color_red, 2)

    progress_text = f"GUARDADO: {saved}/{NUM_SEQUENCES}"
    (prog_w, _), _ = cv2.getTextSize(progress_text, font, 0.8, 2)
    cv2.putText(frame, progress_text, (w - prog_w - 20, 40), font, 0.8, color_white, 2)

    # 2. Hand detection status
    if hand_detected:
        hand_text = "MANO: DETECTADA"
        hand_color = color_green
    else:
        hand_text = "MANO: NO DETECTADA"
        hand_color = color_red
    cv2.putText(frame, hand_text, (20, 80), font, 0.7, hand_color, 2)

    # 3. Buffer Progress Bar (at the bottom)
    bar_w, bar_h = 400, 20
    bar_x = (w - bar_w) // 2
    bar_y = h - 50

    # Draw progress bar background (dark gray)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_gray, -1)

    # Draw filled portion (green)
    if buffer_len > 0:
        fill_w = int((buffer_len / SEQUENCE_LENGTH) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_green, -1)

    # Draw progress bar border (white outline)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_white, 1)

    # 4. Save marker: when buffer is full, draw stride line at SAVE_EVERY offset
    if buffer_len == SEQUENCE_LENGTH:
        marker_x = bar_x + int((SAVE_EVERY / SEQUENCE_LENGTH) * bar_w)
        cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_h), color_white, 2)

    # 5. Flash confirmation or prompt just above the progress bar
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
#def pedir_nombre_gesto() -> tuple[str, int] | tuple[None, None]:
    """
    Ask for the gesture name, create the folder, and detect how many sequences
    already exist so we can resume from the correct index without overwriting.

    Returns:
        (normalized_name, next_index) or (None, None) on error.
    """
    # TODO: Read gesture name from input(), normalize (strip + lowercase)
    # TODO: Validate non-empty
    # TODO: Create the folder at PROJECT_ROOT / "gestos" / normalized
    # TODO: Count existing .npy files to determine the resume index
    # TODO: Print resume info if files exist, else print "folder ready"
    # TODO: Return (normalized, next_index) or (None, None) on error
    pass  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 6 — Main loop: wait → countdown → automatic recording
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, start_index: int, landmarker: vision.HandLandmarker) -> None:
    """
    Phase 0 — Wait: camera is active but does NOT collect until SPACE is pressed.
    Phase 1 — Countdown: displays 3-2-1 so the user can position their hand.
    Phase 2 — Automatic recording: the rolling buffer saves a sequence every
              SAVE_EVERY frames when a hand is detected, no further input needed.

    File naming convention: gestos/<gesture>/<sequence_index>.npy
    """
    # TODO: Setup — compute output_dir, sequences_saved, sequences_needed
    # TODO: Open camera (cv2.VideoCapture), set resolution, create window

    # TODO: Phase 0 — Wait loop: read frames, draw_waiting(), show frame
    #       Break on SPACE, quit on Q

    # TODO: Phase 1 — Countdown loop: for each second, draw_countdown()
    #       Quit on Q during countdown

    # TODO: Phase 2 — Recording loop:
    #       Create deque(maxlen=SEQUENCE_LENGTH) as rolling buffer
    #       Initialize frame_counter and flash_timer
    #       Loop while sequences_saved < NUM_SEQUENCES:
    #         - Read frame, flip, convert BGR→RGB, run landmarker.detect()
    #         - Extract keypoints, append to buffer, increment frame_counter
    #         - Auto-save condition: buffer full + hand visible + frame_counter % SAVE_EVERY == 0
    #         - Save as .npy, increment sequences_saved, trigger flash
    #         - Draw HUD, show frame, handle Q key

    # TODO: Cleanup — release camera, destroy windows, print summary
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
#if __name__ == "__main__":
    # Step 1: Ask for gesture name and detect resume index
    #gesto_creado, next_index = pedir_nombre_gesto()
    #if gesto_creado is None or next_index is None:
        #exit(1)

    # Step 2: Build the MediaPipe HandLandmarker (IMAGE mode)
    #with build_landmarker() as landmarker:
        # Step 3: Wait → countdown → automatic recording
        #grabar_gesto(gesto_creado, next_index, landmarker)
