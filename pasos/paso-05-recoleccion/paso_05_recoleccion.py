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
NUM_SEQUENCES    = 200  # How many examples we collect per gesture

# How often (in frames) we auto-save a sequence.
# With SEQUENCE_LENGTH=30 and SAVE_EVERY=5 there is 83% overlap,
# but the guided phases ensure continuous variations are captured.
SAVE_EVERY       = 5

# Seconds of countdown before automatic recording begins
# We set it to 5 seconds to give the user enough time to prepare
COUNTDOWN_SECS   = 5

# Frames the confirmation flash lasts in the HUD (~0.5 s at 30 fps)
FLASH_DURATION   = 15

# Guided phases for data collection to introduce variability (Left & Right hands)
PHASES: list[tuple[str, range, str]] = [
    ("Derecha - Base", range(0, 20), "Mano DERECHA: Mantenla estable en posicion comoda."),
    ("Derecha - Velocidad", range(20, 40), "Mano DERECHA: Realiza movimientos rapidos y lentos."),
    ("Derecha - Distancia", range(40, 60), "Mano DERECHA: Acerca y aleja la mano de la camara."),
    ("Derecha - Angulo", range(60, 80), "Mano DERECHA: Rota e inclina la muneca a los lados."),
    ("Derecha - Posicion", range(80, 100), "Mano DERECHA: Desplaza la mano por todo el cuadro."),
    ("Izquierda - Base", range(100, 120), "Mano IZQUIERDA: Mantenla estable en posicion comoda."),
    ("Izquierda - Velocidad", range(120, 140), "Mano IZQUIERDA: Realiza movimientos rapidos y lentos."),
    ("Izquierda - Distancia", range(140, 160), "Mano IZQUIERDA: Acerca y aleja la mano de la camara."),
    ("Izquierda - Angulo", range(160, 180), "Mano IZQUIERDA: Rota e inclina la muneca a los lados."),
    ("Izquierda - Posicion", range(180, 200), "Mano IZQUIERDA: Desplaza la mano por todo el cuadro."),
]


# ---------------------------------------------------------------------------
# Step 1 — MediaPipe configuration (IMAGE mode = synchronous, no callback)
# ---------------------------------------------------------------------------
def build_landmarker() -> vision.HandLandmarker:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
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
    """Displays the waiting state: camera active but not yet collecting data."""
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

    #Draw the text onto the frame
    cv2.putText(frame, gesture_text, (20, 40), font, font_scale_hud, color_green, thickness_hud)
    cv2.putText(frame, progress_text, (20, 80), font, font_scale_hud, color_white, thickness_hud) # 
    cv2.putText(frame, quit_text, (20, 120), font, font_scale_hud, color_white, thickness_hud)

    #Get the text size of the instruction label
    (text_w, text_h), _ = cv2.getTextSize(instruction_text, font, font_scale_instruction, thickness_instruction)
    
    #Calculate the center of the frame to position the text
    text_x = (w - text_w) // 2
    text_y = (h + text_h) // 2

    # Draw the instruction text
    cv2.putText(frame, instruction_text, (text_x, text_y), font, font_scale_instruction, color_green, thickness_instruction)

# ---------------------------------------------------------------------------
# Step 3b — HUD rendering for the countdown phase
# ---------------------------------------------------------------------------
def draw_countdown(frame: np.ndarray, gesture: str, seconds_left: int) -> None:
    """Displays the gesture name and the countdown number centered on the screen."""
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
    prepare_text = "PREPARE YOUR GESTURE..."
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
    Renders the recording HUD over the camera frame.

    Elements:
      · Gesture name and saved sequence counter (top)
      · Hand detection indicator
      · Circular buffer progress bar with save marker
      · Confirmation flash when a sequence is saved
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Color definitions in BGR
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_white = (255, 255, 255)
    color_gray = (50, 50, 50)

    # 1. Top HUD bar: REC indicator (left) and saved counter (right)
    rec_text = f"REC: {gesture.upper()}"
    cv2.putText(frame, rec_text, (20, 40), font, 0.8, color_red, 2)

    progress_text = f"GUARDADO: {saved}/{NUM_SEQUENCES}"
    (prog_w, _), _ = cv2.getTextSize(progress_text, font, 0.8, 2)
    cv2.putText(frame, progress_text, (w - prog_w - 20, 40), font, 0.8, color_white, 2)

    # 2. Hand detection status
    if hand_detected:
        hand_text = "HAND: DETECTED"
        hand_color = color_green
    else:
        hand_text = "HAND: NOT DETECTED"
        hand_color = color_red
    cv2.putText(frame, hand_text, (20, 80), font, 0.7, hand_color, 2)

    # 2b. Current phase and variation instructions
    current_phase_name = "Completado"
    for name, r, _ in PHASES:
        if saved in r:
            current_phase_name = name
            break
    phase_text = f"FASE: {current_phase_name}"
    cv2.putText(frame, phase_text, (20, 120), font, 0.7, (255, 255, 0), 2)

    # 3. Circular buffer progress bar (at the bottom)
    bar_w, bar_h = 400, 20
    bar_x = (w - bar_w) // 2
    bar_y = h - 50

    # Draw the progress bar background (dark gray)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_gray, -1)

    # Draw the filled portion (green)
    if buffer_len > 0:
        fill_w = int((buffer_len / SEQUENCE_LENGTH) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_green, -1)

    # Draw the progress bar border (white outline)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_white, 1)

    # 4. Save marker: when the buffer is full, draw a step line at the SAVE_EVERY position
    if buffer_len == SEQUENCE_LENGTH:
        marker_x = bar_x + int((SAVE_EVERY / SEQUENCE_LENGTH) * bar_w)
        cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_h), color_white, 2)

    # 5. Confirmation flash or hint message just above the progress bar
    if flash_timer > 0:
        flash_text = "SEQUENCE SAVED!"
        flash_color = color_green
    else:
        flash_text = "Move your hand to record the gesture..."
        flash_color = color_white

    (flash_w, _), _ = cv2.getTextSize(flash_text, font, 0.7, 2)
    flash_x = (w - flash_w) // 2
    cv2.putText(frame, flash_text, (flash_x, bar_y - 15), font, 0.7, flash_color, 2)


# ---------------------------------------------------------------------------
# Step 5 — Output folder creation with resume support
# ---------------------------------------------------------------------------
def pedir_nombre_gesto() -> tuple[str | None, int | None]:
    """
    Prompt for a gesture name, create the output folder and detect how many
    sequences already exist to resume recording from the correct index without overwriting.

    Returns:
        (normalized_name, next_index) or (None, None) on error.
    """
    try:
        gesture_name = input("Enter the gesture name to record: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        return None, None

    if not gesture_name:
        print("Error: Gesture name cannot be empty.")
        return None, None

    # Resolve the output directory
    output_dir = PROJECT_ROOT / "gestos" / gesture_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing .npy files to resume recording sequentially
    existing_files = list(output_dir.glob("*.npy"))
    next_index = len(existing_files)

    if next_index > 0:
        print(f"Existing folder detected. Resuming from sequence {next_index}.")
    else:
        print(f"Folder created. Starting from sequence 0.")

    return gesture_name, next_index


# ---------------------------------------------------------------------------
# Step 6 — Main loop: wait → countdown → automatic recording
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, start_index: int, landmarker: vision.HandLandmarker) -> None:
    """
    Phase 0 — Wait: camera is active but does NOT collect data until SPACE is pressed.
    Phase 1 — Countdown: shows 5-4-3-2-1 so the user can position their hand.
    Phase 2 — Automatic recording: circular buffer saves a sequence every
             SAVE_EVERY frames if a hand is detected, without additional input.

    Naming convention: gestos/<gesture>/<sequence_index>.npy
    """
    output_dir = PROJECT_ROOT / "gestos" / gesture_name
    sequences_saved = start_index

    # Initialize video capture (index 0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Set standard capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "GestureFlow - Recolección de Datos"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    # -----------------------------------------------------------------------
    # Phase 0 — Waiting Loop
    # -----------------------------------------------------------------------
    print("Phase 0: Waiting for SPACE to start recording...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro de la cámara.")
            break

        frame = cv2.flip(frame, 1)  # Mirror frame
        draw_waiting(frame, gesture_name, sequences_saved) 
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF # Wait for a key press
        if key == ord(' '): # if space bar is pressed
            break
        elif key == ord('q'): # if q key is pressed
            print("Recording cancelled in waiting phase.")
            cap.release()
            cv2.destroyAllWindows()
            return

    # -----------------------------------------------------------------------
    # Phase 1 — Countdown Loop
    # -----------------------------------------------------------------------
    print("Phase 1: Starting countdown...")
    for sec in range(COUNTDOWN_SECS, 0, -1):
        deadline = time.time() + 1.0 # Wait 1 second
        while time.time() < deadline: # While 1 second has not elapsed
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            draw_countdown(frame, gesture_name, sec)
            cv2.imshow(window_name, frame) 

            key = cv2.waitKey(1) & 0xFF # Wait for a key press
            if key == ord('q'): # if q key is pressed
                print("Recording cancelled during countdown.")
                cap.release()
                cv2.destroyAllWindows()
                return

    # -----------------------------------------------------------------------
    # Phase 2 — Automatic Recording Loop
    # -----------------------------------------------------------------------
    print("Phase 2: Recording sequences automatically...")
    buffer: deque = deque(maxlen=SEQUENCE_LENGTH) # Fixed-length circular buffer
    frame_counter = 0
    flash_timer = 0
    start_time = time.time()

    while sequences_saved < NUM_SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el cuadro de la cámara.")
            break

        frame = cv2.flip(frame, 1)

        # Preprocess frame for MediaPipe HandLandmarker
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Run synchronous inference in VIDEO mode with timestamp
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extract features (126 coordinates) and append to the circular buffer
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)
        frame_counter += 1 

        hand_detected = bool(results.hand_landmarks)

        # Automatic save: buffer full + hand visible + frame interval met
        # Validation that allows the model to detect the hand
        if len(buffer) == SEQUENCE_LENGTH and hand_detected and (frame_counter % SAVE_EVERY == 0):
            sequence_data = np.array(buffer, dtype=np.float32) # Convert the buffer to a NumPy array and save as a .npy file
            file_path = output_dir / f"{sequences_saved}.npy" # Build the .npy file path with the sequence index
            np.save(str(file_path), sequence_data) # Save the NumPy array to the .npy file
            print(f"Sequence {sequences_saved} saved successfully.") # Print success message
            sequences_saved += 1 # Increment the saved sequence counter
            flash_timer = FLASH_DURATION # Reset the flash timer

        # Render HUD layers
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
            print("Recording interrupted by user.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Recording finished. Total sequences saved: {sequences_saved}/{NUM_SEQUENCES}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Prompt for gesture name and detect resume index
    gesto_creado, next_index = pedir_nombre_gesto()
    if gesto_creado is None or next_index is None:
        exit(1)

    # Step 2: Build the MediaPipe HandLandmarker (IMAGE mode)
    with build_landmarker() as landmarker:
        # Step 3: Wait → countdown → automatic recording
        grabar_gesto(gesto_creado, next_index, landmarker)
