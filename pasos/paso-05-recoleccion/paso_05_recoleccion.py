"""Dataset collection pipeline for GestureFlow.

Captures camera frames, extracts hand landmark coordinates, and persists
sequences of hand movements to train the LSTM classification model.
Exposes a CollectionManager class for easy integration into dashboards.
"""

from collections import deque
from pathlib import Path
import time
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import config
from utils import HAND_CONNECTIONS, extract_keypoints

# ── Parameters bound to shared configuration ───────────────────────────────────
PROJECT_ROOT: Path = config.PROJECT_ROOT
MODEL_PATH: Path = config.MP_TASK_PATH
SEQUENCE_LENGTH: int = config.SEQUENCE_LENGTH
NUM_FEATURES: int = config.NUM_FEATURES
NUM_SEQUENCES: int = config.NUM_SEQUENCES
SAVE_EVERY: int = config.SAVE_EVERY
COUNTDOWN_SECS: int = config.COUNTDOWN_SECS
FLASH_DURATION: int = config.FLASH_DURATION

# Guided phases for data collection to introduce variability (Left & Right hands)
PHASES: list[tuple[str, range, str]] = [
    ("Derecha - Base", range(0, 20), "Mano DERECHA: Realiza el gesto a velocidad normal de forma natural."),
    ("Derecha - Velocidad", range(20, 40), "Mano DERECHA: Realiza el gesto alternando muy rapido y muy lento."),
    ("Derecha - Distancia", range(40, 60), "Mano DERECHA: Realiza el gesto acercando y alejando la mano de la camara."),
    ("Derecha - Angulo", range(60, 80), "Mano DERECHA: Realiza el gesto rotando e inclinando la muneca a los lados."),
    ("Derecha - Posicion", range(80, 100), "Mano DERECHA: Realiza el gesto moviendo la mano por todo el cuadro."),
    ("Izquierda - Base", range(100, 120), "Mano IZQUIERDA: Realiza el gesto a velocidad normal de forma natural."),
    ("Izquierda - Velocidad", range(120, 140), "Mano IZQUIERDA: Realiza el gesto alternando muy rapido y muy lento."),
    ("Izquierda - Distancia", range(140, 160), "Mano IZQUIERDA: Realiza el gesto acercando y alejando la mano de la camara."),
    ("Izquierda - Angulo", range(160, 180), "Mano IZQUIERDA: Realiza el gesto rotando e inclinando la muneca a los lados."),
    ("Izquierda - Posicion", range(180, 200), "Mano IZQUIERDA: Realiza el gesto moviendo la mano por todo el cuadro."),
]


def build_landmarker() -> vision.HandLandmarker:
    """Build and configure the MediaPipe HandLandmarker in Video running mode.

    Returns:
        The instantiated HandLandmarker object.
    """
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


def draw_waiting(frame: np.ndarray, gesture: str, saved: int) -> None:
    """Display the waiting state: camera active but not collecting data.

    Args:
        frame: BGR frame to draw status text onto.
        gesture: Name of the active gesture class.
        saved: Current progress count.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_hud = 0.8
    font_scale_instruction = 1.0
    thickness_hud = 2
    thickness_instruction = 2

    color_green = (0, 255, 0)
    color_white = (255, 255, 255)

    gesture_text = f"Gesto: {gesture.upper()}"
    progress_text = f"Progreso: {saved}/{NUM_SEQUENCES}"
    instruction_text = "PRESIONA ESPACIO PARA EMPEZAR"
    quit_text = "Q: Salir"

    cv2.putText(frame, gesture_text, (20, 40), font, font_scale_hud, color_green, thickness_hud)
    cv2.putText(frame, progress_text, (20, 80), font, font_scale_hud, color_white, thickness_hud)
    cv2.putText(frame, quit_text, (20, 120), font, font_scale_hud, color_white, thickness_hud)

    (text_w, text_h), _ = cv2.getTextSize(instruction_text, font, font_scale_instruction, thickness_instruction)
    text_x = (w - text_w) // 2
    text_y = (h + text_h) // 2
    cv2.putText(frame, instruction_text, (text_x, text_y), font, font_scale_instruction, color_green, thickness_instruction)


def draw_countdown(frame: np.ndarray, gesture: str, seconds_left: int) -> None:
    """Display the countdown overlay centered on the screen.

    Args:
        frame: BGR frame to draw status text onto.
        gesture: Name of the active gesture class.
        seconds_left: Remaining seconds on countdown.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    gesture_text = f"Gesto: {gesture.upper()}"
    cv2.putText(frame, gesture_text, (20, 40), font, 0.8, (0, 255, 0), 2)

    quit_text = "Q: Salir"
    cv2.putText(frame, quit_text, (20, 80), font, 0.8, (255, 255, 255), 2)

    number_text = str(seconds_left)
    font_scale_num = 6.0
    thickness_num = 12
    (num_w, num_h), _ = cv2.getTextSize(number_text, font, font_scale_num, thickness_num)
    num_x = (w - num_w) // 2
    num_y = (h + num_h) // 2
    cv2.putText(frame, number_text, (num_x, num_y), font, font_scale_num, (0, 255, 0), thickness_num)

    prepare_text = "PREPARA TU GESTO..."
    font_scale_prep = 0.8
    thickness_prep = 2
    (prep_w, prep_h), _ = cv2.getTextSize(prepare_text, font, font_scale_prep, thickness_prep)
    prep_x = (w - prep_w) // 2
    prep_y = h - 60
    cv2.putText(frame, prepare_text, (prep_x, prep_y), font, font_scale_prep, (255, 255, 255), thickness_prep)


def draw_pause_notification(frame: np.ndarray, next_name: str, next_desc: str) -> None:
    """Draw a tutorial overlay panel at the top when dataset collection is paused.

    Args:
        frame: BGR frame to draw status text onto.
        next_name: Name of the upcoming recording phase.
        next_desc: Instructions for the upcoming phase.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    card_w = int(w * 0.9)
    card_h = 85
    card_x = (w - card_w) // 2
    card_y = 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 25, 25), -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 255, 255), 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "[i] INSTRUCCIONES - SIGUIENTE FASE", (card_x + 15, card_y + 20), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Fase: {next_name.upper()}", (card_x + 15, card_y + 42), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, next_desc, (card_x + 15, card_y + 62), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "[ESPACIO] Continuar", (card_x + card_w - 145, card_y + 20), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    gesture: str,
    saved: int,
    buffer_len: int,
    frame_counter: int,
    hand_detected: bool,
    flash_timer: int,
) -> None:
    """Render the detailed recording HUD interface layers.

    Args:
        frame: BGR frame to draw status text onto.
        gesture: Active gesture class.
        saved: Sequences recorded so far.
        buffer_len: Length of sequence buffer deque.
        frame_counter: Relative frame index.
        hand_detected: True if hand is detected.
        flash_timer: Non-zero to flash confirmation text.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_white = (255, 255, 255)
    color_gray = (50, 50, 50)

    rec_text = f"REC: {gesture.upper()}"
    cv2.putText(frame, rec_text, (20, 40), font, 0.8, color_red, 2)

    progress_text = f"GUARDADO: {saved}/{NUM_SEQUENCES}"
    (prog_w, _), _ = cv2.getTextSize(progress_text, font, 0.8, 2)
    cv2.putText(frame, progress_text, (w - prog_w - 20, 40), font, 0.8, color_white, 2)

    hand_text = "MANO: DETECTADA" if hand_detected else "MANO: NO DETECTADA"
    hand_color = color_green if hand_detected else color_red
    cv2.putText(frame, hand_text, (20, 80), font, 0.7, hand_color, 2)

    current_phase_name = "Completado"
    for name, r, _ in PHASES:
        if saved in r:
            current_phase_name = name
            break

    cv2.putText(frame, f"FASE: {current_phase_name.upper()}", (20, 120), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    bar_w, bar_h = 400, 20
    bar_x = (w - bar_w) // 2
    bar_y = h - 50

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_gray, -1)

    if buffer_len > 0:
        fill_w = int((buffer_len / SEQUENCE_LENGTH) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_green, -1)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color_white, 1)

    if buffer_len == SEQUENCE_LENGTH:
        marker_x = bar_x + int((SAVE_EVERY / SEQUENCE_LENGTH) * bar_w)
        cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_h), color_white, 2)

    if flash_timer > 0:
        flash_text = "SEQUENCE SAVED!"
        flash_color = color_green
    else:
        flash_text = "Mueve la mano para grabar el gesto..."
        flash_color = color_white

    (flash_w, _), _ = cv2.getTextSize(flash_text, font, 0.7, 2)
    flash_x = (w - flash_w) // 2
    cv2.putText(frame, flash_text, (flash_x, bar_y - 15), font, 0.7, flash_color, 2)


class CollectionManager:
    """Manages the state machine, buffer processing, and drawing for dataset collection."""

    def __init__(self, gesture_name: str, start_index: int) -> None:
        """Initialize the collection manager state.

        Args:
            gesture_name: Name of the gesture class being recorded.
            start_index: Sequence index to resume recording from.
        """
        self.gesture_name: str = gesture_name
        self.sequences_saved: int = start_index
        self.buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_counter: int = 0
        self.flash_timer: int = 0
        self.state: str = "Waiting"  # States: Waiting, Countdown, Recording, Paused
        self.countdown_start: float = 0.0
        self.start_time: float = 0.0
        self.is_active: bool = True

        self.current_phase_idx: int = 0
        for idx, (_, r, _) in enumerate(PHASES):
            if self.sequences_saved in r:
                self.current_phase_idx = idx
                break

    def handle_space(self) -> None:
        """Handle spacebar press for state transitions."""
        if self.state == "Waiting":
            self.state = "Countdown"
            self.countdown_start = time.time()
        elif self.state == "Paused":
            self.state = "Recording"
            self.frame_counter = 0
            self.buffer.clear()
            self.start_time = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        results: vision.HandLandmarkerResult,
        timestamp_ms: int,
        space_pressed: bool = False,
        new_results: bool = True
    ) -> np.ndarray:
        """Process a camera frame and update the collection state machine.

        Args:
            frame: Camera frame in BGR format.
            results: MediaPipe HandLandmarker inference results.
            timestamp_ms: Monotonic timestamp for video mode tracking.
            space_pressed: True if the space key was triggered.
            new_results: True if the results parameter contains new hand coordinates.

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        if not self.is_active:
            return frame

        if space_pressed:
            self.handle_space()

        h, w = frame.shape[:2]
        display = frame.copy()

        # Phase 0: Waiting
        if self.state == "Waiting":
            draw_waiting(display, self.gesture_name, self.sequences_saved)

        # Phase 1: Countdown
        elif self.state == "Countdown":
            elapsed = time.time() - self.countdown_start
            sec_left = COUNTDOWN_SECS - int(elapsed)
            if sec_left <= 0:
                self.state = "Recording"
                self.buffer.clear()
                self.frame_counter = 0
                self.start_time = time.time()
                return self.process_frame(frame, results, timestamp_ms, space_pressed=False, new_results=new_results)
            else:
                draw_countdown(display, self.gesture_name, sec_left)

        # Phase 2: Recording
        elif self.state == "Recording":
            hand_detected = False
            if results and results.hand_landmarks:
                hand_detected = True
                for hand_landmarks_list in results.hand_landmarks:
                    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks_list]
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(coords) and end_idx < len(coords):
                            cv2.line(display, coords[start_idx], coords[end_idx], (0, 255, 0), 2)
                    for coord in coords:
                        cv2.circle(display, coord, 3, (0, 0, 255), -1)

            if new_results:
                keypoints = extract_keypoints(results)
                self.buffer.append(keypoints)
                self.frame_counter += 1

                if len(self.buffer) == SEQUENCE_LENGTH and hand_detected and (self.frame_counter % SAVE_EVERY == 0):
                    output_dir = PROJECT_ROOT / "gestos" / self.gesture_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    sequence_data = np.array(self.buffer, dtype=np.float32)
                    file_path = output_dir / f"{self.sequences_saved}.npy"
                    np.save(str(file_path), sequence_data)

                    self.sequences_saved += 1
                    self.flash_timer = FLASH_DURATION

                    next_phase_idx = -1
                    for idx, (_, r, _) in enumerate(PHASES):
                        if self.sequences_saved in r:
                            next_phase_idx = idx
                            break

                    if next_phase_idx != -1 and next_phase_idx > self.current_phase_idx:
                        self.current_phase_idx = next_phase_idx
                        self.buffer.clear()
                        self.state = "Paused"

                    if self.sequences_saved >= NUM_SEQUENCES:
                        self.is_active = False
                        self.state = "Waiting"

            draw_hud(
                display,
                self.gesture_name,
                self.sequences_saved,
                len(self.buffer),
                self.frame_counter,
                hand_detected,
                self.flash_timer
            )

            if self.flash_timer > 0:
                self.flash_timer -= 1

        # Phase 3: Paused
        elif self.state == "Paused":
            if results and results.hand_landmarks:
                for hand_landmarks_list in results.hand_landmarks:
                    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks_list]
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(coords) and end_idx < len(coords):
                            cv2.line(display, coords[start_idx], coords[end_idx], (0, 255, 0), 2)

            next_name, _, next_desc = PHASES[self.current_phase_idx]
            draw_pause_notification(display, next_name, next_desc)

        return display


def pedir_nombre_gesto(arg_name: str | None = None) -> tuple[str | None, int | None]:
    """Resolve the gesture name and find resume sequence index.

    Args:
        arg_name: Optional pre-defined name.

    Returns:
        (gesture_name, next_index) or (None, None) on error.
    """
    if arg_name is None:
        try:
            gesture_name = input("Enter the gesture name to record: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return None, None
    else:
        gesture_name = arg_name.strip().lower()

    if not gesture_name:
        print("Error: Gesture name cannot be empty.")
        return None, None

    output_dir = PROJECT_ROOT / "gestos" / gesture_name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(output_dir.glob("*.npy"))
    if existing_files:
        indices: list[int] = []
        for f in existing_files:
            try:
                indices.append(int(f.stem))
            except ValueError:
                pass
        next_index = max(indices) + 1 if indices else 0
    else:
        next_index = 0

    if next_index > 0:
        print(f"Existing folder detected. Resuming from sequence {next_index}.")
    else:
        print(f"Folder created. Starting from sequence 0.")

    return gesture_name, next_index


def grabar_gesto(gesture_name: str, start_index: int, landmarker: vision.HandLandmarker) -> None:
    """Launch the main OpenCV video loop for standalone collection.

    Args:
        gesture_name: Name of the gesture class.
        start_index: Index to begin recording.
        landmarker: Pre-configured MediaPipe HandLandmarker.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "GestureFlow - Recolección de Datos"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    manager = CollectionManager(gesture_name, start_index)
    start_time = time.time()

    print("System active. Press SPACE inside the OpenCV window to begin countdown, Q to quit.")

    while manager.is_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        key = cv2.waitKey(1) & 0xFF
        space_pressed = (key == ord(' '))
        if key == ord('q'):
            print("Recording cancelled by user.")
            break

        display = manager.process_frame(frame, results, timestamp_ms, space_pressed)
        cv2.imshow(window_name, display)

    cap.release()
    cv2.destroyAllWindows()


def main(gesture_name: str | None = None) -> None:
    """Orchestrate the full standalone collection pipeline.

    Args:
        gesture_name: Pre-defined gesture name to use.
    """
    gesto_creado, next_index = pedir_nombre_gesto(gesture_name)
    if gesto_creado is None or next_index is None:
        return

    with build_landmarker() as landmarker:
        grabar_gesto(gesto_creado, next_index, landmarker)


if __name__ == "__main__":
    import sys
    arg_gesture = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_gesture)
