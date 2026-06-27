# --- Libraries ---
import math
from pathlib import Path
import time
from typing import Optional, TypedDict

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
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"

# Inference width (lower = less CPU lag). Landmarks are 0-1 and draw well on the full frame.
ANCHO_INFERENCIA = 320

ultimo_resultado = None
listo_para_inferir = True  # Avoids frame queuing: we only send when the callback has finished

# Variables to validate that the gesture is held per hand (Left/Right)
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
    """Async callback to receive landmarker results."""
    global ultimo_resultado, listo_para_inferir
    ultimo_resultado = result
    listo_para_inferir = True


def dibujar_manos(frame: np.ndarray, results: vision.HandLandmarkerResult) -> bool:
    """
    Draw hand landmarks and connections on the current frame.

    Args:
        frame: BGR canvas on which the markers will be drawn.
        results: Results delivered by the landmarker.

    Returns:
        bool: True if at least one hand was detected and drawn, False otherwise.
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
    Downscale the frame to speed up CPU inference.

    Args:
        frame_bgr: Original image at full resolution.

    Returns:
        np.ndarray: Resized image if it exceeds the configured width.
    """
    h, w = frame_bgr.shape[:2]
    if w <= ANCHO_INFERENCIA:
        return frame_bgr
    escala = ANCHO_INFERENCIA / w
    nuevo_h = int(h * escala)
    return cv2.resize(frame_bgr, (ANCHO_INFERENCIA, nuevo_h), interpolation=cv2.INTER_AREA)


def _distancia(lm1, lm2) -> float:
    """Normalized Euclidean distance between two landmarks (x, y)."""
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)


def get_vowel(hand_landmarks: list, hand_label: str) -> str | None:
    """
    Classify whether the hand gesture corresponds to a vowel.

    Conditions that depend on the X axis (horizontal thumb position)
    are inverted for the left hand, since the frame is mirrored and
    the thumb orientation is symmetric with respect to the right hand.

    Args:
        hand_landmarks: List of 21 landmarks for one hand.
        hand_label: 'Right' or 'Left' as returned by MediaPipe.

    Returns:
        str | None: Detected vowel ('A', 'E', 'I', 'O', 'U') or None.
    """
    lm = hand_landmarks  # Short alias for readability

    # 1. Finger state (Y axis — same for both hands)
    # A finger is closed if its TIP is below its PIP joint
    is_index_closed  = lm[8].y  > lm[6].y
    is_middle_closed = lm[12].y > lm[10].y
    is_ring_closed   = lm[16].y > lm[14].y
    is_pinky_closed  = lm[20].y > lm[18].y

    # Semi-closed: the tip passed the first knuckle but does not reach the palm
    is_ring_semi_closed  = lm[15].y > lm[14].y and lm[16].y > lm[15].y
    is_pinky_semi_closed = lm[19].y > lm[18].y and lm[20].y > lm[19].y

    # 2. Thumb state (Y axis — same for both hands)
    is_thumb_up = (lm[4].y < lm[3].y) and (lm[4].y < lm[6].y)

    # 3. Lateral thumb position (X axis — depends on the hand)
    # In the mirrored frame the left hand has its thumb at greater X (right side of screen)
    # and the right hand has it at smaller X (left side of screen).
    # is_thumb_down: thumb folded TOWARDS the other fingers (crossing the palm)
    # Since the camera feed is mirrored, the screen coordinates are inverted relative to actual hand labels:
    if hand_label == "Left":
        # Left hand (looks like Right hand on screen): thumb folds to the right (+X)
        is_thumb_down = lm[4].x > lm[3].x
    else:
        # Right hand (looks like Left hand on screen): thumb folds to the left (-X)
        is_thumb_down = lm[4].x < lm[3].x

    # is_thumb_on_side: thumb pointing outward (away from the pinky)
    # Works with absolute distances, so it is the same for both hands
    dist_pulgar_menique = abs(lm[4].x - lm[17].x)
    dist_indice_menique = abs(lm[5].x - lm[17].x)
    is_thumb_on_side = dist_pulgar_menique > dist_indice_menique

    # 4. Classification rules
    # 'O': thumb-index and thumb-middle fingertips touch forming a circle
    dist_thumb_index  = _distancia(lm[4], lm[8])
    dist_thumb_middle = _distancia(lm[4], lm[12])
    are_tips_touching = dist_thumb_index < 0.07 and dist_thumb_middle < 0.07
    if are_tips_touching and is_ring_semi_closed and is_pinky_semi_closed:
        return 'O'

    # 'A': all fingers closed, thumb lateral and pointing up
    if is_index_closed and is_middle_closed and is_ring_closed and is_pinky_closed and is_thumb_up and is_thumb_on_side:
        return 'A'
    # 'E': all fingers closed, thumb folded inward
    if is_index_closed and is_middle_closed and is_ring_closed and is_pinky_closed and is_thumb_down:
        return 'E'
    # 'I': only pinky open, thumb up
    if is_index_closed and is_middle_closed and is_ring_closed and not is_pinky_closed and is_thumb_up:
        return 'I'
    # 'U': index and middle open, ring and pinky closed, thumb inward
    if not is_index_closed and not is_middle_closed and is_ring_closed and is_pinky_closed and is_thumb_down:
        return 'U'
class HandState(TypedDict):
    """Type definition for tracking a single hand's vowel validation state."""
    vocal_detectada: Optional[str]
    tiempo_inicio: float
    confirmada: bool


class VowelValidator:
    """Tracks and validates vowel gesture hold time for Left and Right hands."""

    def __init__(self, confirmation_time: float = 1.0) -> None:
        """Initialize the hand vowel validator state.

        Args:
            confirmation_time: Time in seconds a gesture must be held.
        """
        self.confirmation_time = confirmation_time
        self.estado_manos: dict[str, HandState] = {
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

    def update(self, results: vision.HandLandmarkerResult) -> None:
        """Update the vowel classification validation timers.

        Args:
            results: The hand landmarker inference results.
        """
        manos_presentes = set()
        if results and results.hand_landmarks and results.handedness:
            for idx, landmarks in enumerate(results.hand_landmarks):
                hand_label = results.handedness[idx][0].category_name
                manos_presentes.add(hand_label)

                vocal = get_vowel(landmarks, hand_label)

                if vocal:
                    if vocal == self.estado_manos[hand_label]["vocal_detectada"]:
                        # Vowel is held, check the elapsed time
                        elapsed = time.time() - self.estado_manos[hand_label]["tiempo_inicio"]
                        if elapsed >= self.confirmation_time:
                            self.estado_manos[hand_label]["confirmada"] = True
                        else:
                            self.estado_manos[hand_label]["confirmada"] = False
                    else:
                        # Vowel changed or new, start counting
                        self.estado_manos[hand_label]["vocal_detectada"] = vocal
                        self.estado_manos[hand_label]["tiempo_inicio"] = time.time()
                        self.estado_manos[hand_label]["confirmada"] = False
                else:
                    self.estado_manos[hand_label]["vocal_detectada"] = None
                    self.estado_manos[hand_label]["confirmada"] = False

        # Reset the state of hands that did not appear in the frame
        for hand_label in ["Left", "Right"]:
            if hand_label not in manos_presentes:
                self.estado_manos[hand_label]["vocal_detectada"] = None
                self.estado_manos[hand_label]["confirmada"] = False

    def draw_status(self, frame: np.ndarray) -> None:
        """Draw the vowel verification status overlays onto the frame.

        Args:
            frame: BGR frame to draw status text onto.
        """
        y_offset = 80
        for hand_label in ["Left", "Right"]:
            vocal = self.estado_manos[hand_label]["vocal_detectada"]
            if vocal:
                lado = "Left" if hand_label == "Left" else "Right"
                if self.estado_manos[hand_label]["confirmada"]:
                    texto = f"Mano {lado} - Confirmada: {vocal}"
                    cv2.putText(frame, texto, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    elapsed = time.time() - self.estado_manos[hand_label]["tiempo_inicio"]
                    texto = f"Mano {lado} - Validando {vocal}... {elapsed:.1f}s"
                    cv2.putText(frame, texto, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 40


def main() -> None:
    """Run real-time threshold-based vowel recognition."""
    global listo_para_inferir
    listo_para_inferir = True

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera")
        return

    # Create the window with normal GUI to avoid the Qt toolbar
    cv2.namedWindow("Paso 03 - Tiempo real", cv2.WINDOW_GUI_NORMAL)

    # Fewer pixels from the camera = faster capture and conversion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=on_result,
    )

    validator = VowelValidator(confirmation_time=TIEMPO_CONFIRMACION)
    inicio = time.perf_counter()

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Real-time detection | Q = quit")
        print("(No frame queue: the most recent frame is processed when the model finishes)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read the frame")
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            if ultimo_resultado is not None:
                dibujar_manos(display, ultimo_resultado)
                validator.update(ultimo_resultado)
                validator.draw_status(display)

            cv2.putText(
                display, "Tiempo real | Q: salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Paso 03 - Tiempo real", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

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


if __name__ == "__main__":
    main()
