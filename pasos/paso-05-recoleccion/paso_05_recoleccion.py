import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Path resolution — mirrors the pattern used in paso_04_vocales.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"

# ---------------------------------------------------------------------------
# Recording parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30   # Number of frames per recorded sequence (temporal window)
NUM_FEATURES   = 63    # 21 hand landmarks * 3 coordinates (x, y, z)
NUM_SEQUENCES  = 30    # How many example sequences to record per gesture


# ---------------------------------------------------------------------------
# Step 2.1 — MediaPipe configuration (IMAGE mode for synchronous processing)
# ---------------------------------------------------------------------------
# We use RunningMode.IMAGE instead of LIVE_STREAM so every detect() call
# blocks until MediaPipe returns the result. This guarantees we capture
# exactly SEQUENCE_LENGTH frames without any being dropped asynchronously.
def build_landmarker() -> vision.HandLandmarker:
    """Create and return a HandLandmarker configured for synchronous IMAGE mode."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo no encontrado en: {MODEL_PATH}\n"
            "Asegúrate de haber descargado 'hand_landmarker.task' en la carpeta 'prueba/'."
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Synchronous — no callback needed
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Step 2.2 — Keypoint extraction helper
# ---------------------------------------------------------------------------
def extract_keypoints(results: vision.HandLandmarkerResult) -> np.ndarray:
    """
    Flatten the landmarks of the first detected hand into a 1-D array of 63 values.

    Returns:
        np.ndarray of shape (63,): [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        or np.zeros(63) when no hand is detected — keeping data shape consistent.
    """
    if results.hand_landmarks:
        # Only use the first hand detected (index 0)
        hand = results.hand_landmarks[0]
        # Build a flat list: x, y, z for each of the 21 landmarks
        keypoints = []
        for landmark in hand:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints, dtype=np.float32)   # shape: (63,)
    else:
        # No hand visible → return zeros so every frame stays shape (63,)
        return np.zeros(NUM_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 2.3 (helper) — Overlay text on frame for user guidance
# ---------------------------------------------------------------------------
def draw_hud(frame: np.ndarray, gesture: str, sequence: int, frame_num: int, waiting: bool) -> None:
    """Render recording state info directly onto the camera feed."""
    h = frame.shape[0]
    color_accent = (0, 255, 200)
    color_warn   = (0, 200, 255)
    color_label  = (255, 255, 255)

    # Gesture name at the top
    cv2.putText(frame, f"Gesto: {gesture.upper()}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_label, 2)

    # Sequence / total counter
    cv2.putText(frame, f"Secuencia {sequence + 1}/{NUM_SEQUENCES}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_accent, 2)

    if waiting:
        # Pause state: ask user to get ready
        cv2.putText(frame, "PREPARATE...", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_warn, 3)
    else:
        # Active recording state: show frame progress
        cv2.putText(frame, f"Grabando frame {frame_num + 1}/{SEQUENCE_LENGTH}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_accent, 2)


# ---------------------------------------------------------------------------
# Folder creation — Step 1 of the workflow
# ---------------------------------------------------------------------------
def pedir_nombre_gesto() -> str | None:
    """
    Ask the user for a gesture name, create the output folder, and return the
    normalized name. Returns None if folder creation fails.
    """
    raw_name = input("Nombre del gesto a grabar: ")
    normalized = raw_name.strip().lower()

    if not normalized:
        print("Error: El nombre no puede estar vacío.")
        return None

    gesture_folder = PROJECT_ROOT / "gestos" / normalized
    gesture_folder.mkdir(parents=True, exist_ok=True)

    if not gesture_folder.exists():
        print(f"Error: No se pudo crear la carpeta: {gesture_folder}")
        return None

    print(f"Carpeta lista en: {gesture_folder}")
    return normalized


# ---------------------------------------------------------------------------
# Step 2.3 + 2.4 — Main recording loop
# ---------------------------------------------------------------------------
def grabar_gesto(gesture_name: str, landmarker: vision.HandLandmarker) -> None:
    """
    Record NUM_SEQUENCES sequences of SEQUENCE_LENGTH frames each, extract
    keypoints per frame, and save every sequence as a .npy file.

    File naming convention: gestos/<gesture>/<sequence_index>.npy
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
    # Outer loop — one iteration = one full recorded sequence
    # -----------------------------------------------------------------------
    for sequence in range(NUM_SEQUENCES):
        sequence_data = []   # Temporary list — will hold SEQUENCE_LENGTH arrays of shape (63,)

        # -------------------------------------------------------------------
        # Inner loop — one iteration = one captured frame
        # -------------------------------------------------------------------
        for frame_num in range(SEQUENCE_LENGTH):

            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara.")
                break

            # Mirror the image so the user sees their hand correctly
            frame = cv2.flip(frame, 1)

            # UX: At the very first frame of each sequence, pause 2 seconds
            # so the user has time to reposition their hand.
            is_waiting = (frame_num == 0)
            if is_waiting:
                draw_hud(frame, gesture_name, sequence, frame_num, waiting=True)
                cv2.imshow("GestureFlow - Recolección de Datos", frame)
                # 2-second pause; still allow Q to quit
                if cv2.waitKey(2000) & 0xFF == ord("q"):
                    print("\nGrabación cancelada por el usuario.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                # Re-read so the displayed frame after the pause is fresh
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

            # ---------------------------------------------------------------
            # Convert BGR → RGB and wrap in mp.Image for synchronous detection
            # ---------------------------------------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Synchronous call — blocks until MediaPipe returns the result
            results = landmarker.detect(mp_image)

            # Extract the 63 keypoints (zeros if no hand found)
            keypoints = extract_keypoints(results)
            sequence_data.append(keypoints)

            # Display recording progress to the user
            draw_hud(frame, gesture_name, sequence, frame_num, waiting=False)
            cv2.imshow("GestureFlow - Recolección de Datos", frame)

            # 1 ms wait to process UI events; allow early quit with Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nGrabación cancelada por el usuario.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # -------------------------------------------------------------------
        # Step 2.4 — Save the completed sequence as a .npy file
        # -------------------------------------------------------------------
        # sequence_data is a list of SEQUENCE_LENGTH arrays, each of shape (63,)
        # np.array() converts it to shape (SEQUENCE_LENGTH, NUM_FEATURES) = (30, 63)
        npy_array = np.array(sequence_data, dtype=np.float32)   # shape: (30, 63)
        save_path = output_dir / f"{sequence}.npy"
        np.save(str(save_path), npy_array)
        print(f"  [✓] Secuencia {sequence + 1:02d}/{NUM_SEQUENCES} guardada → {save_path.name}  shape={npy_array.shape}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Recolección completa. {NUM_SEQUENCES} secuencias guardadas en: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Ask for gesture name and create output folder
    gesto_creado = pedir_nombre_gesto()
    if not gesto_creado:
        exit(1)

    # Step 2: Build the MediaPipe landmarker (IMAGE mode)
    with build_landmarker() as landmarker:
        # Step 3: Run the double recording loop
        grabar_gesto(gesto_creado, landmarker)
