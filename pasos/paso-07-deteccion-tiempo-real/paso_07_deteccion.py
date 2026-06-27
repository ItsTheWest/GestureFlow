# ── Standard Library ──────────────────────────────────────────────────────────
from collections import deque
from pathlib import Path
import threading
import time
from typing import Callable

# ── Third-Party ────────────────────────────────────────────────────────────────
import cv2
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np

# ── Local ──────────────────────────────────────────────────────────────────────
import config
from utils import HAND_CONNECTIONS, extract_keypoints, get_gesture_names

PROJECT_ROOT: Path = config.PROJECT_ROOT
MP_TASK_PATH: Path = config.MP_TASK_PATH
MODEL_PATH: Path = config.MODEL_PATH
GESTOS_DIR: Path = config.GESTOS_DIR

SEQUENCE_LENGTH: int = config.SEQUENCE_LENGTH
CONFIDENCE_THRESHOLD: float = config.CONFIDENCE_THRESHOLD

def cargar_modelo(model_path: Path):
    try:
        model = load_model(model_path)
        print(f"Modelo cargado exitosamente desde {model_path}")
        return model
    except Exception as e:
        raise FileNotFoundError(f"Model not found at {model_path}: {e}")

def setup_landmarker() -> vision.HandLandmarker:
    """Sets up the MediaPipe inference pipeline to detect hand keypoints."""
    base_options = BaseOptions(model_asset_path=str(MP_TASK_PATH))

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # Changed to VIDEO for real-time streams
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def dibujar_landmarks(frame: np.ndarray, results: vision.HandLandmarkerResult) -> None:
    if not results.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand_landmarks_list in results.hand_landmarks:
       # Store landmark coordinates in a list
        coords = [
            (int(lm.x * w), int(lm.y * h)) 
            for lm in hand_landmarks_list
        ]
        # Draw connections using the pre-calculated coordinates
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 0), 2)
            
        # Draw the keypoints
        for coord in coords:
            cv2.circle(frame, coord, 3, (0, 0, 255), -1)


def predecir_gesto_async(
    model,
    input_sequence: np.ndarray,
    gestures: list[str],
    callback: Callable[[int, float], None],
    error_callback: Callable[[], None]
) -> None:
    """Runs the LSTM model prediction in a secondary thread asynchronously."""
    try:
        input_data = np.expand_dims(input_sequence, axis=0)
        prediction = model(input_data, training=False).numpy()[0]
        gesture_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        callback(gesture_index, confidence)
    except Exception as e:
        print(f"Error in prediction thread: {e}")
        error_callback()


def main() -> None:
    # STEP 1: Load the model
    model = cargar_modelo(MODEL_PATH)
    model.summary()  # display model summary
    
    # STEP 2: Load the gestures
    gestures = get_gesture_names(GESTOS_DIR)
    print(f"Loaded {len(gestures)} gestures: {gestures}") # display loaded gestures
    
    sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    
    # STEP 3: Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access the video camera.")
    
    # STEP 4: Initialize the keypoint detector
    with setup_landmarker() as landmarker:
        print("System ready. Starting detection loop...")
        
        # Save initial time to calculate incremental timestamp_ms
        start_time: float = time.time()
        frame_count: int = 0
        current_gesture: str = ""
        current_confidence: float = 0.0
        prediction_in_progress: bool = False
        last_print_time: float = 0.0
        prediction_lock = threading.Lock()

        def on_prediction_complete(gesture_index: int, confidence: float) -> None:
            nonlocal current_gesture, current_confidence, prediction_in_progress, last_print_time
            now = time.time()
            if now - last_print_time >= 3.0:
                print(f"Pred: {gestures[gesture_index]} ({confidence:.4f})")
                last_print_time = now
            if confidence > CONFIDENCE_THRESHOLD:
                current_gesture = gestures[gesture_index]
            else:
                current_gesture = ""
            current_confidence = confidence
            with prediction_lock:
                prediction_in_progress = False

        def on_prediction_error() -> None:
            nonlocal prediction_in_progress
            with prediction_lock:
                prediction_in_progress = False
        
        # --- Main capture loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Could not read the camera frame.")
                break
                
            # 7: Flip horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # 8: Convert BGR -> RGB and wrap in mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 9: Calculate timestamp_ms and run detection in VIDEO mode
            timestamp_ms = int((time.time() - start_time) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # 10: Draw hand skeleton for visualisation
            dibujar_landmarks(frame, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            frame_count += 1
            
            # 12: Run prediction asynchronously when the sequence is complete
            with prediction_lock:
                can_predict = len(sequence) == SEQUENCE_LENGTH and not prediction_in_progress
                if can_predict:
                    prediction_in_progress = True
            
            if can_predict:
                # Copy sequence to avoid race conditions across concurrent threads
                sequence_snapshot = np.array(sequence, dtype=np.float32)
                threading.Thread(
                    target=predecir_gesto_async,
                    args=(model, sequence_snapshot, gestures, on_prediction_complete, on_prediction_error)
                ).start()
            
            # Show the last detected gesture or a warning if no hand is visible
            if not results.hand_landmarks:
                cv2.putText(frame, "No hand detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                current_gesture = ""
                sequence.clear()
            elif current_gesture and current_confidence > CONFIDENCE_THRESHOLD:
                cv2.putText(frame, f"{current_gesture} ({current_confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 13: Show frame and wait for ESC key (code 27)
            cv2.imshow("GestureFlow Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    # 14: Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
