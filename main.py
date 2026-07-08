"""Unified desktop dashboard launcher for the GestureFlow project.

Provides a premium dark-themed interface built with CustomTkinter to run,
monitor, and manage Steps 4 through 7 of the gesture recognition pipeline.
The camera feed is embedded directly inside the GUI viewport, and its processing
mode changes dynamically depending on the selected step.
"""

from collections import deque
import importlib
import queue
import subprocess
import sys
import threading
import time
from typing import Any
from pathlib import Path


import cv2
import customtkinter as ctk
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageTk
from tkinter import messagebox
import numpy as np

import config
from utils import extract_keypoints, get_gesture_names


class StepCard(ctk.CTkFrame):
    """Reusable card-like container for step controls and description."""

    def __init__(
        self,
        master: Any,
        title: str,
        description: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Initialize the StepCard with standard styling.

        Args:
            master: Parent widget.
            title: Title text for the step.
            description: Description of the step function.
            *args: Variable positional arguments for CTkFrame.
            **kwargs: Variable keyword arguments for CTkFrame.
        """
        super().__init__(
            master,
            corner_radius=12,
            border_width=1,
            border_color="#3B3B3B",
            fg_color="#1E1E1E",
            *args,
            **kwargs
        )

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00FFCC"
        )
        self.title_label.pack(anchor="w", padx=16, pady=(12, 4))

        self.desc_label = ctk.CTkLabel(
            self,
            text=description,
            font=ctk.CTkFont(size=11),
            text_color="#AAAAAA",
            justify="left",
            wraplength=280
        )
        self.desc_label.pack(anchor="w", padx=16, pady=(0, 12))


class GestureFlowApp(ctk.CTk):
    """Main CustomTkinter application class orchestrating steps 4 to 7."""

    def __init__(self) -> None:
        """Initialize the dashboard layout, variables, and camera loop."""
        super().__init__()

        # Dynamic imports of step logic modules to prevent code duplication
        self.paso_04: Any = importlib.import_module("pasos.paso-04-reconocimiento-vocales.paso_04_vocales")
        self.paso_05: Any = importlib.import_module("pasos.paso-05-recoleccion.paso_05_recoleccion")
        self.paso_07: Any = importlib.import_module("pasos.paso-07-deteccion-tiempo-real.paso_07_deteccion")

        # Window settings
        self.title("GestureFlow — Control Panel")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        # Set dark theme styling
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Application state variables
        self.current_mode: str = "Idle"  # Modes: Idle, Vowels, Collection, Training, Inference
        self.active_processes: dict[str, subprocess.Popen] = {}
        self.log_queue: queue.Queue[str] = queue.Queue()

        # MediaPipe instance
        self.landmarker: mp_vision.HandLandmarker | None = None

        # Camera settings
        self.cap: cv2.VideoCapture | None = None
        self.camera_running: bool = False

        # Step 4 State Variables (Vowels)
        self.vowel_validator: Any = None
        self.last_vowel_detected: dict[str, str | None] = {"Left": None, "Right": None}
        self.last_vowel_confirmed: dict[str, bool] = {"Left": False, "Right": False}

        # Step 5 State Variables (Collection)
        self.col_running: bool = False
        self.col_gesture_name: str = ""
        self.col_saved_sequences: int = 0
        self.col_manager: Any = None
        self.col_space_pressed: bool = False

        # Step 7 State Variables (Inference)
        self.lstm_model: Any = None
        self.model_loading: bool = False
        self.inf_gestures: list[str] = []
        self.inf_buffer: deque[np.ndarray] = deque(maxlen=config.SEQUENCE_LENGTH)
        self.inf_prediction_in_progress: bool = False
        self.inf_current_gesture: str = ""
        self.inf_current_confidence: float = 0.0
        self.inf_last_print_time: float = 0.0
        self.prediction_lock: threading.Lock = threading.Lock()

        # MediaPipe Async State Variables
        self.latest_results: Any = None
        self.results_lock: threading.Lock = threading.Lock()
        self.mp_processing: bool = False
        self.results_updated: bool = False

        # Bind space key for collection flow control
        self.bind("<space>", self.on_space_pressed)

        # Layout grids
        self.grid_columnconfigure(0, weight=1, minsize=350)  # Left controls
        self.grid_columnconfigure(1, weight=2, minsize=650)  # Right camera viewport
        self.grid_rowconfigure(0, weight=1)

        # ── LEFT PANEL: Controls ────────────────────────────────────────────────
        self.left_panel = ctk.CTkFrame(self, fg_color="#121212", corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Header Title
        self.header = ctk.CTkLabel(
            self.left_panel,
            text="GESTUREFLOW",
            font=ctk.CTkFont(size=24, weight="bold", family="Helvetica"),
            text_color="#FFFFFF"
        )
        self.header.pack(anchor="w", padx=20, pady=(20, 4))

        self.subtitle = ctk.CTkLabel(
            self.left_panel,
            text="Unified Control Panel",
            font=ctk.CTkFont(size=12),
            text_color="#00FFCC"
        )
        self.subtitle.pack(anchor="w", padx=20, pady=(0, 20))

        # Mode Selection Segmented Button
        self.lbl_modes = ctk.CTkLabel(
            self.left_panel,
            text="Select Pipeline Step Mode:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#DDDDDD"
        )
        self.lbl_modes.pack(anchor="w", padx=20, pady=(0, 6))

        self.mode_selector = ctk.CTkSegmentedButton(
            self.left_panel,
            values=["Idle", "Vowels", "Collection", "Training", "Inference"],
            command=self.change_mode,
            selected_color="#1F538D"
        )
        self.mode_selector.set("Idle")
        self.mode_selector.pack(fill="x", padx=20, pady=(0, 20))

        # Reusable Dynamic Containers Area
        self.container_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.container_frame.pack(fill="both", expand=True, padx=20)

        # Step 4 Details Card
        self.card_step4 = StepCard(
            self.container_frame,
            title="Step 4: Vowel Recognition",
            description="Run static threshold-based vowel recognition using mathematical rules without neural networks."
        )
        self.card_step4.pack_forget()

        # Step 5 Details Card
        self.card_step5 = StepCard(
            self.container_frame,
            title="Step 5: Dataset Collection",
            description="Enter a gesture name below, then use the live viewport to record hand landmark sequences."
        )
        self.entry_frame = ctk.CTkFrame(self.card_step5, fg_color="transparent")
        self.entry_frame.pack(fill="x", padx=16, pady=(0, 12))
        self.lbl_gesture = ctk.CTkLabel(
            self.entry_frame,
            text="Class Name:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#DDDDDD"
        )
        self.lbl_gesture.pack(side="left", padx=(0, 8))
        self.entry_gesture = ctk.CTkEntry(
            self.entry_frame,
            placeholder_text="e.g. ok, click, saludo",
            fg_color="#2B2B2B",
            border_color="#3B3B3B",
            height=28
        )
        self.entry_gesture.pack(side="left", fill="x", expand=True)

        self.btn_collect = ctk.CTkButton(
            self.card_step5,
            text="Start Data Collection",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#2B8E5C",
            hover_color="#1D603E",
            command=self.start_step5_action
        )
        self.btn_collect.pack(fill="x", padx=16, pady=(0, 12))
        self.card_step5.pack_forget()

        # Step 6 Details Card
        self.card_step6 = StepCard(
            self.container_frame,
            title="Step 6: LSTM Model Training",
            description="Train the LSTM network on all folders in your local gesture database. Outputs exported automatically."
        )
        self.btn_train = ctk.CTkButton(
            self.card_step6,
            text="Train Model (LSTM)",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#8E2B8D",
            hover_color="#631D62",
            command=self.start_step6_action
        )
        self.btn_train.pack(fill="x", padx=16, pady=(0, 12))
        self.card_step6.pack_forget()

        # Step 7 Details Card
        self.card_step7 = StepCard(
            self.container_frame,
            title="Step 7: LSTM Inference Detections",
            description="Run real-time neural network predictions inside the viewport to distinguish complex gestures."
        )
        self.card_step7.pack_forget()

        # Output Log Box
        self.console_frame = ctk.CTkFrame(
            self.left_panel,
            corner_radius=12,
            border_width=1,
            border_color="#3B3B3B",
            fg_color="#151515"
        )
        self.console_frame.pack(fill="both", expand=True, padx=0, pady=(20, 20))
        self.console_title = ctk.CTkLabel(
            self.console_frame,
            text="Dashboard Action Logs",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#888888"
        )
        self.console_title.pack(anchor="w", padx=16, pady=(8, 4))
        self.textbox_logs = ctk.CTkTextbox(
            self.console_frame,
            fg_color="#0A0A0A",
            text_color="#A2F5A2",
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.textbox_logs.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Exit Button
        self.btn_exit = ctk.CTkButton(
            self.left_panel,
            text="Salir",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#903030",
            hover_color="#602020",
            command=self.destroy
        )
        self.btn_exit.pack(fill="x", padx=20, pady=(0, 20), side="bottom")

        # Show initial Idle layout
        self.change_mode("Idle")

        # ── RIGHT PANEL: Camera Viewport ────────────────────────────────────────
        self.right_panel = ctk.CTkFrame(self, fg_color="#1A1A1A", corner_radius=0)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Camera viewport frame
        self.viewport_container = ctk.CTkFrame(
            self.right_panel,
            fg_color="#0E0E0E",
            corner_radius=12,
            border_width=1,
            border_color="#3B3B3B"
        )
        self.viewport_container.grid(row=0, column=0, padx=24, pady=24, sticky="nsew")
        self.viewport_container.grid_rowconfigure(0, weight=1)
        self.viewport_container.grid_columnconfigure(0, weight=1)

        self.camera_viewport = ctk.CTkLabel(
            self.viewport_container,
            text="Loading Camera Feed...",
            font=ctk.CTkFont(size=16),
            text_color="#888888"
        )
        self.camera_viewport.grid(row=0, column=0, sticky="nsew")

        # Start camera capturing and Tkinter after loop
        self.start_camera()

        # Start log queue check
        self.after(100, self.check_log_queue)

    def write_log(self, text: str) -> None:
        """Helper to print a log line to the internal textbox.

        Args:
            text: Log content string.
        """
        self.textbox_logs.insert("end", text + "\n")
        self.textbox_logs.see("end")

    def check_log_queue(self) -> None:
        """Drain the log queue and print all logs to the GUI console."""
        while not self.log_queue.empty():
            try:
                line = self.log_queue.get_nowait()
                self.write_log(line)
            except queue.Empty:
                break
        self.after(100, self.check_log_queue)

    def start_camera(self) -> None:
        """Initialize the OpenCV capture stream."""
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_running = True
            self.update_camera()

    def stop_camera(self) -> None:
        """Release the camera resources."""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def draw_standby_frame(self) -> None:
        """Draw a standby dark screen on the camera label viewport."""
        standby_img = Image.new("RGB", (640, 480), color=(15, 15, 15))
        img_tk = ImageTk.PhotoImage(image=standby_img)
        self.camera_viewport.configure(image=img_tk)
        self.camera_viewport.image = img_tk

    def update_camera(self) -> None:
        """Read and process frames from the camera in a loop."""
        if not self.camera_running:
            return

        if not self.cap or not self.cap.isOpened():
            self.draw_standby_frame()
            self.after(33, self.update_camera)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.draw_standby_frame()
            self.after(33, self.update_camera)
            return

        # Mirror camera frame
        frame = cv2.flip(frame, 1)

        # Trigger background hand landmarker detection if loaded and not busy
        if self.landmarker is not None and not self.mp_processing:
            self.mp_processing = True
            frame_copy = frame.copy()
            timestamp_ms = int(time.time() * 1000)

            def detect_bg() -> None:
                try:
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    res = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                    with self.results_lock:
                        self.latest_results = res
                        self.results_updated = True
                except Exception as e:
                    print(f"Error in background landmarker: {e}")
                finally:
                    self.mp_processing = False

            threading.Thread(target=detect_bg, daemon=True).start()

        # Render overlays using the latest results in the main thread (non-blocking!)
        with self.results_lock:
            current_results = self.latest_results
            new_results_available = self.results_updated
            self.results_updated = False

        frame = self.process_viewport_frame(frame, current_results, new_results_available)

        # Convert image formats for CustomTkinter label display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Fit image to the viewport frame width/height, maintaining aspect ratio
        viewport_w = self.viewport_container.winfo_width()
        viewport_h = self.viewport_container.winfo_height()

        if viewport_w < 100 or viewport_h < 100:
            viewport_w, viewport_h = 640, 480

        scale = min(viewport_w / 640, viewport_h / 480)
        new_w = int(640 * scale)
        new_h = int(480 * scale)

        if new_w > 0 and new_h > 0:
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=pil_image)
        self.camera_viewport.configure(image=img_tk, text="")
        self.camera_viewport.image = img_tk

        self.after(33, self.update_camera)

    def on_space_pressed(self, event: Any) -> None:
        """Spacebar callback to drive step 5 data collection phases.

        Args:
            event: Event object containing key details.
        """
        if self.current_mode == "Collection" and self.col_running:
            self.col_space_pressed = True

    def run_subprocess(
        self,
        task_name: str,
        args: list[str],
        button_to_disable: ctk.CTkButton
    ) -> None:
        """Run a script as an asynchronous subprocess.

        Args:
            task_name: Display name of the running step.
            args: Complete CLI execution argument list.
            button_to_disable: Reference to the trigger button to disable/enable.
        """
        if task_name in self.active_processes:
            self.write_log(f"[!] Warning: {task_name} is already running.")
            return

        self.write_log(f"[*] Starting {task_name}...")
        button_to_disable.configure(state="disabled")
        self.mode_selector.configure(state="disabled")

        def target_run() -> None:
            try:
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                self.active_processes[task_name] = process

                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        self.log_queue.put(line.strip())

                process.wait()
                self.log_queue.put(f"[+] {task_name} finished (exit code {process.returncode}).")
            except Exception as e:
                self.log_queue.put(f"[-] Error executing {task_name}: {e}")
            finally:
                self.active_processes.pop(task_name, None)
                self.after(10, lambda: button_to_disable.configure(state="normal"))
                self.after(10, lambda: self.mode_selector.configure(state="normal"))

        threading.Thread(target=target_run, daemon=True).start()

    def update_prediction_result(self, gesture_index: int, confidence: float) -> None:
        """Update inference results in the GUI state (main thread).

        Args:
            gesture_index: Argmax index of prediction.
            confidence: Confidence score of prediction.
        """
        with self.prediction_lock:
            self.inf_prediction_in_progress = False

        now = time.time()
        if now - self.inf_last_print_time >= 3.0:
            self.write_log(f"[+] Pred: {self.inf_gestures[gesture_index]} ({confidence:.4f})")
            self.inf_last_print_time = now

        if confidence > config.CONFIDENCE_THRESHOLD:
            self.inf_current_gesture = self.inf_gestures[gesture_index]
        else:
            self.inf_current_gesture = ""
        self.inf_current_confidence = confidence

    def on_prediction_error(self) -> None:
        """Reset prediction lock state on background thread error."""
        with self.prediction_lock:
            self.inf_prediction_in_progress = False

    def process_viewport_frame(self, frame: np.ndarray, results: Any, new_results: bool) -> np.ndarray:
        """Process the BGR camera frame based on the active dashboard mode.

        Args:
            frame: Raw camera frame in BGR format.
            results: MediaPipe hand landmarker results (computed asynchronously).
            new_results: True if the results parameter contains updated frame keypoints.

        Returns:
            np.ndarray: BGR frame with overlay renderings applied.
        """
        if self.current_mode == "Vowels" and self.landmarker and self.vowel_validator:
            # Process static vowel recognition using reusable class
            if results:
                self.paso_04.dibujar_manos(frame, results)
                self.vowel_validator.update(results)

                # Log state changes
                for side in ["Left", "Right"]:
                    curr_v = self.vowel_validator.estado_manos[side]["vocal_detectada"]
                    curr_c = self.vowel_validator.estado_manos[side]["confirmada"]

                    prev_v = self.last_vowel_detected[side]
                    prev_c = self.last_vowel_confirmed[side]

                    if curr_v != prev_v:
                        if curr_v is not None:
                            self.write_log(f"[*] Vowels: {side} hand started validating '{curr_v}'...")
                        elif prev_v is not None:
                            self.write_log(f"[-] Vowels: {side} hand lost gesture.")
                        self.last_vowel_detected[side] = curr_v

                    if curr_c != prev_c:
                        if curr_c:
                            self.write_log(f"[+] Vowels: {side} hand CONFIRMED '{curr_v}'!")
                        self.last_vowel_confirmed[side] = curr_c
            self.vowel_validator.draw_status(frame)

        elif self.current_mode == "Collection" and self.landmarker and self.col_running and self.col_manager:
            # Process automatic dataset collection using reusable class
            space_pressed = self.col_space_pressed
            self.col_space_pressed = False

            prev_state = self.col_manager.state
            prev_saved = self.col_manager.sequences_saved

            frame = self.col_manager.process_frame(frame, results, int(time.time() * 1000), space_pressed, new_results)

            # Sync progress index back to local app parameters
            self.col_saved_sequences = self.col_manager.sequences_saved

            if self.col_manager.state != prev_state:
                self.write_log(f"[*] Collection: State changed from '{prev_state}' to '{self.col_manager.state}'")
                if self.col_manager.state == "Paused":
                    try:
                        next_name, _, next_desc = self.paso_05.PHASES[self.col_manager.current_phase_idx]
                        self.write_log(f"[i] Next Phase: {next_name} - {next_desc}")
                    except Exception:
                        pass

            if self.col_manager.sequences_saved != prev_saved:
                self.write_log(f"[+] Collection: Saved sequence {self.col_manager.sequences_saved}/{self.paso_05.NUM_SEQUENCES}")

            if not self.col_manager.is_active:
                self.stop_collection_successfully()

        elif self.current_mode == "Inference" and self.landmarker:
            # Process real-time model predictions using reusable functions
            if self.model_loading:
                cv2.putText(frame, "Loading Keras Model...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            elif self.lstm_model is None:
                cv2.putText(frame, "No Model Loaded.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                if results:
                    self.paso_07.dibujar_landmarks(frame, results)

                    if new_results:
                        hand_detected = bool(results.hand_landmarks)
                        if not hand_detected:
                            with self.prediction_lock:
                                self.inf_current_gesture = ""
                            self.inf_buffer.clear()
                        else:
                            keypoints = extract_keypoints(results)
                            self.inf_buffer.append(keypoints)

                            with self.prediction_lock:
                                can_predict = len(self.inf_buffer) == config.SEQUENCE_LENGTH and not self.inf_prediction_in_progress
                                if can_predict:
                                    self.inf_prediction_in_progress = True

                            if can_predict:
                                sequence_snapshot = np.array(self.inf_buffer, dtype=np.float32)
                                threading.Thread(
                                    target=self.paso_07.predecir_gesto_async,
                                    args=(
                                        self.lstm_model,
                                        sequence_snapshot,
                                        self.inf_gestures,
                                        self.update_prediction_result,
                                        self.on_prediction_error
                                    ),
                                    daemon=True
                                ).start()
                else:
                    cv2.putText(frame, "No hand detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                # Display predictions overlay
                with self.prediction_lock:
                    gesture = self.inf_current_gesture
                    confidence = self.inf_current_confidence

                if gesture:
                    cv2.putText(frame, f"{gesture} ({confidence:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"Detecting... ({confidence:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        elif self.current_mode == "Training":
            cv2.putText(frame, "Training model in background...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        return frame

    def change_mode(self, mode: str) -> None:
        """Switch the current active mode and update the controls layout.

        Args:
            mode: Target mode string identifier.
        """
        self.current_mode = mode
        self.write_log(f"[*] Switched mode to: {mode}")

        # Hide all step cards first
        self.card_step4.pack_forget()
        self.card_step5.pack_forget()
        self.card_step6.pack_forget()
        self.card_step7.pack_forget()

        # Show selected mode details card
        if mode == "Vowels":
            self.card_step4.pack(fill="x", pady=10)
            self.vowel_validator = self.paso_04.VowelValidator(confirmation_time=1.0)
            self.last_vowel_detected = {"Left": None, "Right": None}
            self.last_vowel_confirmed = {"Left": False, "Right": False}
        elif mode == "Collection":
            self.card_step5.pack(fill="x", pady=10)
        elif mode == "Training":
            self.card_step6.pack(fill="x", pady=10)
        elif mode == "Inference":
            self.card_step7.pack(fill="x", pady=10)

        # Allocate/release MediaPipe HandLandmarker based on active mode
        if mode in ["Vowels", "Collection", "Inference"]:
            if self.landmarker is None:
                if not config.MP_TASK_PATH.exists():
                    self.write_log(f"[-] Error: HandLandmarker task missing at {config.MP_TASK_PATH}")
                    messagebox.showerror("Error", f"MediaPipe model task not found at {config.MP_TASK_PATH}")
                    self.mode_selector.set("Idle")
                    self.change_mode("Idle")
                    return

                try:
                    self.write_log("[*] Initializing MediaPipe HandLandmarker...")
                    self.landmarker = self.paso_05.build_landmarker()
                    self.write_log("[+] MediaPipe HandLandmarker loaded.")
                except Exception as e:
                    self.write_log(f"[-] Error initializing MediaPipe: {e}")
                    messagebox.showerror("Error", f"Failed to initialize MediaPipe: {e}")
        else:
            if self.landmarker is not None:
                self.landmarker.close()
                self.landmarker = None
                self.write_log("[*] MediaPipe HandLandmarker closed.")

        # Special setup for step 7 Inference mode
        if mode == "Inference":
            if not config.MODEL_PATH.exists():
                self.write_log(f"[-] Error: LSTM Keras model missing at {config.MODEL_PATH}")
                messagebox.showerror("Missing Model", f"Trained LSTM model not found at {config.MODEL_PATH}.\nRun Step 6 first.")
                self.mode_selector.set("Idle")
                self.change_mode("Idle")
                return

            try:
                self.inf_gestures = get_gesture_names(config.GESTOS_DIR)
            except Exception as e:
                self.write_log(f"[-] Error listing gestures: {e}")
                messagebox.showerror("Error", f"Failed to retrieve gesture classes: {e}")
                self.mode_selector.set("Idle")
                self.change_mode("Idle")
                return

            if self.lstm_model is None and not self.model_loading:
                self.model_loading = True

                def load_model_thread() -> None:
                    try:
                        self.write_log("[*] Loading LSTM Keras model (may take a few seconds)...")
                        self.lstm_model = self.paso_07.cargar_modelo(config.MODEL_PATH)
                        self.write_log(f"[+] Model loaded from {config.MODEL_PATH}")
                        self.model_loading = False
                    except Exception as err:
                        self.write_log(f"[-] Error loading model: {err}")
                        self.model_loading = False
                        self.after(0, lambda: messagebox.showerror("Error", f"Failed to load Keras model: {err}"))
                        self.after(0, lambda: self.mode_selector.set("Idle"))
                        self.after(0, lambda: self.change_mode("Idle"))

                threading.Thread(target=load_model_thread, daemon=True).start()
        else:
            self.inf_buffer.clear()
            with self.prediction_lock:
                self.inf_current_gesture = ""

    def start_step5_action(self) -> None:
        """Trigger start/stop of dataset recording (Step 5)."""
        if self.col_running:
            # Stop collection
            self.col_running = False
            self.col_manager = None
            self.btn_collect.configure(text="Start Data Collection", fg_color="#2B8E5C", hover_color="#1D603E")
            self.mode_selector.configure(state="normal")
            self.write_log("[*] Data collection interrupted by user.")
        else:
            # Start collection
            gesture_name = self.entry_gesture.get().strip().lower()
            if not gesture_name:
                messagebox.showwarning("Input Required", "Please enter a class name to record.")
                return

            self.col_gesture_name = gesture_name
            self.write_log(f"[*] Verifying dataset directory index for '{gesture_name}'...")

            # Count existing files to determine starting index
            output_dir = config.PROJECT_ROOT / "gestos" / gesture_name
            output_dir.mkdir(parents=True, exist_ok=True)
            existing_files = list(output_dir.glob("*.npy"))

            if existing_files:
                indices = []
                for f in existing_files:
                    try:
                        indices.append(int(f.stem))
                    except ValueError:
                        pass
                self.col_saved_sequences = max(indices) + 1 if indices else 0
            else:
                self.col_saved_sequences = 0

            if self.col_saved_sequences >= 200:
                self.write_log(f"[!] Warning: Already collected {self.col_saved_sequences}/200 sequences.")
                if messagebox.askyesno("Limit Reached", f"Already have {self.col_saved_sequences} sequences. Overwrite index 0?"):
                    self.col_saved_sequences = 0
                else:
                    return

            self.col_manager = self.paso_05.CollectionManager(self.col_gesture_name, self.col_saved_sequences)
            self.col_running = True
            self.col_space_pressed = False
            self.btn_collect.configure(text="Stop Collection", fg_color="#903030", hover_color="#602020")
            self.mode_selector.configure(state="disabled")
            self.write_log(f"[+] Collection started at sequence index {self.col_saved_sequences}.")
            self.write_log("[i] Press SPACEBAR inside the application to begin recording.")

    def stop_collection_successfully(self) -> None:
        """Callback to finalize collection settings when success threshold met."""
        self.col_running = False
        self.col_manager = None
        self.btn_collect.configure(text="Start Data Collection", fg_color="#2B8E5C", hover_color="#1D603E")
        self.mode_selector.configure(state="normal")
        self.write_log("[+] Data collection completed successfully.")
        messagebox.showinfo("Success", f"Recorded 200 sequences of '{self.col_gesture_name}' successfully!")

    def start_step6_action(self) -> None:
        """Launch the step 6 model training script (Step 6)."""
        script_path = Path("pasos") / "paso-06-entrenamiento" / "paso_06_entrenamiento.py"
        if not script_path.exists():
            messagebox.showerror("Error", f"Training script not found at {script_path}")
            return

        self.run_subprocess(
            "Step 6: LSTM Training",
            [sys.executable, str(script_path)],
            self.btn_train
        )


def main() -> None:
    """Instantiate and run the CustomTkinter dashboard application loop."""
    app = GestureFlowApp()
    try:
        app.mainloop()
    finally:
        app.stop_camera()


if __name__ == "__main__":
    main()
