"""Unified desktop dashboard launcher for the GestureFlow project.

Provides a premium dark-themed interface built with CustomTkinter to run,
monitor, and manage Steps 4 through 7 of the gesture recognition pipeline.
The camera feed is embedded directly inside the GUI viewport, and its processing
mode changes dynamically depending on the selected step.
"""

import queue
import subprocess
import sys
import threading
from typing import Any

import customtkinter as ctk


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
        """Initialize the dashboard layout, variables, and window settings."""
        super().__init__()

        # Window settings
        self.title("GestureFlow — Control Panel")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        # Set dark theme styling
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Application state variables
        self.current_mode: str = "Idle"
        self.active_processes: dict[str, subprocess.Popen] = {}
        self.log_queue: queue.Queue[str] = queue.Queue()

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

        # Show initial Idle layout
        self.change_mode("Idle")

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

    def change_mode(self, mode: str) -> None:
        """Switch the current active mode.

        Args:
            mode: Target mode string identifier.
        """
        pass

    def start_step5_action(self) -> None:
        """Trigger start/stop of dataset recording."""
        pass

    def start_step6_action(self) -> None:
        """Launch the step 6 model training script."""
        pass
