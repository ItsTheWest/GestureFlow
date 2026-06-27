"""Unified desktop dashboard launcher for the GestureFlow project.

Provides a premium dark-themed interface built with CustomTkinter to run,
monitor, and manage Steps 4 through 7 of the gesture recognition pipeline.
The camera feed is embedded directly inside the GUI viewport, and its processing
mode changes dynamically depending on the selected step.
"""

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
