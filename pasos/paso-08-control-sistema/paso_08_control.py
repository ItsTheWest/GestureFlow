"""Core logic for System Gesture Control (Step 8).

Implements gesture recognition for mouse movement (pointing), mouse clicks (pinching),
and cross-platform workspace switching (swiping).
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import math
import platform
import subprocess
import time
from typing import Any, Literal

import pynput
from pynput.keyboard import Key, Controller as KeyboardController

import config

class SystemMouseController(ABC):
    @abstractmethod
    def set_position(self, x: int, y: int) -> None: pass
    
    @abstractmethod
    def click(self) -> None: pass

class EvdevMouseController(SystemMouseController):
    def __init__(self, screen_w: int, screen_h: int):
        import evdev
        from evdev import UInput, ecodes as e, AbsInfo
        cap: dict[int, Any] = {
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(value=0, min=0, max=screen_w, fuzz=0, flat=0, resolution=0)),
                (e.ABS_Y, AbsInfo(value=0, min=0, max=screen_h, fuzz=0, flat=0, resolution=0))
            ],
            e.EV_KEY: [e.BTN_LEFT]
        }
        self.ui = UInput(cap, name="gestureflow-mouse", version=0x1)
        self.e = e
        
    def set_position(self, x: int, y: int) -> None:
        self.ui.write(self.e.EV_ABS, self.e.ABS_X, x)
        self.ui.write(self.e.EV_ABS, self.e.ABS_Y, y)
        self.ui.syn()
        
    def click(self) -> None:
        self.ui.write(self.e.EV_KEY, self.e.BTN_LEFT, 1)
        self.ui.syn()
        self.ui.write(self.e.EV_KEY, self.e.BTN_LEFT, 0)
        self.ui.syn()

class PynputMouseController(SystemMouseController):
    def __init__(self):
        from pynput.mouse import Button, Controller
        self.mouse = Controller()
        self.Button = Button
        
    def set_position(self, x: int, y: int) -> None:
        self.mouse.position = (x, y)
        
    def click(self) -> None:
        self.mouse.click(self.Button.left)

def build_mouse_controller(screen_w: int, screen_h: int) -> SystemMouseController:
    if platform.system() == "Linux":
        try:
            return EvdevMouseController(screen_w, screen_h)
        except Exception as e:
            print(f"Evdev failed: {e}. Falling back to pynput.")
    return PynputMouseController()


@dataclass
class ControlAction:
    """Represents a computed action to execute based on gesture state."""
    move: tuple[int, int] | None = None
    click: bool = False
    swipe: Literal["left", "right"] | None = None


class WorkspaceSwitcher(ABC):
    """Abstract base class for cross-platform workspace switching."""
    @abstractmethod
    def switch(self, direction: Literal["left", "right"]) -> None:
        pass


class HyprlandWorkspaceSwitcher(WorkspaceSwitcher):
    """Workspace switcher for Hyprland using hyprctl dispatch."""
    def switch(self, direction: Literal["left", "right"]) -> None:
        cmd_arg = "+1" if direction == "right" else "-1"
        try:
            subprocess.run(["hyprctl", "dispatch", "workspace", cmd_arg], check=False)
        except FileNotFoundError:
            pass # Ignore if hyprctl is missing


class KeyboardWorkspaceSwitcher(WorkspaceSwitcher):
    """Workspace switcher for macOS/Windows using keyboard hotkeys."""
    def __init__(self, left_keys: list[Key], right_keys: list[Key]):
        self.keyboard = KeyboardController()
        self.left_keys = left_keys
        self.right_keys = right_keys

    def switch(self, direction: Literal["left", "right"]) -> None:
        keys = self.right_keys if direction == "right" else self.left_keys
        for key in keys:
            self.keyboard.press(key)
        for key in reversed(keys):
            self.keyboard.release(key)


def build_workspace_switcher() -> WorkspaceSwitcher:
    """Factory method to instantiate the correct switcher for the OS."""
    system = platform.system()
    if system == "Linux":
        # Assumes Hyprland by default for this project
        return HyprlandWorkspaceSwitcher()
    elif system == "Windows":
        return KeyboardWorkspaceSwitcher(left_keys=[Key.ctrl, Key.cmd, Key.left],
                                         right_keys=[Key.ctrl, Key.cmd, Key.right])
    elif system == "Darwin":
        return KeyboardWorkspaceSwitcher(left_keys=[Key.ctrl, Key.left],
                                         right_keys=[Key.ctrl, Key.right])
    
    # Fallback to a dummy switcher if unsupported
    class DummySwitcher(WorkspaceSwitcher):
        def switch(self, direction: Literal["left", "right"]) -> None: pass
    return DummySwitcher()


class GestureController:
    """Manages the state and processing of landmarks to emit system control actions."""
    
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        self.mouse = build_mouse_controller(screen_w, screen_h)
        self.workspace_switcher = build_workspace_switcher()
        
        self.cursor_x: float | None = None
        self.cursor_y: float | None = None
        
        self.pinch_frames = 0
        self.is_pinched = False
        
        self.wrist_x_history: deque[float] = deque(maxlen=config.SWIPE_FRAMES)
        self.last_swipe_time: float = 0.0

    def process_landmarks(self, results: Any) -> ControlAction:
        """Processes mediapipe results to determine the control action."""
        action = ControlAction()
        
        if not results or not results.hand_landmarks:
            self.wrist_x_history.clear()
            return action

        # Use the first detected hand
        hand = results.hand_landmarks[0]
        
        # 1. State extraction
        is_index_closed  = hand[8].y  > hand[6].y
        is_middle_closed = hand[12].y > hand[10].y
        is_ring_closed   = hand[16].y > hand[14].y
        is_pinky_closed  = hand[20].y > hand[18].y
        
        # Pointing: only index is open (y is flipped in image coords)
        is_pointing = (not is_index_closed and 
                       is_middle_closed and 
                       is_ring_closed and 
                       is_pinky_closed)
        
        # Open hand: all fingers are open
        is_open_hand = (not is_index_closed and 
                        not is_middle_closed and 
                        not is_ring_closed and 
                        not is_pinky_closed)

        # Distance between thumb (4) and index (8) tips
        dist_pinch = math.sqrt((hand[4].x - hand[8].x)**2 + (hand[4].y - hand[8].y)**2)
        current_pinching = dist_pinch < config.PINCH_THRESHOLD
        
        # 2. Process Pointing & Mouse Movement
        if is_pointing:
            # Map index tip (8) to screen coordinates (1:1 full screen)
            target_x = hand[8].x * self.screen_w
            target_y = hand[8].y * self.screen_h
            
            # Apply EMA smoothing
            if self.cursor_x is None or self.cursor_y is None:
                self.cursor_x, self.cursor_y = target_x, target_y
            else:
                alpha = config.CURSOR_SMOOTHING
                self.cursor_x = alpha * target_x + (1 - alpha) * self.cursor_x
                self.cursor_y = alpha * target_y + (1 - alpha) * self.cursor_y
                
            action.move = (int(self.cursor_x), int(self.cursor_y))
            
            # 3. Process Pinch to Click (on release)
            if current_pinching:
                self.pinch_frames += 1
                if self.pinch_frames >= config.PINCH_MIN_FRAMES:
                    self.is_pinched = True
            else:
                if self.is_pinched:
                    # Released! Trigger click.
                    action.click = True
                    self.is_pinched = False
                self.pinch_frames = 0
        else:
            # Reset pinch state if not pointing
            self.pinch_frames = 0
            self.is_pinched = False
            self.cursor_x = None
            self.cursor_y = None

        # 4. Process Swipe (Open Hand)
        if is_open_hand:
            wrist_x = hand[0].x
            self.wrist_x_history.append(wrist_x)
            
            now = time.time()
            if len(self.wrist_x_history) == config.SWIPE_FRAMES and (now - self.last_swipe_time > config.SWIPE_COOLDOWN):
                start_x = self.wrist_x_history[0]
                end_x = self.wrist_x_history[-1]
                delta_x = end_x - start_x
                
                # Note: The camera frame is mirrored in main.py, so moving hand right 
                # (physically) moves right on screen (+delta_x).
                if delta_x > config.SWIPE_VELOCITY:
                    action.swipe = "right"
                    self.last_swipe_time = now
                    self.wrist_x_history.clear()
                elif delta_x < -config.SWIPE_VELOCITY:
                    action.swipe = "left"
                    self.last_swipe_time = now
                    self.wrist_x_history.clear()
        else:
            self.wrist_x_history.clear()

        return action

    def execute_action(self, action: ControlAction) -> None:
        """Executes the given control action using system APIs."""
        if action.move:
            self.mouse.set_position(action.move[0], action.move[1])
        
        if action.click:
            self.mouse.click()
            
        if action.swipe:
            self.workspace_switcher.switch(action.swipe)
