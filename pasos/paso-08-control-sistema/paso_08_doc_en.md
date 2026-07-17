# Documentation: Step 08 — System Control (`paso_08_control.py`)

Pipeline for **operating system control using hand gestures**: translates landmark features and rule-based pose evaluations into actual OS actions — mouse cursor movement, left click, and virtual workspace switching — across Linux, Windows, and macOS.

To read about keypoint extraction and the foundational MediaPipe HandLandmarker stream feeding this step, refer to the common glossary in [REFERENCIA_COMUN.md](../../pasos/REFERENCIA_COMUN.md).

---

## Index

- [1. Step Objective](#1-step-objective)
- [2. Folder Files](#2-folder-files)
- [3. Pipeline Flow](#3-pipeline-flow)
- [4. Hand Pose Detection Rules](#4-hand-pose-detection-rules)
- [5. Cursor Movement (Two-Finger Pointing)](#5-cursor-movement-two-finger-pointing)
- [6. Pinch to Click](#6-pinch-to-click)
- [7. Workspace Switch (Open-Hand Swipe)](#7-workspace-switch-open-hand-swipe)
- [8. Cross-platform Strategy Architecture](#8-cross-platform-strategy-architecture)
- [9. Code Classes and Functions](#9-code-classes-and-functions)
- [10. Config Constants](#10-config-constants)
- [11. How to Run](#11-how-to-run)
- [12. Troubleshooting Common Errors](#12-troubleshooting-common-errors)

---

## 1. Step Objective

**Objective:** Map hand coordinates and shapes to OS-level inputs (mouse movements, clicks, desktop swipes).

| Included in this module | Not included |
|-------------------------|-------------|
| EMA-smoothed relative mouse movement | Training the LSTM model (Step 6) |
| Pinch gesture detection and left-click | Inference of custom dynamic gestures (Step 7) |
| Virtual desktop / workspace switching | GUI layout definitions (main.py) |
| Strategy pattern for cross-platform inputs | Scrolling or secondary clicks |
| Geometric rule-based pose estimation | — |

**Success Criteria:**
- Placing two fingers open moves mouse cursor.
- Pinching index and thumb registers left click.
- Swiping open hand left/right switches desktops.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_08_control.py](../../pasos/paso-08-control-sistema/paso_08_control.py) | Step script |
| [paso_08_doc.md](../../pasos/paso-08-control-sistema/paso_08_doc.md) | Spanish documentation |
| [paso_08_doc_en.md](../../pasos/paso-08-control-sistema/paso_08_doc_en.md) | This English documentation |

---

## 3. Pipeline Flow
- Evaluates landmarks array against finger open/closed calculations.
- If pointing, tracks index-middle midpoint and increments cursor position relatively using an EMA filter (`alpha = 0.4`).
- If pinch distance < 0.06, triggers mouse left click event.
- If open hand velocity exceeds threshold, switches workspaces.

---

## 4. Hand Pose Detection Rules
Finger open/closed states are calculated by comparing y coordinates of joints:
```python
is_index_closed  = hand[8].y  > hand[6].y
is_middle_closed = hand[12].y > hand[10].y
is_ring_closed   = hand[16].y > hand[14].y
is_pinky_closed  = hand[20].y > hand[18].y
```

---

## 5. Cursor Movement (Two-Finger Pointing)
Relative position changes (deltas) are scaled by sensitivity and smoothed using an Exponential Moving Average to prevent pointer tremors.

---

## 6. Pinch to Click
Monitors Euclidean distance between index and thumb tips. A debounce counter locks the click trigger until index releases.

---

## 7. Workspace Switch (Open-Hand Swipe)
Monitors wrist positions across 8 frames. Cooldown limits prevent multiple transitions.

---

## 8. Cross-platform Strategy Architecture
Utilizes Strategy pattern. Fallbacks to `pynput` on non-Linux systems or when `evdev` virtual kernel permissions are missing.

---

## 9. How to Run
```bash
python main.py
```
Switch panel mode option to **Control** to initiate system mappings.
