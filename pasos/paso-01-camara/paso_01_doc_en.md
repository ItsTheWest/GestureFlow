# Documentation: Step 01 — Live Camera (`paso_01_camara.py`)

Opens the webcam, displays video in **color** with a **mirror** effect, frame counter, and console logs. Uses OpenCV only; no MediaPipe.

For detail on camera capture concepts, horizontal flipping, window refreshing, and resource releasing, refer to [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## Index

- [1. Step Objective](#1-step-objective)
- [2. Folder Files](#2-folder-files)
- [3. Pipeline](#3-pipeline)
- [4. Imports and Variables](#4-imports-and-variables)
- [5. Code Blocks](#5-code-blocks)
- [6. OpenCV, Keys and Window](#6-opencv-keys-and-window)
- [7. Console: What Logs You Will See](#7-console-what-logs-you-will-see)
- [8. How to Run](#8-how-to-run)
- [9. Common Errors](#9-common-errors)
- [10. Can I Proceed to the Next Step?](#10-can-i-proceed-to-the-next-step)
- [11. Source Code Reference](#11-source-code-reference)

---

## 1. Step Objective

**Objective:** Open the webcam, read frames in a loop, show them in color with mirror orientation, display the frame count on the screen, log progress in the console, and exit with `q` releasing the camera.

| Included in this script | Not included (Steps 2 and 3) |
|-------------------------|------------------------------|
| `VideoCapture`, `read()` loop | `HandLandmarker` / MediaPipe |
| Mirror `cv2.flip(frame, 1)` | Landmarks, circles and lines |
| Console logs + `putText` | `LIVE_STREAM` mode |
| Exit with `q` | Gesture detection |

**Success Criteria:**
- Window named `Paso 01 - Camara` with color and mirror video.
- Frame counter visible and rising.
- Console: first frame message, logs every 100 frames, total on exit.
- `q` closes without leaving the camera locked.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_01_camara.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-01-camara/paso_01_camara.py) | Step script |
| [paso_01_doc.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-01-camara/paso_01_doc.md) | Spanish documentation |
| [paso_01_doc_en.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-01-camara/paso_01_doc_en.md) | This English documentation |

---

## 3. Pipeline

```text
1. import cv2
2. VideoCapture(0) → cap
3. isOpened() → if fails: print + exit(1)
4. frame_count = 0, first_frame_logged = False
5. while True:
6.      read() → ret, frame
7.      if not ret → log error with frame_count + break
8.      flip(frame, 1)        → mirror
9.      frame_count += 1
10.     console logs (1st frame / every 100)
11.     putText (counter + "Press Q")
12.     imshow (color)
13.     waitKey(1) → if 'q', break
14. print total frames
15. release() + destroyAllWindows()
```

---

## 4. Imports and Variables

### `import cv2`
OpenCV: capture, mirror flipping, text rendering, display, and resource releasing.

### Variables Table

| Name | Role | General Concept |
|------|------|-----------------|
| `cap` | Camera capture object. | [REF §2.1](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#21-captura-de-cámara-cv2videocapture) |
| `ret` | `True` if `read()` returned a valid frame. | [REF §2.1](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#21-captura-de-cámara-cv2videocapture) |
| `frame` | BGR image `(height, width, 3)`. | [REF §2.1](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#21-captura-de-cámara-cv2videocapture) |
| `frame_count` | Successfully read frames count. | Specific to this step |
| `first_frame_logged` | Prevents repeating details for the first frame. | Local flow control |

---

## 5. Code Blocks

### Camera Open and Fatal Error
Opens device `0`. If opening fails, the script ends.

### Mirror and Counter
Applies horizontal flip to create a mirror view.

### HUD and Display
Renders frame count and key instructions.

### Keyboard Control (`q`)
Ends the loop when `q` is pressed.

---

## 6. OpenCV, Keys and Window
Refer to Section 2 of [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#2-opencv-y-captura-de-vídeo) for functions details.

---

## 7. Console: What Logs You Will See
```text
Primer frame OK: shape=(480, 640, 3), dtype=uint8
Frames OK: 100
Frames OK: 200
Total frames leidos: 247
```

---

## 8. How to Run
```bash
python pasos/paso-01-camara/paso_01_camara.py
```

---

## 9. Common Errors
Refer to Section 6 of [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).

---

## 10. Can I Proceed to the Next Step?
Yes, if:
- Window shows color mirrored video.
- Frame counter climbs continuously.
- Console shows `Primer frame OK`.
- Pressing `q` prints total frames and closes safely.
