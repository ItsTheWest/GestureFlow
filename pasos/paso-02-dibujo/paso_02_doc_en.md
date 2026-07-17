# Documentation: Step 02 — Drawing with SPACE (`paso_02_dibujo.py`)

Bridge between **live camera** (step 1) and **continuous detection** (step 3): pressing **SPACE** freezes a frame, processes it with MediaPipe in **IMAGE** (synchronous) mode, and draws the hand landmark skeleton.

For detail on path resolutions, model setup, color spaces, and landmarks drawing, refer to [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

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

**Objective:** Keep live camera rendering from step 1, freeze a frame when **SPACE** is pressed, detect up to 2 hands with `HandLandmarker` in `IMAGE` mode, draw landmarks, and pause display until another key is pressed.

| Included in this script | Not included (Step 3) |
|-------------------------|----------------------|
| MediaPipe + `.task` model | `RunningMode.LIVE_STREAM` |
| Synchronous `detect()` | `detect_async()` |
| SPACE key freeze | Automated frame processing |
| Mirror + loop + `q` | `on_result` callback |

**Success Criteria:**
- Smooth video display with HUD guide.
- Pressing **SPACE** freezes frame and draws hand skeleton.
- Console shows `Manos detectadas: N` or `No se detectaron manos`.
- **Q** exits safely.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_02_dibujo.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-02-dibujo/paso_02_dibujo.py) | Step script |
| [paso_02_doc.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-02-dibujo/paso_02_doc.md) | Spanish documentation |
| [paso_02_doc_en.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-02-dibujo/paso_02_doc_en.md) | This English documentation |

---

## 3. Pipeline

```text
1. Check MODEL_PATH
2. VideoCapture(0)
3. HandLandmarker (RunningMode.IMAGE)
4. while True:
5.      read → flip
6.      preview = frame copy (no drawing)
7.      imshow(preview) + waitKey(1)
8.      if 'q' → break
9.      if SPACE:
10.         snapshot = frame copy
11.         BGR → RGB → mp.Image
12.         results = detect(snapshot)
13.         dibujar_manos(snapshot, results)
14.         imshow(snapshot) + waitKey(0)   ← pause
15. release + destroyAllWindows
```

---

## 4. Imports and Variables

Refer to [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md) for details.

### Specific Variables Table

| Variable | Role | General Concept |
|----------|------|-----------------|
| `preview` | Frame copy for clean live preview. | Local UI |
| `snapshot` | Frozen frame copy used for inference. | Freeze frame logic |
| `landmarker` | MediaPipe hand landmark detector. | [REF §3.1](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#31-carga-del-modelo-handlandmarkeroptions-y-baseoptions) |

---

## 5. Code Blocks

### Hand Drawing (`dibujar_manos`)
Draws joints and bones skeleton on the frame.

### HandLandmarker `IMAGE` mode
Setup for single-frame synchronous inference.

### Frame Freeze Logic (SPACE)
Duplicates frame, runs `detect()` synchronously, draws landmarks, and blocks loop with `cv2.waitKey(0)` acting as a pause until keypress.

---

## 6. OpenCV, Keys and Window
- **SPACE**: Freezes current frame to run inference.
- **Q**: Quits script (works in live view).
- **Any other key**: Unfreezes paused frame.

---

## 7. Console: What Logs You Will See
```text
ESPACIO = detectar y dibujar manos | Q = salir
Manos detectadas: 1
No se detectaron manos
```

---

## 8. How to Run
```bash
python pasos/paso-02-dibujo/paso_02_dibujo.py
```

---

## 9. Common Errors
Refer to Section 6 of [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).

---

## 10. Can I Proceed to the Next Step?
Yes, if:
- SPACE freezes frame and draws hand skeleton when a hand is visible.
- Terminal logs correct hand counts.
- Exits cleanly on `q`.
