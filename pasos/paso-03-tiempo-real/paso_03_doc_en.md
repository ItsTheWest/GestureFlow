# Documentation: Step 03 — Real-Time (`paso_03_tiempo_real.py`)

Hand landmark detection **on every frame** using MediaPipe in asynchronous `LIVE_STREAM` mode, removing the need for key triggers.

For detailed concepts on real-time execution, callback setups, increasing timestamps, and loop lag controls, refer to [REFERENCIA_COMUN.md](../../pasos/REFERENCIA_COMUN.md).

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
- [10. Next Steps](#10-next-steps)

---

## 1. Step Objective

**Objective:** Get a real-time mirrored camera display where hand skeletons **update and follow hand movements in real time**, using `detect_async` and `on_result` callback.

| Included in this script | No longer required |
|-------------------------|--------------------|
| `RunningMode.LIVE_STREAM` | Pressing SPACE to process |
| Callback `on_result` updates | `cv2.waitKey(0)` pauses |
| Async `detect_async` every frame | Synchronous blocking `detect()` |

**Success Criteria:**
- Hand skeleton overlay smoothly tracks hands in real time.
- Pressing **Q** exits instantly.
- Without hands visible, video keeps running at normal camera FPS.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_03_tiempo_real.py](../../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py) | Step script |
| [paso_03_doc.md](../../pasos/paso-03-tiempo-real/paso_03_doc.md) | Spanish documentation |
| [paso_03_doc_en.md](../../pasos/paso-03-tiempo-real/paso_03_doc_en.md) | This English documentation |

---

## 3. Pipeline

```text
1. MODEL_PATH + VideoCapture
2. HandLandmarker (LIVE_STREAM + result_callback)
3. ultimo_resultado = None
4. while True:
5.      read → flip
6.      display = frame copy
7.      if ultimo_resultado → dibujar_manos(display, ultimo_resultado)
8.      putText + imshow(display)
9.      mp.Image RGB from current frame
10.     detect_async(mp_image, increasing timestamp_ms)
11.     waitKey(1) → if 'q', break
12. release + destroyAllWindows
```

---

## 4. Imports and Variables

Refer to [REFERENCIA_COMUN.md](../../pasos/REFERENCIA_COMUN.md) for details.

### Specific Variables Table

| Variable | Role | General Concept |
|----------|------|-----------------|
| `ultimo_resultado` | Global container for latest callback results. | [REF §3.3](../../pasos/REFERENCIA_COMUN.md#33-modos-de-inferencia-runningmode) |
| `on_result` | Callback function executed by MediaPipe. | [REF §3.3](../../pasos/REFERENCIA_COMUN.md#33-modos-de-inferencia-runningmode) |
| `display` | Frame copy for skeleton rendering. | Frame buffer isolation |

---

## 5. Code Blocks

### Callback `on_result`
Saves results into `ultimo_resultado` as soon as inference finishes.

### Asynchronous Inference (`detect_async`)
Converts frames to RGB, wraps in `mp.Image`, and passes to `detect_async` with growing timestamps.

### Queue Control and Resizing
Although this script sends every frame, advanced steps resize frames (`ANCHO_INFERENCIA`) to avoid queue accumulation.

---

## 6. OpenCV, Keys and Window
- **Q**: Quits immediately.
- **cv2.waitKey(1)**: Ensures video loop rendering runs smoothly at ~30 FPS.

---

## 7. Console: What Logs You Will See
```text
Deteccion en tiempo real | Q = salir
```

---

## 8. How to Run
```bash
python pasos/paso-03-tiempo-real/paso_03_tiempo_real.py
```

---

## 9. Common Errors
Refer to Section 6 of [REFERENCIA_COMUN.md](../../pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).

---

## 10. Next Steps
Move to:
- **Step 4**: Vowel Recognition (rule-based geometric classification).
- **Step 5**: Dataset Collection.
