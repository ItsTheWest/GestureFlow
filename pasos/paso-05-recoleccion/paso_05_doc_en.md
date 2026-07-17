# Documentation: Step 05 — Dataset Collection (`paso_05_recoleccion.py`)

Data collection pipeline for **model training**: opens camera, detects hands using MediaPipe in `IMAGE` mode, and stores sequences as `.npy` arrays inside folder structures.

For details on keypoint extraction (126 coordinates), padding, sequences, and temporal overlays, refer to [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## Index

- [1. Step Objective](#1-step-objective)
- [2. Folder Files](#2-folder-files)
- [3. Pipeline](#3-pipeline)
- [4. Recording Constants](#4-recording-constants)
- [5. Code Functions](#5-code-functions)
- [6. Three Recording Phases](#6-three-recording-phases)
- [7. Output Structure](#7-output-structure)
- [8. How to Run](#8-how-to-run)
- [9. Common Errors](#9-common-errors)

---

## 1. Step Objective

**Objective:** Capture dynamic gesture trajectories. Each gesture is recorded as a sequence of 30 frames and saved in `.npy` files.

| Included in this script | Not included (Step 6) |
|-------------------------|----------------------|
| Synchronous `RunningMode.IMAGE` | LSTM Model Training |
| Circular buffer `deque(maxlen=30)` | Real-Time Predictions |
| Auto-save stride (`SAVE_EVERY=15`) | Accuracy Metrics |

**Success Criteria:**
- Prompt asks for gesture name.
- Pressing **SPACE** starts countdown.
- Auto-generates 30 sequences of shape `(30, 126)` in `gestos/<gesture_name>/`.
- Resuming with same name starts indexing from the last saved file.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_05_recoleccion.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-05-recoleccion/paso_05_recoleccion.py) | Step script |
| [paso_05_doc.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-05-recoleccion/paso_05_doc.md) | Spanish documentation |
| [paso_05_doc_en.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-05-recoleccion/paso_05_doc_en.md) | This English documentation |

---

## 3. Pipeline

```text
1. Ask for gesture name and resume index.
2. Initialize HandLandmarker in IMAGE mode.
3. run recording:
4.      Phase 0 — Waiting:
5.        read → flip → show instructions
6.        SPACE → advance to Phase 1
7.      Phase 1 — Countdown:
8.        show 3, 2, 1 with live camera
9.      Phase 2 — Recording:
10.       while collected sequences < 30:
11.         read → flip → RGB → detect()
12.         extract_keypoints() → buffer.append()
13.         if buffer full and hand visible and frame % 15 == 0:
14.           np.save() → gestos/<gesture_name>/<index>.npy
```

---

## 4. Recording Constants

| Constant | Value | Concept | Reference |
|----------|-------|---------|-----------|
| `SEQUENCE_LENGTH` | `30` | Temporal frames count | [REF §4.4](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#44-formato-npy-y-secuencias-temporales) |
| `NUM_FEATURES` | `126` | 21 points × 3 (x,y,z) × 2 hands | [REF §4.3](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#43-extracción-y-relleno-de-características-extract_keypoints) |
| `SAVE_EVERY` | `15` | Window stride (50% overlap) | [REF §4.5](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#45-solapamiento-temporal-aumento-de-datos) |

---

## 5. Code Functions
- `build_landmarker()`: Configures detector in `IMAGE` mode.
- `extract_keypoints()`: Flattens landmarks to `(126,)` array.
- `draw_hud()`: Renders progress, status, and alerts.

---

## 6. Three Recording Phases
- **Phase 0 — Waiting**: Standby until SPACE is hit.
- **Phase 1 — Countdown**: Renders `3-2-1` without using `time.sleep()`.
- **Phase 2 — Recording**: Records frames and auto-saves once buffer reaches 30 frames.

---

## 7. Output Structure
```text
gestos/
└── saludar/
    ├── 0.npy   → shape (30, 126)
    └── 29.npy  → shape (30, 126)
```

---

## 8. How to Run
```bash
python pasos/paso-05-recoleccion/paso_05_recoleccion.py
```

---

## 9. Common Errors
Refer to Section 6 of [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).
