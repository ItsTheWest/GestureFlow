# Documentation: Step 04 — Vowel Recognition (`paso_04_vocales.py`)

Using the real-time async stream (Step 3), this step processes hand landmarks mathematically to statically classify five Spanish vowels ('A', 'E', 'I', 'O', 'U') for **both hands** independently.

For details on path configuration, webcam captures, async streams, queue logic, and skeletal drawing, refer to [REFERENCIA_COMUN.md](../../pasos/REFERENCIA_COMUN.md).

---

## Index

- [1. Step Objective](#1-step-objective)
- [2. Folder Files](#2-folder-files)
- [3. Pipeline](#3-pipeline)
- [4. Vowel Classification Rules](#4-vowel-classification-rules)
- [5. Hand-Based Temporal Validation](#5-hand-based-temporal-validation)
- [6. OpenCV, Keys and Window](#6-opencv-keys-and-window)
- [7. How to Run](#7-how-to-run)
- [8. Common Errors](#8-common-errors)
- [9. Next Steps](#9-next-steps)

---

## 1. Step Objective

**Objective:** Process landmarks for **both hands** in real time to determine if they match vowel shapes in sign language, validating over a 1.0-second window to prevent fluttering predictions.

| Included in this script | Common Reference |
|-------------------------|------------------|
| Geometric classification of open/closed fingers | Specific to this step |
| Finger tip Euclidean distance for 'O' | Specific to this step |
| Independent side-based temporal validation | Specific to this step |
| Vertical layout HUD offset | Specific to this step |
| Queue controls (`listo_para_inferir` & `ANCHO_INFERENCIA`) | [REF §3.4](../../pasos/REFERENCIA_COMUN.md#34-control-de-flujo-de-inferencia-asíncrona) |
| Async real-time stream (`LIVE_STREAM`) | [REF §3.3](../../pasos/REFERENCIA_COMUN.md#33-modos-de-inferencia-runningmode) |

**Success Criteria:**
- Signing a vowel displays `Validando [Vowel]...` or `Confirmada: [Vowel]` on the HUD.
- The validation states of both hands operate independently.
- Removing a hand resets its validation state immediately.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_04_vocales.py](../../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py) | Step script |
| [paso_04_doc.md](../../pasos/paso-04-reconocimiento-vocales/paso_04_doc.md) | Spanish documentation |
| [paso_04_doc_en.md](../../pasos/paso-04-reconocimiento-vocales/paso_04_doc_en.md) | This English documentation |

---

## 3. Pipeline

```text
1. Start HandLandmarker (LIVE_STREAM, ANCHO_INFERENCIA=320)
2. Initialize estado_manos tracker for Left and Right sides
3. while True:
4.      read → flip → display copy
5.      if listo_para_inferir:
6.          BGR → RGB → detect_async()
7.      if results has landmarks:
8.          dibujar_manos()
9.          for each detected hand:
10.             side = handedness[idx].category_name
11.             vowel = get_vowel(landmarks)
12.             update estado_manos[side] (start time and label)
13.         clean state for absent hands
14.         render HUD status with vertical offsets
15.      imshow(display)
16.      waitKey(1) → if 'q', break
17. release + destroyAllWindows
```

---

## 4. Vowel Classification Rules

The `get_vowel` function evaluates finger positions in the `y` axis:
- **'A'**: Index, middle, ring, pinky **closed** (tip below PIP joint). Thumb extended outwards.
- **'E'**: Index, middle, ring, pinky **closed**. Thumb folded in front.
- **'I'**: Index, middle, ring **closed**. Pinky **open** (extended upwards).
- **'U'**: Index, middle **open**. Ring, pinky **closed**.
- **'O'**: Tip distances between thumb and index, and thumb and middle, both under `0.05`.

---

## 5. Hand-Based Temporal Validation
Tracks validation metrics (`vocal_detectada`, `tiempo_inicio`, `confirmada`) inside a global `estado_manos` dictionary for `"Left"` and `"Right"`.

---

## 6. OpenCV, Keys and Window
- **Q**: Quits program.
- **Offsets**: Left hand rendered at `y = 80`, right hand at `y = 120`.

---

## 7. How to Run
```bash
python pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py
```

---

## 8. Common Errors
Refer to Section 6 of [REFERENCIA_COMUN.md](../../pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).

---

## 9. Next Steps
- **Step 5**: Dataset Collection.
- **Step 6**: LSTM Model Training.
