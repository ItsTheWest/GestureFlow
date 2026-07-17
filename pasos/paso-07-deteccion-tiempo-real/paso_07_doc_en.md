# Documentation: Step 07 — Real-Time Detection (`paso_07_deteccion.py`)

Real-time **inference pipeline**: loads the trained LSTM model from step 6, captures video, buffers landmarks in a circular queue of 30 frames, and classifies gesture classes in real time.

For details on keypoint extraction formats, MediaPipe `VIDEO` mode, and tensor inputs, refer to [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## Index

- [1. Step Objective](#1-step-objective)
- [2. Folder Files](#2-folder-files)
- [3. Pipeline](#3-pipeline)
- [4. VIDEO vs LIVE_STREAM Modes](#4-video-vs-live_stream-modes)
- [5. Circular Buffer and Sliding Windows](#5-circular-buffer-and-sliding-windows)
- [6. Asynchronous Prediction Threads](#6-asynchronous-prediction-threads)
- [7. Confidence Thresholds](#7-confidence-thresholds)
- [8. How to Run](#8-how-to-run)
- [9. Common Errors](#9-common-errors)

---

## 1. Step Objective

**Objective:** Map live camera feeds to custom gestures in real time using the trained LSTM model.

| Included in this script | Not included |
|-------------------------|-------------|
| Model loader (.keras) | Model training (Step 6) |
| MediaPipe VIDEO mode configuration | OS input events (Step 8) |
| Circular queue `deque(maxlen=30)` | GUI interface layout |
| Async prediction threads | |

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_07_deteccion.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-07-deteccion-tiempo-real/paso_07_deteccion.py) | Step script |
| [paso_07_doc.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-07-deteccion-tiempo-real/paso_07_doc.md) | Spanish documentation |
| [paso_07_doc_en.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-07-deteccion-tiempo-real/paso_07_doc_en.md) | This English documentation |

---

## 3. Pipeline
- Captures frame → Mirror-flips BGR to RGB.
- Submits to HandLandmarker in VIDEO mode.
- Extracts (126,) keypoints and appends to circular queue.
- If queue length reaches 30 frames, spawns background prediction thread.
- If prediction probability > 0.80, updates UI text display.

---

## 4. VIDEO vs LIVE_STREAM Modes
Unlike Step 3's callback-based asynchrony, `VIDEO` mode uses synchronous landmark queries. Timestamps must increase sequentially in milliseconds to avoid model errors.

---

## 5. Circular Buffer and Sliding Windows
Using `collections.deque(maxlen=30)` allows continuous rolling inference on the last 30 frames. When hands disappear, the buffer is cleared to avoid classification on stale frames.

---

## 6. Asynchronous Prediction Threads
Spawning predictions in secondary threads keeps UI loops rendering at high frame rates. A `prediction_lock` mutex prevents thread multiplication.

---

## 7. Confidence Thresholds
Outputs are validated using a minimum confidence constant (`CONFIDENCE_THRESHOLD = 0.8`). If maximum softmax value is lower, predictions are ignored.

---

## 8. How to Run
```bash
python pasos/paso-07-deteccion-tiempo-real/paso_07_deteccion.py
```
