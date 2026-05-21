# Referencia común — pasos de cámara y MediaPipe

Patrones que **se repiten** en `paso-02-dibujo`, `paso-03-tiempo-real` y en `prueba/prueba.py`. Cada `paso_XX_doc.md` enlaza aquí en lugar de copiar todo.

---

## Índice

- [1. Estructura de carpetas](#1-estructura-de-carpetas)
- [2. Rutas con Path](#2-rutas-con-path)
- [3. Cámara OpenCV](#3-cámara-opencv)
- [4. Modelo HandLandmarker](#4-modelo-handlandmarker)
- [5. BGR → RGB y mp.Image](#5-bgr--rgb-y-mpimage)
- [6. Dibujar landmarks](#6-dibujar-landmarks)
- [7. Modos IMAGE vs LIVE_STREAM](#7-modos-image-vs-live_stream)
- [8. Cierre de recursos](#8-cierre-de-recursos)
- [9. Errores frecuentes (todos los pasos)](#9-errores-frecuentes-todos-los-pasos)
- [10. Documentación por paso](#10-documentación-por-paso)

---

## 1. Estructura de carpetas

```text
GestureFlow/
├── prueba/
│   ├── prueba.py                    ← Fase 0: imagen fija
│   ├── hand_landmarker.task         ← modelo (compartido)
│   └── DOCUMENTACION_PRUEBA.md
├── pasos/
│   ├── REFERENCIA_COMUN.md          ← este archivo
│   ├── paso-01-camara/
│   ├── paso-02-dibujo/
│   └── paso-03-tiempo-real/
└── requirements.txt
```

**Ejecutar desde la raíz** (con `venv` activado):

```powershell
python pasos/paso-02-dibujo/paso_02_dibujo.py
python pasos/paso-03-tiempo-real/paso_03_tiempo_real.py
```

---

## 2. Rutas con Path

Mismo bloque en pasos 2 y 3 (y variante en `prueba.py`):

```python
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent   # GestureFlow/
MODEL_PATH = PROJECT_ROOT / "prueba" / "hand_landmarker.task"
```

| Variable | Valor típico |
|----------|----------------|
| `SCRIPT_DIR` | `pasos/paso-02-dibujo/` o `paso-03-tiempo-real/` |
| `PROJECT_ROOT` | Raíz del repo (dos niveles arriba del script) |
| `MODEL_PATH` | `prueba/hand_landmarker.task` |

Comprobar antes de abrir la cámara:

```python
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")
```

---

## 3. Cámara OpenCV

| Pieza | Uso |
|-------|-----|
| `cv2.VideoCapture(0)` | Cámara por defecto (`1` si falla) |
| `cap.isOpened()` | Error fatal si no abre |
| `ret, frame = cap.read()` | Un frame BGR por iteración |
| `cv2.flip(frame, 1)` | Espejo horizontal |
| `cv2.waitKey(1) & 0xFF` | Bucle fluido + tecla `q` |
| `cap.release()` + `destroyAllWindows()` | Siempre al salir |

Detalle del paso 1 (solo OpenCV): [paso_01_doc.md](paso-01-camara/paso_01_doc.md).

---

## 4. Modelo HandLandmarker

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=...,   # IMAGE o LIVE_STREAM (ver §7)
    num_hands=2,
)
```

| Opción | Valor habitual |
|--------|----------------|
| `num_hands` | `2` (máximo a buscar) |
| `model_asset_path` | Ruta absoluta al `.task` como `str` |

Context manager recomendado:

```python
with vision.HandLandmarker.create_from_options(options) as landmarker:
    ...
```

---

## 5. BGR → RGB y mp.Image

OpenCV usa **BGR**; MediaPipe Tasks espera **SRGB**:

```python
import mediapipe as mp
import cv2

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
)
```

- **Paso 2:** `landmarker.detect(mp_image)` — síncrono, un frame (p. ej. al pulsar ESPACIO).
- **Paso 3:** `landmarker.detect_async(mp_image, timestamp_ms)` — asíncrono, cada frame del bucle.

---

## 6. Dibujar landmarks

La API Tasks devuelve landmarks “simples”; `draw_landmarks` necesita **protobuf**:

```python
from mediapipe.framework.formats import landmark_pb2

def dibujar_manos(frame, results):
    if not results.hand_landmarks:
        return False
    for hand_landmarks in results.hand_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            frame, proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )
    return True
```

Coordenadas **normalizadas** (0–1 respecto al ancho/alto del frame). Más detalle en [DOCUMENTACION_PRUEBA.md](../prueba/DOCUMENTACION_PRUEBA.md).

---

## 7. Modos IMAGE vs LIVE_STREAM

| | **Paso 2** (`IMAGE`) | **Paso 3** (`LIVE_STREAM`) |
|--|----------------------|----------------------------|
| Modo | `RunningMode.IMAGE` | `RunningMode.LIVE_STREAM` |
| Inferencia | `detect(mp_image)` | `detect_async(mp_image, timestamp_ms)` |
| Cuándo | Al pulsar ESPACIO (frame congelado) | Cada frame del bucle |
| Resultado | Devuelto al instante | Llega en `result_callback` |
| `waitKey` | `waitKey(0)` tras dibujar (pausa) | Solo `waitKey(1)` en bucle |
| Timestamp | No hace falta | **Obligatorio** y creciente |

**Paso 3 — callback y estado:**

```python
ultimo_resultado = None

def on_result(result, output_image, timestamp_ms):
    global ultimo_resultado
    ultimo_resultado = result

options = vision.HandLandmarkerOptions(
    ...,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=on_result,
)
```

En el bucle principal: dibujar `ultimo_resultado` sobre el frame **actual** y llamar `detect_async` solo cuando el callback anterior terminó (evita cola y retardo). Timestamp con `time.perf_counter()` en ms; inferencia opcional en frame reducido (landmarks 0–1).

Enlaces:

- [Hand Landmarker Python](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python)
- [RunningMode](https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/python/vision/RunningMode)

---

## 8. Cierre de recursos

Orden habitual al salir del `while` (tecla `q` o error de `read`):

1. Salir del `with landmarker` (libera el modelo).
2. `cap.release()`
3. `cv2.destroyAllWindows()`

Si el programa crashea, la cámara puede quedar ocupada hasta reiniciar el script o el IDE.

---

## 9. Errores frecuentes (todos los pasos)

| Síntoma | Revisar |
|---------|---------|
| `FileNotFoundError` del modelo | ¿Existe `prueba/hand_landmarker.task`? |
| Ventana negra | `ret`, índice de cámara `0`/`1`, `waitKey(1)` |
| No se dibuja nada | BGR→RGB, mano visible, iluminación |
| Paso 3 “congelado” o sin dibujo | ¿`result_callback` definido? ¿`timestamp_ms` **sube** cada frame? |
| Paso 3 muy lento | Normal en CPU; reduce resolución de cámara si hace falta |
| Crash al cerrar | `release()` y `destroyAllWindows()` |

---

## 10. Documentación por paso

| Paso | Script | Documentación |
|------|--------|----------------|
| 01 — Cámara | `paso_01_camara.py` | [paso_01_doc.md](paso-01-camara/paso_01_doc.md) |
| 02 — Dibujo (IMAGE + ESPACIO) | `paso_02_dibujo.py` | [paso_02_doc.md](paso-02-dibujo/paso_02_doc.md) |
| 03 — Tiempo real (LIVE_STREAM) | `paso_03_tiempo_real.py` | [paso_03_doc.md](paso-03-tiempo-real/paso_03_doc.md) |
| Fase 0 — Imagen fija | `prueba/prueba.py` | [DOCUMENTACION_PRUEBA.md](../prueba/DOCUMENTACION_PRUEBA.md) |

**Orden de aprendizaje:** 01 → 02 → 03 (documentación en cada `paso_XX_doc.md`).
