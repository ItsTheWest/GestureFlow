# Documentación: Paso 01 — Cámara en vivo (`paso_01_camara.py`)

Describe **qué hace cada parte** del script, **cómo comprobar** que la webcam entrega frames y **si puedes pasar al siguiente paso** del plan.

---

## Índice

- [1. Objetivo del paso](#1-objetivo-del-paso)
- [2. Estado respecto al plan](#2-estado-respecto-al-plan)
- [3. Archivos de esta carpeta](#3-archivos-de-esta-carpeta)
- [4. Pipeline](#4-pipeline)
- [5. Importaciones y variables](#5-importaciones-y-variables)
- [6. Bloques del código (línea por línea)](#6-bloques-del-código-línea-por-línea)
- [7. Funciones de OpenCV](#7-funciones-de-opencv)
- [8. Consola: qué logs verás](#8-consola-qué-logs-verás)
- [9. Cómo ejecutar](#9-cómo-ejecutar)
- [10. Errores frecuentes](#10-errores-frecuentes)
- [11. ¿Puedo ir al siguiente paso?](#11-puedo-ir-al-siguiente-paso)
- [12. Referencia del código fuente](#12-referencia-del-código-fuente)

---

## 1. Objetivo del paso

**Objetivo:** abrir la webcam, leer frames en bucle, mostrarlos en **color** con orientación tipo **espejo**, ver el número de frame en pantalla y **información de frames en consola**, y salir con `q` liberando la cámara.

| Incluido en este script | No incluido (pasos 3+) |
|-------------------------|-------------------------|
| `VideoCapture`, bucle `read()` | `HandLandmarker` / MediaPipe |
| Espejo `cv2.flip(frame, 1)` | Landmarks, círculos y líneas |
| Logs en consola + `putText` | Modo `LIVE_STREAM` |
| Salida con `q` | Detección de gestos |

**Criterio de éxito:**

- Ventana `Paso 01 - Camara` con vídeo en color y espejo.
- Contador de frame visible y subiendo.
- Consola: mensaje del primer frame, logs cada 100 frames, total al salir.
- `q` cierra sin dejar la cámara bloqueada.

---

## 2. Estado respecto al plan

| Paso del plan | ¿Cubierto? | Notas |
|---------------|------------|--------|
| **Paso 1** — abrir cámara y leer frames | Sí | Incluye bucle (no solo un frame con `waitKey(0)`). |
| **Paso 2** — bucle en vivo + `q` | Sí | Mismo archivo; no hace falta duplicar en `paso-02-bucle` salvo que quieras carpeta aparte para estudiar. |
| **Extras** — espejo, color, logs | Sí | Van más allá del plan mínimo del paso 1. |
| **Paso 3** — MediaPipe en un frame | Pendiente | Siguiente paso recomendado. |

---

## 3. Archivos de esta carpeta

| Archivo | Rol |
|---------|-----|
| `paso_01_camara.py` | Script del paso |
| `paso_01_doc.md` | Esta documentación |

**Dependencias (raíz del proyecto):** `opencv-python` en `requirements.txt`, entorno `venv/`.

**No necesitas:** `prueba/hand_landmarker.task`, imágenes en `assets/`.

---

## 4. Pipeline

```text
1. import cv2
2. VideoCapture(0) → cap
3. isOpened() → si falla: print + exit(1)
4. frame_count = 0, primer_frame_logeado = False
5. while True:
     read() → ret, frame
     si not ret → log con frame_count + break
     flip(frame, 1)        → espejo
     frame_count += 1
     logs consola (1.er frame / cada 100)
     putText (contador + "Pulsa Q")
     imshow (color)
     waitKey(1) → si 'q', break
6. print total frames
7. release() + destroyAllWindows()
```

```mermaid
flowchart TD
    A[VideoCapture] --> B{isOpened?}
    B -->|No| C[exit 1]
    B -->|Sí| D[read]
    D --> E{ret?}
    E -->|No| F[log error + break]
    E -->|Sí| G[flip + contador + logs]
    G --> H[putText + imshow]
    H --> I{q?}
    I -->|No| D
    I -->|Sí| J[total + release]
```

---

## 5. Importaciones y variables

### `import cv2`

OpenCV: captura, `flip`, `putText`, `imshow`, `waitKey`, liberación de recursos. Los frames están en **BGR** (3 canales). En el paso 3 convertirás a **RGB** para MediaPipe.

### Tabla de variables

| Nombre | Significado |
|--------|-------------|
| `cap` | Objeto de captura de la cámara. |
| `ret` | `True` si `read()` devolvió un frame válido. |
| `frame` | Imagen BGR `(alto, ancho, 3)`, p. ej. `(480, 640, 3)`. |
| `frame_count` | Frames leídos con éxito desde el inicio del bucle. |
| `primer_frame_logeado` | Evita repetir el log detallado del primer frame. |

---

## 6. Bloques del código (línea por línea)

### Líneas 1–7 — Apertura y error fatal

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit(1)
```

Abre el dispositivo `0` (primera cámara). `exit(1)` indica error al sistema/terminal.

---

### Líneas 9–10 — Inicialización antes del bucle

```python
frame_count = 0
primer_frame_logeado = False
```

Sin `frame_count = 0`, la línea `frame_count += 1` lanzaría `NameError`.

---

### Líneas 12–17 — Lectura y validación

```python
ret, frame = cap.read()
if not ret:
    print(f"Error: No se pudo leer el frame (tras {frame_count} frames OK)")
    break
```

**Orden importante:** primero `read()`, luego comprobar `ret`. Solo después se modifica `frame` (flip, texto).

---

### Líneas 19–20 — Espejo y contador

```python
frame = cv2.flip(frame, 1)
frame_count += 1
```

| `flip(..., 1)` | Volteo **horizontal** (efecto espejo). |
| `0` | Volteo vertical. |
| `-1` | Ambos ejes. |

---

### Líneas 22–26 — Logs en consola

```python
if not primer_frame_logeado:
    print(f"Primer frame OK: shape={frame.shape}, dtype={frame.dtype}")
    primer_frame_logeado = True
elif frame_count % 100 == 0:
    print(f"Frames OK: {frame_count}")
```

| Momento | Qué imprime |
|---------|-------------|
| Primer frame OK | `shape` y `dtype` (comprueba que hay imagen real). |
| Cada 100 frames | Progreso sin llenar la terminal (a ~30 FPS ≈ cada 3 s). |
| Tras el bucle (l. 37) | `Total frames leidos: N` |

No se hace `print` en cada frame a propósito (demasiado ruido).

---

### Líneas 28–32 — Texto y ventana (color)

```python
cv2.putText(frame, f"Frame: {frame_count}", ...)
cv2.putText(frame, "Pulsa Q para salir", ...)
cv2.imshow("Paso 01 - Camara", frame)
```

Dibuja en el **mismo** `frame` en BGR y muestra una sola ventana. Color verde en BGR: `(0, 255, 0)`.

**No hay** `cvtColor` a gris: el vídeo se ve a color. MediaPipe en pasos posteriores usará RGB, no escala de grises.

---

### Líneas 34–35 — Salida con Q

```python
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

`waitKey(1)` refresca la ventana (~1 ms). `& 0xFF` evita lecturas erróneas de tecla en Windows.

---

### Líneas 37–39 — Cierre

```python
print(f"Total frames leidos: {frame_count}")
cap.release()
cv2.destroyAllWindows()
```

Siempre se ejecuta tras salir del `while` (por `q` o por error de `read()`).

---

## 7. Funciones de OpenCV

| Función | Uso en este paso |
|---------|------------------|
| `VideoCapture(0)` | Abrir cámara. |
| `isOpened()` | Comprobar apertura. |
| `read()` | Siguiente frame → `(ret, frame)`. |
| `flip(image, 1)` | Espejo horizontal. |
| `putText(...)` | Contador y ayuda en pantalla. |
| `imshow(title, frame)` | Mostrar BGR en vivo. |
| `waitKey(1)` | Eventos de ventana + teclado. |
| `release()` | Liberar cámara. |
| `destroyAllWindows()` | Cerrar ventanas. |

Enlace: [Tutorial vídeo OpenCV](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)

---

## 8. Consola: qué logs verás

Ejemplo típico al ejecutar y pulsar `q` tras unos segundos:

```text
Primer frame OK: shape=(480, 640, 3), dtype=uint8
Frames OK: 100
Frames OK: 200
Total frames leidos: 247
```

| Log | Significado |
|-----|-------------|
| `Primer frame OK` | Al menos un frame válido; `shape` con 3 canales = color BGR. |
| `Frames OK: N` | El bucle sigue leyendo bien cada 100 iteraciones. |
| `Total frames leidos` | Cuántos frames procesaste antes de salir. |
| `Error: No se pudo abrir...` | Índice de cámara o permisos. |
| `Error: No se pudo leer el frame (tras X frames OK)` | Fallo a mitad de sesión. |

---

## 9. Cómo ejecutar

Desde la raíz del proyecto, con `venv` activado:

```powershell
python pasos/paso-01-camara/paso_01_camara.py
```

| En pantalla | En consola |
|-------------|------------|
| Vídeo color, espejo | Primer `shape` / `dtype` |
| `Frame: N` en verde | Cada 100 frames |
| `Pulsa Q para salir` | Total al cerrar con `q` |

---

## 10. Errores frecuentes

| Síntoma | Qué revisar |
|---------|-------------|
| Ventana negra / congelada | `waitKey(1)` presente; probar `VideoCapture(1)`. |
| Sin logs salvo error | ¿Llegaste a leer al menos un frame? ¿Saliste antes del frame 100? |
| Imagen sin espejo | ¿`flip(frame, 1)` está **después** de `if not ret`? |
| `NameError: frame_count` | Falta inicialización antes del `while`. |
| Cámara ocupada tras crash | Falta `release()`; en pasos futuros usar `try/finally`. |

---

## 11. ¿Puedo ir al siguiente paso?

**Sí**, si al ejecutar este script se cumple todo esto:

- [ ] La ventana muestra vídeo **en color** y orientación **espejo** aceptable.
- [ ] El número de frame en pantalla **sube** de forma continua.
- [ ] En consola aparece **`Primer frame OK`** con `shape` de 3 dimensiones.
- [ ] Al pulsar **`q`**, ves **`Total frames leidos`** y el programa termina sin colgar.
- [ ] (Opcional) Tras ~3 s de vídeo, aparece al menos un **`Frames OK: 100`** si no sales antes.

### Qué sigue (Paso 3 del plan)

Crear carpeta sugerida: `pasos/paso-03-detectar/` con un script que:

1. Reutilice el bucle de cámara (o capture **un** frame con tecla).
2. Cargue `prueba/hand_landmarker.task` con `Path`.
3. Use `RunningMode.IMAGE` y `detect()` como en `prueba/prueba.py`.
4. Dibuje landmarks (círculos y líneas) sobre el frame de la cámara.

**Paso 2 del plan** (bucle + `q`): ya lo tienes integrado aquí; puedes **saltarlo** como carpeta nueva o copiar este script a `paso-02-bucle` solo como repaso.

**No hace falta** repetir el Paso 1 salvo que quieras un script mínimo de un solo frame para comparar.

---

## 12. Referencia del código fuente

```1:39:pasos/paso-01-camara/paso_01_camara.py
import cv2

cap = cv2.VideoCapture(0)  # 0 = cámara por defecto; prueba 1 si no abre

if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit(1)

frame_count = 0
primer_frame_logeado = False

while True:
    ret, frame = cap.read()

    if not ret:
        print(f"Error: No se pudo leer el frame (tras {frame_count} frames OK)") # Log de error
        break # Salir del bucle si no se pudo leer el frame

    frame = cv2.flip(frame, 1)  # espejo horizontal (orientación natural)
    frame_count += 1 # Incrementar el contador de frames

    if not primer_frame_logeado:
        print(f"Primer frame OK: shape={frame.shape}, dtype={frame.dtype}") # Log de primer frame OK
        primer_frame_logeado = True
    elif frame_count % 100 == 0: # Log de frames OK cada 100 frames
        print(f"Frames OK: {frame_count}") # Log de frames OK

    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Texto de frames OK
    cv2.putText(frame,
        "Pulsa Q para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Texto de salida

    cv2.imshow("Paso 01 - Camara", frame) # Mostrar el frame en la ventana

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break # Salir del bucle si se presiona la tecla 'q'

print(f"Total frames leidos: {frame_count}") # Log de total frames leidos
cap.release()
cv2.destroyAllWindows()
```

*Fuente de verdad: el archivo `.py` en disco.*
