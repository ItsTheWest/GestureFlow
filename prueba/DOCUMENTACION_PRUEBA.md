# Documentación detallada: `prueba.py`

Este documento describe **línea por línea** (conceptualmente) qué hace el script de prueba, qué importa cada módulo, qué significa cada variable, qué hace cada función invocada y **cuál es el proceso completo** desde cargar una foto hasta mostrar una imagen con las marcas de la mano (landmarks y conexiones).

---

## Índice

- [1. Objetivo de la prueba](#1-objetivo-de-la-prueba)
- [2. Archivos necesarios para la prueba](#2-archivos-necesarios-para-la-prueba)
- [3. Proceso completo (pipeline)](#3-proceso-completo-pipeline)
- [4. Importaciones](#4-importaciones)
  - [4.1 `import cv2`](#41-import-cv2)
  - [4.2 `import mediapipe as mp`](#42-import-mediapipe-as-mp)
  - [4.3 `from mediapipe.tasks import python`](#43-from-mediapipectasks-import-python)
  - [4.4 `from mediapipe.tasks.python import vision`](#44-from-mediapipectaskspython-import-vision)
  - [4.5 `landmark_pb2`](#45-landmark_pb2)
  - [4.6 `from pathlib import Path`](#46-from-pathlib-import-path)
- [5. Variables globales del script](#5-variables-globales-del-script)
- [6. Funciones y métodos invocados](#6-funciones-y-metodos-invocados)
  - [6.1 Rutas y sistema de archivos](#61-rutas-y-sistema-de-archivos)
  - [6.2 OpenCV — carga y visualización](#62-opencv-carga-y-visualizacion)
  - [6.3 MediaPipe Tasks — configuración](#63-mediapipe-tasks-configuracion)
  - [6.4 MediaPipe — imagen de entrada a la red](#64-mediapipe-imagen-de-entrada-a-la-red)
  - [6.5 Inferencia](#65-inferencia)
  - [6.6 Conversión a protobuf](#66-conversion-a-protobuf)
  - [6.7 Dibujo de marcas (API Solutions)](#67-dibujo-de-marcas-api-solutions)
- [7. Los 21 landmarks de la mano](#7-los-21-landmarks-de-la-mano)
- [8. Estructuras de control del script](#8-estructuras-de-control-del-script)
  - [8.1 Validación de imagen cargada](#81-validacion-de-imagen-cargada)
  - [8.2 Comprobación de manos detectadas](#82-comprobacion-de-manos-detectadas)
  - [8.3 Bucle por cada mano](#83-bucle-por-cada-mano)
- [9. Dos APIs de MediaPipe en un solo flujo](#9-dos-apis-de-mediapipe-en-un-solo-flujo)
- [10. Archivo `hand_landmarker.task`](#10-archivo-hand_landmarkertask)
- [11. Cómo ejecutar la prueba y qué esperar](#11-como-ejecutar-la-prueba-y-que-esperar)
- [12. Resumen ejecutivo](#12-resumen-ejecutivo)
- [13. Referencia rápida del código fuente](#13-referencia-rapida-del-codigo-fuente)


---

<a id="1-objetivo-de-la-prueba"></a>

## 1. Objetivo de la prueba

El script `prueba.py` es una **prueba de concepto offline** (sin cámara en vivo):

1. Carga una imagen estática desde `assets/image2.png`.
2. Ejecuta un modelo de inteligencia artificial (**Hand Landmarker** de MediaPipe) que localiza hasta **2 manos**.
3. Obtiene **21 puntos clave por mano** (landmarks) en coordenadas normalizadas.
4. **Dibuja** sobre la misma imagen:
   - Círculos en cada punto (estilo por defecto de MediaPipe).
   - Líneas que conectan los puntos formando el “esqueleto” de la mano.
5. Muestra el resultado en una ventana de OpenCV titulada `"Result"`.

El resultado visual son las **marcas específicas**: puntos coloreados en articulaciones/yemas y segmentos que unen muñeca, dedos y nudillos según el mapa oficial `HAND_CONNECTIONS`.

---

<a id="2-archivos-necesarios-para-la-prueba"></a>

## 2. Archivos necesarios para la prueba

| Archivo | Ubicación | Rol |
|---------|-----------|-----|
| `prueba.py` | `prueba/` | Script principal |
| `hand_landmarker.task` | `prueba/` | Modelo TensorFlow Lite empaquetado para MediaPipe Tasks |
| `image2.png` | `assets/` (raíz del proyecto) | Imagen de entrada |
| Dependencias Python | `requirements.txt` (raíz) | `opencv-python`, `mediapipe`, etc. |

Si falta la imagen o el `.task`, el script fallará al cargar (`FileNotFoundError` o error del landmarker).

---

<a id="3-proceso-completo-pipeline"></a>

## 3. Proceso completo (pipeline)

**Orden temporal al ejecutar:**

1. Resolver rutas (`SCRIPT_DIR`, `PROJECT_ROOT`, `image_path`, `model_path`).
2. Leer imagen en memoria.
3. Crear opciones del modelo y abrir `HandLandmarker`.
4. Convertir imagen al formato que MediaPipe espera.
5. Inferencia (`detect`).
6. Si hay manos: por cada mano, convertir landmarks y dibujar.
7. Cerrar el landmarker (al salir del `with`).
8. Mostrar ventana hasta pulsar una tecla.

---

<a id="4-importaciones"></a>

## 4. Importaciones (qué es cada módulo y por qué se usa)

<a id="41-import-cv2"></a>

### 4.1 `import cv2`

- **Qué es:** OpenCV (Open Source Computer Vision Library), binding de Python `opencv-python`.
- **Uso en este script:**
  - `cv2.imread`: decodifica PNG/JPEG a un array NumPy.
  - `cv2.cvtColor`: cambia espacio de color BGR ↔ RGB.
  - `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows`: interfaz gráfica mínima para ver el resultado.

**Nota importante:** OpenCV usa por defecto el orden de canales **BGR** (azul, verde, rojo), no RGB. Eso afecta toda la cadena hasta la conversión explícita para MediaPipe.

---

<a id="42-import-mediapipe-as-mp"></a>

### 4.2 `import mediapipe as mp`

- **Qué es:** Paquete principal de Google MediaPipe para visión/percepción.
- **Uso en este script:**
  - `mp.Image` y `mp.ImageFormat.SRGB`: contenedor de imagen para la API **Tasks**.
  - `mp.solutions.drawing_utils`, `mp.solutions.hands`, `mp.solutions.drawing_styles`: API **legacy “Solutions”** usada **solo para dibujar**, no para detectar.

Aquí conviven **dos generaciones de API** de MediaPipe en el mismo archivo (detección moderna + dibujo clásico).

---

<a id="43-from-mediapipectasks-import-python"></a>

### 4.3 `from mediapipe.tasks import python`

- **Qué es:** Submódulo de configuración de la API **MediaPipe Tasks** (modelos `.task`).
- **Uso:** `python.BaseOptions` indica la ruta del archivo de modelo en disco.

No define funciones propias del script; expone clases de configuración.

---

<a id="44-from-mediapipectaskspython-import-vision"></a>

### 4.4 `from mediapipe.tasks.python import vision`

- **Qué es:** Tareas de visión por computadora (detección de manos, caras, poses, etc.) en modo Tasks.
- **Uso en este script:**
  - `vision.HandLandmarkerOptions`: parámetros del detector.
  - `vision.RunningMode`: modo IMAGE vs VIDEO vs LIVE_STREAM.
  - `vision.HandLandmarker`: clase del detector.
  - `HandLandmarker.create_from_options()`: factory que carga el `.task`.
  - `landmarker.detect()`: ejecuta inferencia en una imagen.

---

<a id="45-landmark_pb2"></a>

### 4.5 `from mediapipe.framework.formats import landmark_pb2`

- **Qué es:** Mensajes **Protocol Buffers** (`protobuf`) que describen listas de landmarks normalizados.
- **Uso:** Construir `NormalizedLandmarkList` y `NormalizedLandmark` para que `draw_landmarks` entienda la geometría.

La API Tasks devuelve landmarks como objetos Python simples; `draw_landmarks` fue diseñado para el formato protobuf de la API antigua.

---

<a id="46-from-pathlib-import-path"></a>

### 4.6 `from pathlib import Path`

- **Qué es:** Módulo estándar de Python para rutas orientadas a objetos (multiplataforma).
- **Uso:** `Path(__file__)`, concatenar con `/`, `.resolve()`, `.parent` sin strings frágiles con `\` o `/`.

---

<a id="5-variables-globales-del-script"></a>

## 5. Variables globales del script

| Variable | Tipo / valor típico | Significado detallado |
|----------|---------------------|------------------------|
| `SCRIPT_DIR` | `pathlib.Path` | Directorio absoluto donde está `prueba.py` (`.../GestureFlow/prueba`). Se obtiene con `Path(__file__).resolve().parent`. `__file__` es la ruta del script en ejecución; `.resolve()` elimina `..` y convierte a absoluta; `.parent` sube un nivel desde el archivo a la carpeta. |
| `PROJECT_ROOT` | `pathlib.Path` | Padre de `SCRIPT_DIR`: raíz del repo (`.../GestureFlow`). |
| `image_path` | `Path` | `PROJECT_ROOT / "assets" / "image2.png"`. Ruta completa a la imagen de prueba. |
| `model_path` | `Path` | `SCRIPT_DIR / "hand_landmarker.task"`. Modelo debe estar junto al script (o la ruta que definas). |
| `image` | `numpy.ndarray` uint8, shape `(H, W, 3)` | Píxeles en **BGR** tras `imread`. También es el **lienzo** donde se dibujan las marcas (se modifica in-place). |
| `base_options` | `mediapipe.tasks.python.BaseOptions` | Contenedor de opciones base; aquí solo se setea `model_asset_path`. |
| `options` | `vision.HandLandmarkerOptions` | Configuración completa del landmarker (modelo, modo, número de manos). |
| `landmarker` | `vision.HandLandmarker` | Instancia del modelo cargado en memoria/GPU según backend. |
| `mp_image` | `mediapipe.Image` | Vista de la imagen en **SRGB** para `detect`. |
| `results` | `HandLandmarkerResult` | Resultado de inferencia; campos usados: `hand_landmarks`. |
| `hand_landmarks` | lista de ~21 objetos landmark | Una mano detectada; cada elemento tiene `.x`, `.y`, `.z`. |
| `hand_landmarks_proto` | `landmark_pb2.NormalizedLandmarkList` | Misma mano en formato protobuf para dibujo. |
| `lm` | landmark individual | Uno de los 21 puntos en el bucle de comprensión de lista. |

---

<a id="6-funciones-y-metodos-invocados"></a>

## 6. Funciones y métodos invocados (referencia específica)

<a id="61-rutas-y-sistema-de-archivos"></a>

### 6.1 Rutas y sistema de archivos

#### `Path(__file__)`

- Crea un objeto `Path` apuntando al archivo `.py` ejecutado.

#### `.resolve()`

- Convierte a ruta absoluta y resuelve enlaces simbólicos.

#### `.parent`

- Directorio contenedor del archivo (no del proceso actual de trabajo).

#### `PROJECT_ROOT / "assets" / "image2.png"`

- Operador `/` de `pathlib` une segmentos de forma correcta en Windows y Linux.

#### `str(image_path)` / `str(model_path)`

- OpenCV y MediaPipe esperan `str` o rutas compatibles; `Path` se convierte explícitamente.

---

<a id="62-opencv-carga-y-visualizacion"></a>

### 6.2 OpenCV — carga y visualización

#### `cv2.imread(str(image_path))`

- **Entrada:** ruta al archivo de imagen.
- **Salida:** array NumPy 3D o `None` si falla (archivo inexistente, formato no soportado, permisos).
- **Formato de color:** BGR, 8 bits por canal.
- **Dimensiones:** alto × ancho × 3.

#### `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`

- Reordena canales: el pixel `(B,G,R)` pasa a `(R,G,B)`.
- **No** redimensiona la imagen; solo cambia interpretación de color.

#### `cv2.imshow("Result", image)`

- Crea o actualiza una ventana del sistema operativo con el contenido de `image`.
- `"Result"` es el título de la ventana.

#### `cv2.waitKey(0)`

- Bloquea el hilo hasta que el usuario pulse una tecla.
- El argumento `0` = espera indefinida (milisegundos si fuera > 0).

#### `cv2.destroyAllWindows()`

- Cierra todas las ventanas creadas por `imshow` en este proceso.

---

<a id="63-mediapipe-tasks-configuracion"></a>

### 6.3 MediaPipe Tasks — configuración

#### `python.BaseOptions(model_asset_path=...)`

- **Parámetro clave:** `model_asset_path` → ruta al archivo `.task` (modelo empaquetado).
- Otros parámetros existen (delegates, GPU) pero no se usan en esta prueba.

#### `vision.HandLandmarkerOptions(...)`

| Parámetro | Valor en prueba | Efecto |
|-----------|-----------------|--------|
| `base_options` | `base_options` | Enlaza el modelo. |
| `running_mode` | `vision.RunningMode.IMAGE` | Optimizado para un frame aislado; no mantiene estado entre frames de video. Alternativas: `VIDEO` (secuencia con timestamps), `LIVE_STREAM` (callback asíncrono). |
| `num_hands` | `2` | Máximo de manos a devolver. Si hay 3, las dos con mayor score típicamente. |

Otros parámetros opcionales no usados: `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence`.

#### `vision.HandLandmarker.create_from_options(options)`

- Carga el modelo TFLite desde disco.
- Devuelve un objeto landmarker listo para `detect`.

#### Context manager `with ... as landmarker:`

- Al salir del bloque `with`, se liberan recursos nativos del modelo (memoria, handles).

---

<a id="64-mediapipe-imagen-de-entrada-a-la-red"></a>

### 6.4 MediaPipe — imagen de entrada a la red

#### `mp.Image(image_format=mp.ImageFormat.SRGB, data=...)`

- **`image_format`:** declara que `data` está en RGB (tres canales).
- **`data`:** array NumPy contiguo H×W×3, dtype habitual uint8.
- MediaPipe puede validar dimensiones y tipo; datos incorrectos provocan error en `detect`.

---

<a id="65-inferencia"></a>

### 6.5 Inferencia

#### `landmarker.detect(mp_image)`

- **Entrada:** un `mp.Image` en modo IMAGE.
- **Salida:** objeto resultado (p. ej. `HandLandmarkerResult`) con campos entre otros:
  - `hand_landmarks`: lista de listas de landmarks normalizados.
  - También puede incluir `hand_world_landmarks`, handedness, etc. (no usados en este script).

**Coordenadas normalizadas:**

- `x`, `y` ∈ [0, 1] relativo al ancho y alto de la imagen (origen suele ser esquina superior izquierda).
- `z` profundidad relativa (más pequeño = más cerca de la cámara en el modelo; escala aproximada).

---

<a id="66-conversion-a-protobuf"></a>

### 6.6 Conversión a protobuf (puente entre APIs)

#### `landmark_pb2.NormalizedLandmarkList()`

- Crea lista vacía de landmarks en formato protobuf.

#### `landmark_pb2.NormalizedLandmark(x=..., y=..., z=...)`

- Un punto 3D normalizado en el mensaje protobuf.

#### `hand_landmarks_proto.landmark.extend([...])`

- Añade los 21 elementos a la lista interna del mensaje.
- List comprehension recorre `hand_landmarks` de la API Tasks y copia x, y, z.

**Por qué es necesario:** `mp.solutions.drawing_utils.draw_landmarks` fue escrito para la API `mediapipe.solutions.hands` que exponía `results.multi_hand_landmarks` ya como `NormalizedLandmarkList`. Tasks no devuelve ese tipo directamente.

---

<a id="67-dibujo-de-marcas-api-solutions"></a>

### 6.7 Dibujo de marcas (API Solutions)

#### `mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks_proto, HAND_CONNECTIONS, style_landmarks, style_connections)`

| Argumento | Rol |
|-----------|-----|
| `image` | Matriz BGR donde se pinta (modificación in-place). |
| `hand_landmarks_proto` | Geometría de la mano. |
| `mp.solutions.hands.HAND_CONNECTIONS` | Tuplas `(índice_a, índice_b)` que definen qué landmarks se unen con línea. |
| `get_default_hand_landmarks_style()` | Diccionario/estilo: color y radio de cada punto. |
| `get_default_hand_connections_style()` | Estilo de líneas (color, grosor). |

**Qué dibuja exactamente:**

1. Para cada landmark, un círculo/relleno en la posición pixel `(x * ancho, y * alto)`.
2. Para cada par en `HAND_CONNECTIONS`, un segmento entre esos dos índices.

Las marcas “específicas” son el **conjunto estándar de 21 puntos + topología de conexiones** definida por MediaPipe Hands, no puntos arbitrarios.

---

<a id="7-los-21-landmarks-de-la-mano"></a>

## 7. Los 21 landmarks de la mano (índices)

Cada mano tiene **exactamente 21 puntos** con índices fijos:

| Índice | Nombre habitual | Ubicación |
|--------|-----------------|-----------|
| 0 | WRIST | Muñeca |
| 1–4 | THUMB | Pulgar (base → punta) |
| 5–8 | INDEX | Índice |
| 9–12 | MIDDLE | Medio |
| 13–16 | RING | Anular |
| 17–20 | PINKY | Meñique |

Cada dedo usa 4 puntos: base (CMC/MCP según dedo), articulaciones intermedias y punta (TIP).

`HAND_CONNECTIONS` une, por ejemplo, muñeca con bases de dedos, y a lo largo de cada dedo de proximal a distal, formando el esqueleto visible en la prueba.

---

<a id="8-estructuras-de-control-del-script"></a>

## 8. Estructuras de control del script

<a id="81-validacion-de-imagen-cargada"></a>

### `if image is None: raise FileNotFoundError(...)`

- Falla rápido con mensaje claro si la ruta o el archivo es inválido.

<a id="82-comprobacion-de-manos-detectadas"></a>

### `if results.hand_landmarks:`

- Solo dibuja si el modelo encontró al menos una mano.
- Si la imagen no tiene manos visibles, se muestra la foto original sin marcas.

<a id="83-bucle-por-cada-mano"></a>

### `for hand_landmarks in results.hand_landmarks:`

- Itera cada mano (0, 1 o 2 entradas según detección y `num_hands`).
- Cada iteración dibuja un esqueleto completo encima de la misma `image`.

---

<a id="9-dos-apis-de-mediapipe-en-un-solo-flujo"></a>

## 9. Dos APIs de MediaPipe en un solo flujo

| Aspecto | API Tasks (`HandLandmarker`) | API Solutions (`mp.solutions.*`) |
|---------|------------------------------|----------------------------------|
| Rol aquí | **Detección** | **Solo dibujo** |
| Modelo | Archivo `.task` externo | Antes iba embebido en el paquete |
| Salida landmarks | Objetos con `.x`, `.y`, `.z` | Protobuf `NormalizedLandmarkList` |
| Estado | Recomendada por Google para nuevos proyectos | Legacy pero útil para `draw_landmarks` |

La prueba **no** usa `mp.solutions.hands.Hands()` para detectar; solo reutiliza sus constantes de conexión y estilos de dibujo.

---

<a id="10-archivo-hand_landmarkertask"></a>

## 10. Archivo `hand_landmarker.task`

- Binario que contiene el modelo **Hand Landmarker** (red neuronal convertida a TensorFlow Lite).
- MediaPipe lo carga mediante `model_asset_path`.
- Debe coincidir la versión con la del paquete `mediapipe` instalado (en el proyecto: `mediapipe==0.10.18`).
- Si el archivo está corrupto o es de otra versión, `create_from_options` o `detect` pueden fallar.

---

<a id="11-como-ejecutar-la-prueba-y-que-esperar"></a>

## 11. Cómo ejecutar la prueba y qué esperar

Desde la raíz del proyecto (con entorno virtual activado y dependencias instaladas):

```bash
python prueba/prueba.py
```

**Resultado esperado:**

1. Se abre una ventana **"Result"**.
2. Sobre `image2.png` aparecen **puntos** en articulaciones y **líneas** entre ellos por cada mano detectada.
3. Al pulsar cualquier tecla en la ventana enfocada, el script termina y cierra ventanas.

**Si no ves marcas:**

- No se detectaron manos (`results.hand_landmarks` vacío).
- Manos fuera de cuadro, muy pequeñas, borrosas o ocultas.
- Iluminación o postura dificultan el modelo.

---

<a id="12-resumen-ejecutivo"></a>

## 12. Resumen ejecutivo

| Paso | Acción | Resultado |
|------|--------|-----------|
| 1 | Resolver rutas con `pathlib` | Saber dónde están imagen y modelo |
| 2 | `cv2.imread` | Matriz BGR en RAM |
| 3 | Configurar `HandLandmarkerOptions` + crear landmarker | Modelo listo |
| 4 | BGR→RGB + `mp.Image` | Entrada válida para MediaPipe |
| 5 | `detect` | 0–2 manos × 21 landmarks normalizados |
| 6 | Protobuf + `draw_landmarks` | Imagen BGR anotada con esqueleto de mano |
| 7 | `imshow` / `waitKey` | Visualización para validar la prueba |

Este flujo es la base para evolucionar GestureFlow: el mismo landmarker y landmarks pueden alimentar **reconocimiento de gestos** (comparar posiciones de dedos, ángulos, distancias) en lugar de solo dibujar.

---

<a id="13-referencia-rapida-del-codigo-fuente"></a>

## 13. Referencia rápida del código fuente

El script completo vive en: `prueba/prueba.py` (69 líneas).

Documentación oficial útil:

- [MediaPipe Hand Landmarker (Tasks)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
