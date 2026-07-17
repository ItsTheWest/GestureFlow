# Referencia Común — Glosario y Conceptos Compartidos de GestureFlow

Este documento centraliza los conceptos técnicos, funciones de OpenCV, configuraciones de MediaPipe, procesamiento de datos y fundamentos de redes neuronales que se repiten a lo largo del proyecto `GestureFlow` (pasos 01 al 06). 

---

## Índice

- [1. Rutas y Entorno](#1-rutas-y-entorno)
- [2. OpenCV y Captura de Vídeo](#2-opencv-y-captura-de-vídeo)
- [3. MediaPipe y Detección de Manos](#3-mediapipe-y-detección-de-manos)
- [4. Procesamiento de Landmarks y Estructuras de Datos](#4-procesamiento-de-landmarks-y-estructuras-de-datos)
- [5. Entrenamiento LSTM y Redes Neuronales](#5-entrenamiento-lstm-y-redes-neuronales)
- [6. Tabla de Errores Frecuentes Unificada](#6-tabla-de-errores-frecuentes-unificada)

---

## 1. Rutas y Entorno

### 1.1 Entorno Virtual (`venv`)
- **Explicación**: Aísla las dependencias del proyecto (como OpenCV, MediaPipe y TensorFlow) de la instalación de Python global del sistema operativo, garantizando la reproducibilidad y evitando conflictos de versiones.
- **Uso en el repositorio**:
  - Utilizado de manera transversal para la ejecución de cualquier script del proyecto.

### 1.2 Rutas con Path (`pathlib.Path`)
- **Explicación**: El módulo `pathlib` de Python proporciona una interfaz orientada a objetos para interactuar con el sistema de archivos de forma multiplataforma. Evita fallos comunes al unir carpetas con barras inclinadas distintas (`/` o `\`) según el sistema operativo.
- **Términos Clave**:
  - `SCRIPT_DIR`: Directorio absoluto de la carpeta que contiene al script que se está ejecutando (`Path(__file__).resolve().parent`).
  - `PROJECT_ROOT`: Directorio raíz del proyecto completo (subiendo dos niveles desde la carpeta del script).
  - `MODEL_PATH`: Ruta absoluta donde se encuentra el archivo de modelo `.task`.
- **Uso en el código**:
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L243-L245) (Líneas 243-245)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L252-L254) (Líneas 252-254)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L14-L16) (Líneas 14-16)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L14-L16) (Líneas 14-16)

---

## 2. OpenCV y Captura de Vídeo

### 2.1 Captura de Cámara (`cv2.VideoCapture`)
- **Explicación**: Inicializa el hardware de captura de vídeo (webcam). El parámetro `0` representa la cámara por defecto del sistema. Se valida la correcta inicialización mediante el método `isOpened()`. Cada iteración lee un frame individual usando `read()`, el cual devuelve una tupla `(ret, frame)` donde `ret` es un booleano indicando el éxito de la lectura.
- **Uso en el código**:
  - [paso_01_camara.py](../pasos/paso-01-camara/paso_01_camara.py#L311-L315) (Líneas 311-315)
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L268-L271) (Líneas 268-271)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L283-L286) (Líneas 283-286)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L112-L115) (Líneas 112-115)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L292-L295) (Líneas 292-295)

### 2.2 Volteo Horizontal (`cv2.flip`)
- **Explicación**: Invierte las columnas de la matriz de imagen. El parámetro `1` indica que se realiza un volteo horizontal (efecto espejo). Esto facilita la interacción con el usuario en pantalla, ya que el movimiento reflejado corresponde al movimiento físico directo.
- **Uso en el código**:
  - [paso_01_camara.py](../pasos/paso-01-camara/paso_01_camara.py#L327) (Línea 327)
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L288) (Línea 288)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L306) (Línea 306)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L144) (Línea 144)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L314) (Línea 314)

### 2.3 Refresco y Detección de Teclado (`cv2.waitKey`)
- **Explicación**: Detiene la ejecución del hilo durante los milisegundos especificados para actualizar y redibujar la ventana gráfica. Retorna el código ASCII de la tecla presionada. La operación a nivel de bits `& 0xFF` extrae el byte relevante para compararlo con el valor de teclas estándar como `'q'` o `' '` (espacio). 
  - `cv2.waitKey(1)`: Actualiza la ventana inmediatamente y espera 1 ms (ideal para bucles de vídeo en tiempo real).
  - `cv2.waitKey(0)`: Pausa indefinidamente la ejecución en ese frame hasta que el usuario presione cualquier tecla.
- **Uso en el código**:
  - [paso_01_camara.py](../pasos/paso-01-camara/paso_01_camara.py#L342-L343) (Líneas 342-343)
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L296-L298) (Líneas 296-298 y 314)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L331-L332) (Líneas 331-332)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L178-L179) (Líneas 178-179 y 193-194)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L318-L325) (Líneas 318-325 y 404-407)

### 2.4 Elementos Gráficos e Interfaz (HUD)
- **Explicación**: OpenCV permite dibujar directamente sobre los fotogramas (matrices BGR in-place). Se utilizan:
  - `cv2.putText`: Dibuja cadenas de texto sobre la imagen especificando coordenadas, tipografía, tamaño y color.
  - `cv2.rectangle`: Dibuja un rectángulo (puede estar relleno con `-1`).
  - `cv2.line`: Dibuja un segmento de línea entre dos puntos.
  - `cv2.addWeighted`: Realiza una mezcla lineal de dos imágenes, útil para overlays semitransparentes en cuentas atrás.
- **Uso en el código**:
  - [paso_01_camara.py](../pasos/paso-01-camara/paso_01_camara.py#L336-L339) (Líneas 336-339)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L85-L236) (Líneas 85-236, funciones `draw_waiting`, `draw_countdown`, `draw_hud`)

### 2.5 Liberación de Recursos
- **Explicación**: Cierra los descriptores de hardware y la cámara con `cap.release()`, y destruye las ventanas de la interfaz gráfica con `cv2.destroyAllWindows()`. Si no se ejecutan estos pasos al finalizar (por ejemplo, ante un crash del programa), el hardware de la webcam puede quedar bloqueado por el sistema operativo, requiriendo reiniciar el script o el entorno de ejecución.
- **Uso en el código**:
  - Implementado al final de todos los scripts ejecutables en el proyecto.

---

## 3. MediaPipe y Detección de Manos

### 3.1 Carga del Modelo (`HandLandmarkerOptions` y `BaseOptions`)
- **Explicación**: Define la configuración para instanciar la red neuronal de MediaPipe de detección de marcas de manos (`vision.HandLandmarker`). Utiliza `BaseOptions` para cargar el archivo binario del modelo pre-entrenado `.task` y configurar parámetros como el número máximo de manos a buscar (`num_hands=2`).
- **Uso en el código**:
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L273-L280) (Líneas 273-280)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L288-L294) (Líneas 288-294)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L124-L130) (Líneas 124-130)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L40-L52) (Líneas 40-52)

### 3.2 Espacio de Color BGR a RGB (`mp.Image`)
- **Explicación**: OpenCV captura y decodifica las imágenes en formato **BGR** (Blue, Green, Red). Sin embargo, los modelos de MediaPipe Tasks requieren que los canales estén en formato **RGB** (Red, Green, Blue) y encapsulados en el objeto de datos `mp.Image`.
- **Uso en el código**:
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L302-L305) (Líneas 302-305)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L323-L326) (Líneas 323-326)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L187-L190) (Líneas 187-190)
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L366-L367) (Líneas 366-367)

### 3.3 Modos de Inferencia (`RunningMode`)
- **Explicación**: MediaPipe Tasks soporta diferentes modos de ejecución adaptados al caso de uso:
  - `IMAGE`: Modo de inferencia síncrona. La llamada a `detect()` bloquea el hilo principal y devuelve el resultado de inmediato. Se utiliza en imágenes aisladas o grabaciones pausadas.
  - `LIVE_STREAM`: Modo de inferencia asíncrona optimizado para webcams. La llamada a `detect_async()` envía el frame en un hilo secundario y continúa inmediatamente. El resultado se devuelve de forma asíncrona a través de una función de callback (`result_callback`), la cual requiere que se le proporcione un timestamp en milisegundos que sea estrictamente creciente.
- **Uso en el código**:
  - **IMAGE**: [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L276) (Línea 276) y [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L48) (Línea 48).
  - **LIVE_STREAM**: [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L291) (Línea 291) y [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L127) (Línea 127).

### 3.4 Control de Flujo de Inferencia Asíncrona
- **Explicación**: Cuando se ejecuta en modo `LIVE_STREAM` en hardware de CPU limitada, la tasa de fotogramas de la cámara puede superar la velocidad de procesamiento del modelo. Para evitar encolar frames antiguos y sufrir de latencia acumulativa, se utiliza:
  - `listo_para_inferir` (bandera booleana): Controla que solo se envíe un nuevo fotograma a `detect_async` cuando el callback del proceso anterior ha terminado de ejecutarse.
  - `ANCHO_INFERENCIA` / `cv2.resize`: Redimensiona el fotograma temporalmente a un ancho menor (p. ej., `320` píxeles) antes de enviarlo al modelo, acelerando drásticamente el cómputo. Dado que los landmarks resultantes son normalizados (`0.0` a `1.0`), estos se pueden dibujar de vuelta con precisión matemática en la resolución completa original.
- **Uso en el código**:
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L18-L22) (Líneas 18-22 y 184-191)

---

## 4. Procesamiento de Landmarks y Estructuras de Datos

### 4.1 Dibujo de Landmarks (`dibujar_manos`)
- **Explicación**: Toma la lista de coordenadas normalizadas `results.hand_landmarks` y las pinta sobre la matriz BGR in-place. Para poder utilizar las librerías clásicas de trazado (`mp.solutions.drawing_utils.draw_landmarks`), es necesario convertir la lista de la API moderna de Tasks al formato anterior basado en Protocol Buffers (protobuf) mediante `NormalizedLandmarkList`.
- **Uso en el código**:
  - [paso_02_dibujo.py](../pasos/paso-02-dibujo/paso_02_dibujo.py#L247-L263) (Líneas 247-263)
  - [paso_03_tiempo_real.py](../pasos/paso-03-tiempo-real/paso_03_tiempo_real.py#L262-L278) (Líneas 262-278)
  - [paso_04_vocales.py](../pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py#L36-L49) (Líneas 36-49)

### 4.2 Los 21 Landmarks de la Mano
- **Explicación**: Cada mano detectada cuenta con un conjunto estructurado y ordenado de exactamente 21 puntos anatómicos (landmarks). Cada landmark almacena coordenadas `x` (horizontal), `y` (vertical) y `z` (profundidad relativa).

```text
       8   12  16  20      Tip de los dedos:
       |   |   |   |       4: Pulgar (THUMB_TIP)
       7   11  15  19      8: Índice (INDEX_FINGER_TIP)
       |   |   |   |       12: Medio (MIDDLE_FINGER_TIP)
   4   6   10  14  18      16: Anular (RING_FINGER_TIP)
    \  |   |   |   |       20: Meñique (PINKY_TIP)
     3 5---9---13--17
      \|            /
       2           /
        \         /
         1       /
          \     /
           0---/           0: Muñeca (WRIST)
```

### 4.3 Extracción y Relleno de Características (`extract_keypoints`)
- **Explicación**: Convierte la salida de detección del landmarker en un vector aplanado NumPy de forma unidimensional `(126,)` conteniendo floats de 32 bits:
  - **Mano izquierda**: 21 landmarks × 3 coordenadas (x, y, z) = Primeros 63 elementos (índices `0` a `62`).
  - **Mano derecha**: 21 landmarks × 3 coordenadas (x, y, z) = Siguientes 63 elementos (índices `63` a `125`).
  - **Relleno con ceros (Padding)**: Si una de las dos manos (o ambas) no es visible en pantalla, su slot correspondiente (63 floats) se rellena automáticamente con ceros (`np.zeros`). Esto previene errores de dimensión en la entrada de la red LSTM y enseña al modelo a reconocer la ausencia de extremidades.
- **Uso en el código**:
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L58-L79) (Líneas 58-79)

### 4.4 Formato `.npy` y Secuencias Temporales
- **Explicación**: El formato binario `.npy` de NumPy se utiliza para guardar arrays de forma directa en disco conservando su dimensionalidad y tipos.
  - Para alimentar redes LSTM y capturar gestos dinámicos (que involucran movimiento temporal en lugar de una foto estática), se definen **secuencias**.
  - Cada secuencia representa una grabación consecutiva de fotogramas, guardándose con una dimensión bidimensional `(SEQUENCE_LENGTH, NUM_FEATURES)` = `(30, 126)`.
- **Uso en el código**:
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L380-L384) (Líneas 380-384)

### 4.5 Solapamiento Temporal (Aumento de Datos)
- **Explicación**: El parámetro `SAVE_EVERY=15` en relación con una longitud de secuencia `SEQUENCE_LENGTH=30` produce que las grabaciones consecutivas compartan un 50% de sus datos temporales (15 fotogramas). Es una técnica altamente eficiente de aumento de datos (data augmentation) en series de tiempo para entrenar modelos más robustos a menor coste de recopilación física.
- **Uso en el código**:
  - [paso_05_recoleccion.py](../pasos/paso-05-recoleccion/paso_05_recoleccion.py#L21-L28) (Líneas 21-28 y 380)

---

## 5. Entrenamiento LSTM y Redes Neuronales

### 5.1 Redes LSTM (Long Short-Term Memory)
- **Explicación**: Tipo de capa en Redes Neuronales Recurrentes (RNN). Cuentan con un estado de celda interno (celda de memoria) que les permite retener o descartar información a lo largo de pasos temporales. Esto resulta fundamental para clasificar gestos del mundo real (por ejemplo, diferenciar entre "saludar" e "ir a la izquierda"), donde el orden y la trayectoria del movimiento de la mano en los 30 frames consecutivos determinan el significado.

### 5.2 Estructura del Tensor de Entrada
- **Explicación**: El entrenamiento de una red de aprendizaje profundo exige estructurar las secuencias cargadas en un tensor de tres dimensiones (3D): `(batch_size, SEQUENCE_LENGTH, NUM_FEATURES)` = `(N, 30, 126)`.
  - `batch_size` (`N`): Número total de secuencias de gestos que se pasan al optimizador en un lote.
  - `SEQUENCE_LENGTH` (`30`): Pasos de tiempo en el eje temporal.
  - `NUM_FEATURES` (`126`): Número de variables observadas en cada paso.

### 5.3 Categorización de Etiquetas (One-Hot Encoding)
- **Explicación**: El modelo no puede predecir directamente strings textuales. Se asocia un índice a cada carpeta de gesto (p. ej., `saludar=0`, `traer=1`), y posteriormente con la función `to_categorical` de Keras se convierte cada etiqueta en un vector de probabilidades binario (ej. `[1, 0]` para clase 0 en un sistema binario), lo cual es el formato estándar que exige la función de pérdida de clasificación.
- **Uso en el código**:
  - Documentado en [paso_06_doc.md](../pasos/paso-06-entrenamiento/paso_06_doc.md) (Líneas 102-110, y se implementa en [paso_06_recolecion.py](../pasos/paso-06-entrenamiento/paso_06_recolecion.py)).

### 5.4 Hiperparámetros de Compilación
- **Optimizador (`Adam`)**: Algoritmo de descenso de gradiente estocástico basado en la estimación adaptativa de momentos de primer y segundo orden.
- **Función de pérdida (`categorical_crossentropy`)**: Mide el error entre la distribución de probabilidad predicha por la capa final `Softmax` de la red y el vector `One-hot` real.
- **Métrica (`accuracy`)**: Porcentaje de predicciones correctas sobre el total del dataset de validación.

---

## 6. Tabla de Errores Frecuentes Unificada

Esta tabla resume las causas raíz de los fallos más recurrentes del proyecto completo y cómo diagnosticarlos:

| Síntoma de Fallo | Causa Raíz Probable | Solución / Qué Inspeccionar |
|------------------|---------------------|-----------------------------|
| `FileNotFoundError` al iniciar | El archivo del modelo `.task` no se encuentra en la ruta esperada. | Comprobar que `hand_landmarker.task` esté en la carpeta `prueba/`. Verificar variables de rutas. |
| La ventana gráfica se abre en negro o se congela | Bloqueo en el hilo de captura o falta del disparador de refresco en la ventana. | Asegurarse de que `cv2.waitKey(1)` se ejecute dentro del bucle de cámara. Probar con índice de cámara `1` o `2` en `cv2.VideoCapture`. |
| No se dibuja ningún esqueleto en pantalla | Fallo en la detección de landmarks o inconsistencia del color. | La mano debe estar totalmente visible en el encuadre. Verificar que el frame se convierta de BGR a RGB antes de enviarse al detector. |
| La cámara se queda bloqueada tras forzar la parada del script | El script terminó abruptamente sin ejecutar las funciones de liberación de hardware. | Ejecutar siempre la rutina de salida en bloque `finally`: `cap.release()` y `cv2.destroyAllWindows()`. |
| Latencia severa o desfase temporal en el dibujo de la mano | Acumulación de frames en el hilo de inferencia asíncrona de MediaPipe. | Utilizar el control por bandera `listo_para_inferir` para descartar frames si la CPU está saturada. Disminuir el ancho del frame de inferencia. |
| `ValueError` en el ajuste del modelo (`model.fit`) | Inconsistencia en la forma de las características leídas de los archivos. | Verificar que todos los archivos `.npy` dentro del directorio `gestos/` tengan exactamente las dimensiones `(30, 126)`. |
