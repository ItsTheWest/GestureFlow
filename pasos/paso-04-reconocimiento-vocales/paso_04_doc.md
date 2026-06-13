# Documentación: Paso 04 — Reconocimiento de Vocales (`paso_04_vocales.py`)

A partir de la detección asíncrona en tiempo real (Paso 03), este paso procesa algebraicamente las coordenadas de los landmarks para clasificar de forma estática las cinco vocales en español ('A', 'E', 'I', 'O', 'U') para **ambas manos** de forma independiente.

Para conocer en detalle los conceptos de rutas, cámara, inferencia asíncrona, control de colas y dibujo del esqueleto, consulta la [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## Índice

- [1. Objetivo del paso](#1-objetivo-del-paso)
- [2. Archivos de esta carpeta](#2-archivos-de-esta-carpeta)
- [3. Pipeline](#3-pipeline)
- [4. Lógica de Clasificación de Vocales](#4-lógica-de-clasificación-de-vocales)
- [5. Validación Temporal por Mano](#5-validación-temporal-por-mano)
- [6. OpenCV, teclas y ventana](#6-opencv-teclas-y-ventana)
- [7. Cómo ejecutar](#7-cómo-ejecutar)
- [8. Errores frecuentes](#8-errores-frecuentes)
- [9. ¿Qué sigue después?](#9-qué-sigue-después)
- [10. Referencia del código fuente](#10-referencia-del-código-fuente)

---

## 1. Objetivo del paso

**Objetivo:** procesar los landmarks de **ambas manos** detectadas en tiempo real para determinar si corresponden a la forma de una vocal en lengua de señas de manera independiente, validando temporalmente que el gesto se mantenga estable en cada una antes de confirmarlo.

| Funcionalidad Incluida | Origen Común / Referencia |
|-------------------------|--------------------------|
| Clasificación geométrica de dedos abiertos/cerrados | Exclusivo de este paso |
| Algoritmo de distancia para la letra 'O' | Exclusivo de este paso |
| Validación temporal independiente por mano (`Left`/`Right`) | Exclusivo de este paso |
| Evitar superposición en el HUD con offset vertical | Exclusivo de este paso |
| Control de colas (`listo_para_inferir` y `ANCHO_INFERENCIA`) | [REF §3.4](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#34-control-de-flujo-de-inferencia-asíncrona) |
| Inferencia asíncrona en tiempo real (`LIVE_STREAM`) | [REF §3.3](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#33-modos-de-inferencia-runningmode) |

**Criterio de éxito:**
- Al gesticular una vocal con la mano izquierda, derecha o ambas, el sistema muestra `"Mano Izquierda - Validando X..."` o `"Mano Derecha - Validando Y..."` en el HUD.
- Si se mantiene el gesto durante 1.0 segundo, el HUD muestra `"Mano [Lado] - Confirmada: X"` (rojo gigante).
- Si una mano sale de la imagen o cambia el gesto, su estado se reinicia de manera independiente, sin alterar la validación de la otra mano.

---

## 2. Archivos de esta carpeta

| Archivo | Rol |
|---------|-----|
| [paso_04_vocales.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py) | Script de clasificación analítica de vocales |
| [paso_04_doc.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-04-reconocimiento-vocales/paso_04_doc.md) | Esta documentación |

**Glosario y conceptos comunes:** [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## 3. Pipeline

```text
1. Cargar HandLandmarker (LIVE_STREAM, ANCHO_INFERENCIA=320)
2. Inicializar estado_manos para Left y Right
3. while True:
4.      read → flip → display = copia
5.      si listo_para_inferir:
6.          BGR → RGB → detect_async()
7.      si ultimo_resultado tiene landmarks:
8.          dibujar_manos()
9.          por cada mano detectada (idx):
10.             lado = handedness[idx].category_name
11.             vocal = get_vowel(landmarks)
12.             actualizar estado_manos[lado] (tiempo e inicio)
13.         resetear estado de las manos ausentes en este frame
14.         dibujar en display estado_manos con offset vertical dinámico
15.      imshow(display)
16.      waitKey(1) → si 'q', break
17. release + destroyAllWindows
```

---

## 4. Lógica de Clasificación de Vocales

La función `get_vowel` determina el estado de la mano basándose en la posición de sus 21 landmarks en el eje `y`. Ver mapa en [REF §4.2](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#42-los-21-landmarks-de-la-mano).

### Reglas algebraicas para cada vocal:
- **Vocal 'A'**: Dedos índice, medio, anular y meñique **cerrados** (yema debajo de la articulación PIP). Pulgar apuntando hacia arriba y al lateral exterior de la mano.
- **Vocal 'E'**: Dedos índice, medio, anular y meñique **cerrados**. Pulgar hacia abajo plegado frente a los dedos.
- **Vocal 'I'**: Dedos índice, medio y anular **cerrados**. Dedo meñique **abierto** (hacia arriba). Pulgar arriba.
- **Vocal 'U'**: Dedos índice y medio **abiertos**. Dedos anular y meñique **cerrados**. Pulgar abajo.
- **Vocal 'O'**: Distancia euclidiana entre las yemas del pulgar y el índice menor a `0.05`, y entre las yemas del pulgar y el medio menor a `0.05`. Dedos anular y meñique semi-cerrados.

---

## 5. Validación Temporal por Mano

Para admitir múltiples manos simultáneamente sin interferencias de variables:
- Se implementa un diccionario global `estado_manos` con llaves `"Left"` y `"Right"`.
- Cada llave contiene su propio `"vocal_detectada"`, `"tiempo_inicio"` y estado de `"confirmada"`.
- Cada frame asocia y actualiza el estado de las manos detectadas. Si una de las manos desaparece del cuadro de captura, el sistema detecta que no está en el conjunto de manos presentes y limpia su estado de validación de inmediato.

---

## 6. OpenCV, teclas y ventana

- **Q**: Cierra el programa.
- **cv2.WINDOW_GUI_NORMAL**: Elimina barras de menú de Qt del frame.
- **Offset Vertical Dinámico**: Para evitar superposición, el HUD del estado de la mano izquierda se posiciona en `y = 80` y el de la mano derecha en `y = 120`.

---

## 7. Cómo ejecutar

Desde la raíz del proyecto, con el entorno `venv` activado:

```bash
python pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py
```

---

## 8. Errores frecuentes

Si notas fallas comunes de la webcam o del modelo, consulta la **Tabla de Errores Frecuentes Unificada** en la [Sección 6 de REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).

---

## 9. ¿Qué sigue después?

Has aprendido a clasificar gestos **estáticos** de múltiples manos mediante reglas heurísticas directas.

**Siguiente etapa**:
- [Paso 05 — Recolección](file:///home/thewest/proyectos/GestureFlow/pasos/paso-05-recoleccion/paso_05_doc.md): Grabar secuencias continuas de landmarks en archivos `.npy`.
- [Paso 06 — Entrenamiento](file:///home/thewest/proyectos/GestureFlow/pasos/paso-06-entrenamiento/paso_06_doc.md): Alimentar una red neuronal recurrente LSTM para clasificar movimientos complejos.

---

## 10. Referencia del código fuente

El script completo vive en [paso_04_vocales.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py).
