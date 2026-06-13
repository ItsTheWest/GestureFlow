# Documentación: Paso 04 — Reconocimiento de Vocales (`paso_04_vocales.py`)

A partir de la detección asíncrona en tiempo real (Paso 03), este paso procesa algebraicamente las coordenadas de los landmarks para clasificar de forma estática las cinco vocales en español ('A', 'E', 'I', 'O', 'U').

Para conocer en detalle los conceptos de rutas, cámara, inferencia asíncrona, control de colas y dibujo del esqueleto, consulta la [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## Índice

- [1. Objetivo del paso](#1-objetivo-del-paso)
- [2. Archivos de esta carpeta](#2-archivos-de-esta-carpeta)
- [3. Pipeline](#3-pipeline)
- [4. Lógica de Clasificación de Vocales](#4-lógica-de-clasificación-de-vocales)
- [5. Validación Temporal](#5-validación-temporal)
- [6. OpenCV, teclas y ventana](#6-opencv-teclas-y-ventana)
- [7. Cómo ejecutar](#7-cómo-ejecutar)
- [8. Errores frecuentes](#8-errores-frecuentes)
- [9. ¿Qué sigue después?](#9-qué-sigue-después)
- [10. Referencia del código fuente](#10-referencia-del-código-fuente)

---

## 1. Objetivo del paso

**Objetivo:** procesar los landmarks de la mano detectada en tiempo real para determinar si corresponden a la forma de una vocal en lengua de señas, validando que el gesto se mantenga de forma estable antes de confirmarlo en pantalla.

| Funcionalidad Incluida | Origen Común / Referencia |
|-------------------------|--------------------------|
| Clasificación geométrica de dedos abiertos/cerrados | Exclusivo de este paso |
| Algoritmo de distancia para la letra 'O' | Exclusivo de este paso |
| Validación temporal (`TIEMPO_CONFIRMACION = 1.0s`) | Exclusivo de este paso |
| Control de colas (`listo_para_inferir` y `ANCHO_INFERENCIA`) | [REF §3.4](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#34-control-de-flujo-de-inferencia-asíncrona) |
| Inferencia asíncrona en tiempo real (`LIVE_STREAM`) | [REF §3.3](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#33-modos-de-inferencia-runningmode) |

**Criterio de éxito:**
- Al gesticular una vocal ('A', 'E', 'I', 'O', 'U') de forma estática frente a la cámara, el sistema muestra `"Validando X..."`.
- Si se mantiene el gesto durante 1.0 segundo, el HUD muestra `"Vocal Confirmada: X"` en letras rojas grandes.
- Si se quita la mano o cambia de gesto, la confirmación se cancela de inmediato.

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
2. while True:
3.      read → flip → display = copia
4.      si listo_para_inferir:
5.          BGR → RGB → detect_async()
6.      si ultimo_resultado tiene landmarks:
7.          dibujar_manos()
8.          vocal = get_vowel(mano_0)
9.          si vocal == vocal_detectada:
10.             si tiempo_transcurrido >= 1.0s:
11.                 putText("Vocal Confirmada")
12.             sino:
13.                 putText("Validando...")
14.         sino:
15.             vocal_detectada = vocal, reiniciar tiempo
16.      imshow(display)
17.      waitKey(1) → si 'q', break
18. release + destroyAllWindows
```

---

## 4. Lógica de Clasificación de Vocales

La función `get_vowel` determina el estado de la mano basándose en la posición relativa en el eje `y` (vertical: valores más bajos representan puntos más altos en la pantalla) de los siguientes landmarks de la mano (ver mapa en [REF §4.2](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#42-los-21-landmarks-de-la-mano)):

- **Articulación PIP** (intermedia): puntos `6` (índice), `10` (medio), `14` (anular), `18` (meñique).
- **Tip** (yema/punta): puntos `8` (índice), `12` (medio), `16` (anular), `20` (meñique).

### Reglas algebraicas para cada vocal:

1. **Dedos Cerrados**: Se considera que un dedo está cerrado si su punta está debajo de su articulación PIP (p. ej., `lm[8].y > lm[6].y`).
2. **Clasificación**:
   - **Vocal 'A'**: Dedos índice, medio, anular y meñique **cerrados**. Pulgar arriba y extendido hacia el lateral exterior de la mano.
   - **Vocal 'E'**: Dedos índice, medio, anular y meñique **cerrados**. Pulgar hacia abajo y plegado frente a los dedos.
   - **Vocal 'I'**: Dedos índice, medio y anular **cerrados**. Dedo meñique **abierto** (apuntando hacia arriba). Pulgar arriba.
   - **Vocal 'U'**: Dedos índice y medio **abiertos**. Dedos anular y meñique **cerrados**. Pulgar abajo.
   - **Vocal 'O'**: Las puntas del pulgar (punto `4`) e índice (punto `8`) se tocan (distancia euclidiana menor a `0.05`), y las puntas del pulgar y medio (punto `12`) también se tocan. Dedos anular y meñique semi-cerrados.

---

## 5. Validación Temporal

Para evitar que detecciones erróneas o parpadeos (glitches del modelo) activen letras de forma descontrolada:
- `vocal_detectada` almacena la vocal detectada en la iteración previa.
- `tiempo_inicio_vocal` registra con `time.time()` el instante en que se observó la vocal por primera vez.
- Si el modelo retorna la misma vocal de forma continua durante más de `TIEMPO_CONFIRMACION = 1.0` segundo, se considera confirmada. Si la mano cambia de posición o sale del cuadro, el estado de validación se reinicia inmediatamente.

---

## 6. OpenCV, teclas y ventana

- **Q**: Cierra el programa.
- **cv2.WINDOW_GUI_NORMAL**: Se utiliza para configurar una ventana limpia de OpenCV sin menús nativos complejos de Qt que ralenticen el flujo.
- **Visualización**: Pinta las marcas en color verde y la notificación de validación/confirmación en la parte superior izquierda de la pantalla.

---

## 7. Cómo ejecutar

Desde la raíz del proyecto, con el entorno `venv` activado:

```bash
python pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py
```

---

## 8. Errores frecuentes

Si notas que no se detecta la letra **O** o las letras se confirman demasiado lento, revisa la distancia euclidiana en el código o reduce el umbral de confirmación. Para fallos comunes de la webcam o del modelo, consulta la **Tabla de Errores Frecuentes Unificada** en la [Sección 6 de REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).

---

## 9. ¿Qué sigue después?

Has aprendido a clasificar gestos **estáticos** calculando relaciones algebraicas directas sobre los landmarks de un solo frame. Sin embargo, para gestos **dinámicos** (que involucran una trayectoria en el tiempo) las reglas estáticas no bastan. 

**Siguiente etapa**:
- [Paso 05 — Recolección](file:///home/thewest/proyectos/GestureFlow/pasos/paso-05-recoleccion/paso_05_doc.md): Grabar secuencias continuas de landmarks en archivos `.npy`.
- [Paso 06 — Entrenamiento](file:///home/thewest/proyectos/GestureFlow/pasos/paso-06-entrenamiento/paso_06_doc.md): Alimentar una red neuronal recurrente LSTM para clasificar movimientos complejos.

---

## 10. Referencia del código fuente

El script completo vive en [paso_04_vocales.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-04-reconocimiento-vocales/paso_04_vocales.py).
