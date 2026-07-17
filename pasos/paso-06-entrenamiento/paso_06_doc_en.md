# Documentation: Step 06 — LSTM Model Training (`paso_06_entrenamiento.py`)

Final training step: trains a stacked **LSTM** neural network using temporal sequence data collected in step 5.

For details on recurrent cells, input tensor shapes, labels encoding, and optimizer architectures, refer to [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md).

---

## Index

- [1. Step Objective](#1-step-objective)
- [2. Folder Files](#2-folder-files)
- [3. Pipeline](#3-pipeline)
- [4. LSTM Model Architecture](#5-lstm-model-architecture)
- [5. Training Parameters](#6-training-parameters)
- [6. How to Run](#7-how-to-run)
- [7. Common Errors](#8-common-errors)

---

## 1. Step Objective

**Objective:** Load `.npy` files from `gestos/` and train an LSTM deep learning model to classify dynamic gestures.

| Included in this script | Not included |
|-------------------------|-------------|
| Automatic loading of `.npy` by label folder | Video recording (Step 5) |
| Train/test splitting and encoding | Live inference (Step 7) |
| Keras Sequential LSTM training | Graphical dashboard HUD |
| Evaluation on validation set | OS dispatch (Step 8) |

**Success Criteria:**
- Loads dataset from `gestos/`.
- Compiles and trains the network.
- Exports `modelo_gestos.h5` or `lstm_gestos.keras` to models directory.

---

## 2. Folder Files

| File | Role |
|------|------|
| [paso_06_entrenamiento.py](file:///home/thewest/proyectos/GestureFlow/pasos/paso-06-entrenamiento/paso_06_entrenamiento.py) | Step script |
| [paso_06_doc.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-06-entrenamiento/paso_06_doc.md) | Spanish documentation |
| [paso_06_doc_en.md](file:///home/thewest/proyectos/GestureFlow/pasos/paso-06-entrenamiento/paso_06_doc_en.md) | This English documentation |

---

## 3. Pipeline

```text
1. Scan gestos/ folders.
2. Load and concatenate sequence .npy arrays.
3. Apply Label Encoder and split train/test datasets.
4. Build sequential Keras model with recurrent units.
5. Compile and fit the model (epochs=100).
6. Evaluate accuracy metrics.
7. Save model inside modelos/ folder.
```

---

## 4. LSTM Model Architecture

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(gestos), activation='softmax'),
])
```
- **`return_sequences=True`**: Keeps sequence dimensions for downstream recurrent layers.
- **`Dropout(0.2)`**: Discards 20% of activations to prevent overfitting.
- **`softmax`**: Normalizes predictions to probability distributions.

---

## 5. Training Parameters
- **Loss**: `categorical_crossentropy` for multi-class targets.
- **Optimizer**: `Adam` with default learning rate.
- **Epochs**: `100`.
- **Batch Size**: `32`.

---

## 6. How to Run
```bash
python pasos/paso-06-entrenamiento/paso_06_entrenamiento.py
```

---

## 7. Common Errors
Refer to Section 6 of [REFERENCIA_COMUN.md](file:///home/thewest/proyectos/GestureFlow/pasos/REFERENCIA_COMUN.md#6-tabla-de-errores-frecuentes-unificada).
