"""
paso_06_recolecion.py — Entrenamiento LSTM
==========================================

Carga los arrays .npy generados por paso_05, construye y entrena una red
LSTM (Keras / TensorFlow), evalúa la precisión final y guarda el modelo
entrenado en ``modelo_gestos.h5`` en la raíz del proyecto.

Pipeline
--------
1. Escanear ``gestos/`` → lista ordenada de nombres de gestos.
2. Cargar cada .npy y asignar etiqueta entera (label encoding).
3. Dividir en conjuntos de entrenamiento y prueba (80 / 20).
4. Convertir etiquetas a one-hot encoding.
5. Construir el modelo LSTM secuencial.
6. Compilar con Adam + categorical_crossentropy.
7. Entrenar (model.fit) con datos de validación.
8. Evaluar accuracy en el conjunto de prueba.
9. Guardar el modelo en .h5.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
# Path resolution  (same convention as previous steps)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

GESTOS_DIR   = PROJECT_ROOT / "gestos"
MODEL_PATH   = PROJECT_ROOT / "modelo_gestos.h5"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30    # Frames per sequence — must match paso_05
NUM_FEATURES    = 126   # 42 landmarks × 3 coords  — must match paso_05
TEST_SIZE       = 0.20  # Fraction of data reserved for evaluation
EPOCHS          = 100
BATCH_SIZE      = 32
RANDOM_STATE    = 42    # Reproducible splits


# ---------------------------------------------------------------------------
# Step 1 — Data loading
# ---------------------------------------------------------------------------
def cargar_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Escanea ``gestos/`` y carga todos los archivos ``.npy``.

    Retorna:
        X       -- array de forma ``(N, 30, 126)``
        Y       -- array de enteros de forma ``(N,)``
        gestos  -- lista ordenada de nombres de gestos (índice == etiqueta)

    Lanza:
        FileNotFoundError  si el directorio ``gestos/`` no existe.
        ValueError         si se encuentra menos de 2 gestos.
    """
    if not GESTOS_DIR.is_dir():
        raise FileNotFoundError(
            f"No se encontró el directorio de gestos: {GESTOS_DIR}\n"
            "Ejecuta paso_05 primero para recolectar datos."
        )

    gestos = sorted([d.name for d in GESTOS_DIR.iterdir() if d.is_dir()])

    if len(gestos) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 gestos para entrenar. "
            f"Solo se encontró: {gestos}"
        )

    print(f"Gestos encontrados: {gestos}")

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for label_idx, gesto in enumerate(gestos):
        carpeta = GESTOS_DIR / gesto
        archivos = sorted(carpeta.glob("*.npy"))

        if not archivos:
            print(f"  [AVISO] Carpeta vacía, se omite: {gesto}")
            continue

        for archivo in archivos:
            secuencia = np.load(str(archivo))  # expected shape: (30, 126)
            sequences.append(secuencia)
            labels.append(label_idx)

    X = np.array(sequences, dtype=np.float32)  # (N, 30, 126)
    Y = np.array(labels, dtype=np.int32)        # (N,)

    print(f"Total de secuencias cargadas: {len(X)}")
    print(f"Forma del tensor X: {X.shape}  |  Forma de Y: {Y.shape}")

    return X, Y, gestos


# ---------------------------------------------------------------------------
# Step 2 — Preprocessing: train/test split + one-hot encoding
# ---------------------------------------------------------------------------
def preprocesar(
    X: np.ndarray,
    Y: np.ndarray,
    num_clases: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide el dataset y convierte las etiquetas a vectores one-hot.

    Retorna:
        (X_train, X_test, Y_train_cat, Y_test_cat)
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y,        # mantener proporción de clases en ambos splits
    )

    Y_train_cat = to_categorical(Y_train, num_classes=num_clases)
    Y_test_cat  = to_categorical(Y_test,  num_classes=num_clases)

    print(f"\nSplit  →  train: {len(X_train)}  |  test: {len(X_test)}")
    return X_train, X_test, Y_train_cat, Y_test_cat


# ---------------------------------------------------------------------------
# Step 3 — Model construction
# ---------------------------------------------------------------------------
def construir_modelo(num_clases: int) -> Sequential:
    """
    Construye y retorna la arquitectura LSTM secuencial.

    Arquitectura
    ------------
    LSTM(64, return_sequences=True)  →  Dropout(0.2)
    LSTM(128, return_sequences=False) →  Dropout(0.2)
    Dense(64, relu)
    Dense(num_clases, softmax)

    Args:
        num_clases: Número de gestos distintos a clasificar.

    Retorna:
        Modelo Keras compilado listo para entrenar.
    """
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(num_clases, activation="softmax"),
        ],
        name="GestureFlow_LSTM",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ---------------------------------------------------------------------------
# Step 4 — Training
# ---------------------------------------------------------------------------
def entrenar(
    model: Sequential,
    X_train: np.ndarray,
    Y_train_cat: np.ndarray,
    X_test: np.ndarray,
    Y_test_cat: np.ndarray,
) -> None:
    """
    Entrena el modelo con Early Stopping y guarda el mejor checkpoint.

    Callbacks
    ---------
    - EarlyStopping: detiene si ``val_accuracy`` no mejora en 15 épocas.
    - ModelCheckpoint: guarda el mejor modelo automáticamente durante el
      entrenamiento (basado en ``val_accuracy``).

    Args:
        model:       Modelo compilado.
        X_train:     Secuencias de entrenamiento ``(N_train, 30, 126)``.
        Y_train_cat: Etiquetas one-hot ``(N_train, num_clases)``.
        X_test:      Secuencias de evaluación ``(N_test, 30, 126)``.
        Y_test_cat:  Etiquetas one-hot ``(N_test, num_clases)``.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nIniciando entrenamiento — máximo {EPOCHS} épocas ...\n")
    model.fit(
        X_train,
        Y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test_cat),
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# Step 5 — Evaluation
# ---------------------------------------------------------------------------
def evaluar(
    model: Sequential,
    X_test: np.ndarray,
    Y_test_cat: np.ndarray,
) -> None:
    """
    Evalúa el modelo final en el conjunto de prueba e imprime la métrica.

    Args:
        model:      Modelo entrenado (mejores pesos restaurados).
        X_test:     Secuencias de prueba.
        Y_test_cat: Etiquetas one-hot de prueba.
    """
    loss, accuracy = model.evaluate(X_test, Y_test_cat, verbose=0)
    print(f"\n{'='*50}")
    print(f"  Accuracy final en test : {accuracy:.2%}")
    print(f"  Loss final en test     : {loss:.4f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Step 6 — Save
# ---------------------------------------------------------------------------
def guardar_modelo(model: Sequential) -> None:
    """
    Guarda el modelo entrenado en formato HDF5 en la raíz del proyecto.

    El archivo ya fue guardado por ModelCheckpoint (mejor época), aquí se
    realiza un guardado final explícito como confirmación.

    Args:
        model: Modelo con mejores pesos restaurados por EarlyStopping.
    """
    model.save(str(MODEL_PATH))
    print(f"Modelo guardado en: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Orquesta el pipeline completo de entrenamiento LSTM."""

    # 1. Cargar datos
    X, Y, gestos = cargar_dataset()

    # 2. Preprocesar
    num_clases = len(gestos)
    X_train, X_test, Y_train_cat, Y_test_cat = preprocesar(X, Y, num_clases)

    # 3. Construir modelo
    model = construir_modelo(num_clases)

    # 4. Entrenar (callbacks guardan el mejor modelo automáticamente)
    entrenar(model, X_train, Y_train_cat, X_test, Y_test_cat)

    # 5. Evaluar
    evaluar(model, X_test, Y_test_cat)

    # 6. Guardado explícito final
    guardar_modelo(model)


if __name__ == "__main__":
    main()
