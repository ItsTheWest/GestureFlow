from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint #para el monitoreo y control del entrenamiento
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical #para convertir los datos a one-hot encoding
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split #para dividir los datos en conjuntos de entrenamiento y prueba


GESTOS_DIR = Path("gestos")

SEQUENCE_LENGTH = 30 #Se define la longitud de la secuencia

NUM_FEATURES = 126 #Se define el numero de caracteristicas

TEST_SIZE    = 0.20 #Se define el tamaño del conjunto de prueba
RANDOM_STATE = 42 #Se define la semilla para la generacion de numeros aleatorios

EPOCHS     = 100 #Numero de epochs (los epochs son las veces que el modelo se entrenara con los datos)
BATCH_SIZE = 32 #Tamaño del batch (el batch es el numero de muestras que se procesaran al mismo tiempo)
MODEL_PATH = Path("modelos/lstm_gestos.keras") #Ruta donde se guardara el modelo

def verificar_entorno() -> None:
    """Print TF version and list available physical devices."""
    print(f"TensorFlow version: {tf.__version__}")
    dispositivos = tf.config.list_physical_devices()
    print(f"Dispositivos físicos encontrados: {dispositivos}")

def cargar_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not GESTOS_DIR.exists(): #Se evalua si la carpeta con gestos existe
        raise FileNotFoundError(f"Gestos no encontrados:{GESTOS_DIR}")
    subdirs = [] #definimos la lista de sub directorios 

    for subdir in sorted(GESTOS_DIR.iterdir()):#recorremos ordenadamente los archivos 
        if subdir.is_dir():  #si es un directorio
            subdirs.append(subdir.name) #agregamos el nombre del directorio a la lista
    if len(subdirs)<2:
        raise ValueError("Deben haber al menos 2 gestos")

    gestos = subdirs  # Las clases corresponden exactamente a los subdirectorios ordenados
    X, Y = [], [] # Se definen las listas que almacenaran los datos

    for i, gesto in enumerate(subdirs): # se recorre cada gesto en busqueda de los archivos npy
        gesto_path = GESTOS_DIR / gesto
        for npy_file in gesto_path.glob("*.npy"):
            secuencia = np.load(npy_file) # Se carga el archivo .npy
            
            # Ajustar longitud de la secuencia (frames)
            f_count = secuencia.shape[0]
            if f_count < SEQUENCE_LENGTH:
                # Rellenar con ceros al final si faltan frames
                padding = np.zeros((SEQUENCE_LENGTH - f_count, secuencia.shape[1]), dtype=np.float32)
                secuencia = np.concatenate([secuencia, padding], axis=0)
            elif f_count > SEQUENCE_LENGTH:
                # Recortar si tiene frames de mas
                secuencia = secuencia[:SEQUENCE_LENGTH, :]

            # Ajustar numero de caracteristicas (coordenadas)
            feat_count = secuencia.shape[1]
            if feat_count < NUM_FEATURES:
                # Rellenar con ceros si solo se detecto una mano (ej. 63 features a 126)
                padding = np.zeros((secuencia.shape[0], NUM_FEATURES - feat_count), dtype=np.float32)
                secuencia = np.concatenate([secuencia, padding], axis=1)
            elif feat_count > NUM_FEATURES:
                # Recortar si excede las caracteristicas esperadas
                secuencia = secuencia[:, :NUM_FEATURES]
                
            X.append(secuencia) # Se agrega la secuencia a la lista X
            Y.append(i) # Se agrega el indice del gesto a la lista Y
    
    X = np.array(X, dtype=np.float32) # Convertimos la lista X a un array de numpy
    Y = np.array(Y, dtype=np.int32) # Convertimos la lista Y a un array de numpy

    # print(f"X:{X.shape}\nY:{Y.shape}")

    return X, Y, gestos # Se retorna la lista X, la lista Y y la lista gestos

def procesar(X:np.ndarray, Y:np.ndarray, num_clases:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=TEST_SIZE,  # se reserva el 20% porciento de modelo para probar y el restante para el entrenamiento
    random_state=RANDOM_STATE, 
    stratify=Y) # estratificación para asegurar que las proporciones de clases sean iguales en ambos conjuntos

    Y_train_cat = to_categorical(Y_train, num_classes=num_clases) # representa los datos en formato one-hot encoding  legible para el modelo
    Y_test_cat  = to_categorical(Y_test, num_classes=num_clases) 

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_train_cat shape:", Y_train_cat.shape)
    print("Y_test_cat shape:", Y_test_cat.shape)
    return X_train, X_test, Y_train_cat, Y_test_cat

def construir_modelo_mejorado(input_shape: tuple[int, ...], num_classes: int) -> Sequential:
    model = Sequential() # Se inicializa el modelo secuencial de keras
    
    # Primera capa LSTM: procesa la secuencia y devuelve secuencias para la siguiente capa LSTM
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, 
                   kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization()) # Se agrega una capa de normalizacion por lotes
    model.add(Dropout(0.3)) # Se agrega una capa dropout con una tasa de dropout del 30% 
    
    # Segunda capa LSTM: resume la secuencia en un único vector de 64 dimensiones (return_sequences=False)
    model.add(LSTM(64, return_sequences=False, 
                   kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization()) # Se agrega una capa de normalizacion por lotes
    model.add(Dropout(0.3)) # Se agrega una capa dropout con una tasa de dropout del 30% 
    
    # Capa densa de clasificación
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001))) # Capa intermedia oculta para refinar características
    model.add(BatchNormalization()) # Se agrega una capa de normalizacion por lotes
    model.add(Dropout(0.3)) # Se agrega una capa dropout con una tasa de dropout del 30% 
    
    # Capa de salida con activación softmax para las probabilidades de clase
    model.add(Dense(num_classes, activation='softmax')) # Capa de salida con activación softmax para las probabilidades de clase
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compilación del modelo en el que se define la funcion de perdida y el optimizador
    
    model.summary() # Se muestra un resumen del modelo en consola
    return model

def entrenar_modelo(model: Sequential, X_train: np.ndarray, Y_train_cat: np.ndarray, X_test: np.ndarray, Y_test_cat: np.ndarray) -> None:
   callback = [] # Se inicializa la lista de callbacks
   callback.append(EarlyStopping(monitor="val_accuracy",patience=EPOCHS,restore_best_weights=True)) # Se agrega el callback de early stopping
   callback.append(ModelCheckpoint(
    filepath=MODEL_PATH, # Se define la ruta donde se guardara el modelo
    monitor='val_accuracy', # Se define la metrica a monitorear
    verbose=1, # Se habilita el verbose (sirve para que se muestre informacion sobre el entrenamiento)
    save_best_only=True, # Se guarda solo el mejor modelo
    save_weights_only=False, # Se guardan los pesos del modelo
    mode='auto', # Se define el modo de guardado (en este caso, auto, esto quiere decir que se guardara el modelo si la metrica monitoreada mejora)
    save_freq='epoch' # Se define la frecuencia de guardado (en este caso, cada epoch osea cada pasada por los datos)
   )) 
   history = model.fit(
    X_train, Y_train_cat, # Se define los datos de entrenamiento
    validation_data=(X_test, Y_test_cat), # Se define los datos de prueba
    epochs=EPOCHS, # Se define el numero de epochs
    batch_size=BATCH_SIZE, # Se define el tamaño del batch
    callbacks=callback, # Se define la lista de callbacks
    verbose=2 # Se habilita el verbose (el dos nos ayuda a ver el entrenamiento con una barra de progreso)
    )

def evaluar(model: Sequential, X_test: np.ndarray, Y_test_cat: np.ndarray) -> None:
   loss, accuracy = model.evaluate(X_test, Y_test_cat, verbose=0) # Se evalua el modelo en el conjunto de prueba
   print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}") # Se muestra el loss y la accuracy
   

def evaluar_f1(model: Sequential, X_test: np.ndarray, Y_test_cat: np.ndarray, gestos: list[str]) -> None:
    predictions=model.predict(X_test) # Se obtienen las predicciones del modelo
    predicciones_clase = np.argmax(predictions, axis=1) # Se obtienen las predicciones en formato de clase
    y_true = np.argmax(Y_test_cat, axis=1) # Se obtienen las etiquetas verdaderas en formato de clase
    print(classification_report(y_true, predicciones_clase, target_names=gestos, labels=range(len(gestos)))) # Se muestra el reporte de clasificación

def guardar_modelo(model: Sequential) -> None:
    """Serialize the model to disk."""
    model.save(str(MODEL_PATH))
    print(f"Modelo guardado en: {MODEL_PATH}")

def main() -> None:
    """Orchestrate the full training pipeline."""
    verificar_entorno()

    X, Y, gestos = cargar_dataset()
    print("--- Dataset cargado ---")
    
    num_classes = len(gestos)
    X_train, X_test, Y_train_cat, Y_test_cat = procesar(X, Y, num_classes)
    print("--- Dataset procesado ---")
    
    input_shape = X_train.shape[1:]
    print("--- Construyendo modelo ---")
    model = construir_modelo_mejorado(input_shape, num_classes)
    print("\n--- Modelo construido ---\n")
    
    print("--- Entrenando modelo ---")
    entrenar_modelo(model, X_train, Y_train_cat, X_test, Y_test_cat)
    print("\n--- Modelo entrenado ---\n")
    
    print("--- Evaluando modelo ---")
    evaluar(model, X_test, Y_test_cat)
    print("\n--- Modelo evaluado ---\n")
    
    print("--- Evaluando modelo F1 ---")
    evaluar_f1(model, X_test, Y_test_cat, gestos)
    print("--- Modelo evaluado F1 ---\n")

    guardar_modelo(model)

if __name__ == "__main__":
    main()