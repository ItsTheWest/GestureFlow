import tensorflow as tf
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

GESTOS_DIR = Path("gestos")

SEQUENCE_LENGTH = 30 #Se define la longitud de la secuencia

NUM_FEATURES = 126 #Se define el numero de caracteristicas

TEST_SIZE    = 0.20 #Se define el tamaño del conjunto de prueba
RANDOM_STATE = 42 #Se define la semilla para la generacion de numeros aleatorios

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

def procesar(X:np.ndarray, Y:np.ndarray, num_clases:int):
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=TEST_SIZE,  # se reserva el 20% porciento de modelo para probar y el restante para el entrenamiento
    random_state=RANDOM_STATE, 
    stratify=Y) 
    
if __name__ == "__main__":
    X,Y,gestos = cargar_dataset()
    print(gestos)