import tensorflow as tf
from pathlib import Path
import numpy as np

GESTOS_DIR = Path("gestos")

SEQUENCE_LENGTH = 30

NUM_FEATURES = 126 

def cargar_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not GESTOS_DIR.exists(): #Se evalua si la carpeta con gestos existe
        raise FileNotFoundError(f"Gestos no encontrados:{GESTOS_DIR}")
    subdirs = [] #definimos la lista de sub directorios 

    for subdir in sorted(GESTOS_DIR.iterdir()):#recorremos ordenadamente los archivos 
        if subdir.is_dir():  #si es un directorio
            subdirs.append(subdir.name) #agregamos el nombre del directorio a la lista
    if len(subdirs)<2:
        raise ValueError("Deben haber al menos 2 gestos")

    X, Y, gestos = [], [], [] #Se definen las listas que almacenaran los datos

    for i, gesto in enumerate(subdirs): #se recorre cada gesto en busqueda de losarchivos npy
        gesto_path = GESTOS_DIR / gesto
        for npy_file in gesto_path.glob("*.npy"):
            secuencia = np.load(npy_file) #Se carga el archivo .npy
            if secuencia.shape[0] != SEQUENCE_LENGTH: #Se valida que la secuencia tenga la longitud correcta
                continue
            if secuencia.shape[1] != NUM_FEATURES: #Se valida que la secuencia tenga el numero correcto de coordenadas
                continue
            X.append(secuencia) #Se agrega la secuencia a la lista X
            Y.append(i) #Se agrega el indice del gesto a la lista Y
            gestos.append(gesto) #Se agrega el nombre del gesto a la lista gestos
    
    X = np.array(X, dtype=np.float32) #Convertimos la lista X a un array de numpy con tipo de dato float32 para que el modelo lo pueda procesar
    Y = np.array(Y, dtype=np.int32) #Convertimos la lista Y a un array de numpy con tipo de dato int32

    print(f"X:{X.shape}\nY:{Y.shape}") #Imprimimos la forma de X y Y

    return X, Y, gestos #Se retorna la lista X, la lista Y y la lista gestos



if __name__ == "__main__":
    X,Y,gestos = cargar_dataset()
    print(gestos)