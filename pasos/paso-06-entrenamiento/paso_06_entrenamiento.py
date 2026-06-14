import tensorflow as tf
from pathlib import Path
import numpy as np

GESTOS_DIR = Path("gestos")

SEQUENCE_LENGTH = 30

NUM_FEATURES = 126 

def cargar_dataset(): #-> tuple[np.ndarray, np.ndarray, list[str]]
    if not GESTOS_DIR.exists(): #Se evalua si la carpeta con gestos existe
        raise FileNotFoundError(f"Gestos no encontrados:{GESTOS_DIR}")
    subdirs = [] #definimos la lista de sub directorios 

    for subdir in sorted(GESTOS_DIR.iterdir()):#recorremos ordenadamente los archivos 
        if subdir.is_dir():  #si es un directorio
            subdirs.append(subdir.name) #agregamos el nombre del directorio a la lista
    return subdirs

if __name__ == "__main__":
    subdirs = cargar_dataset()
    print(subdirs)