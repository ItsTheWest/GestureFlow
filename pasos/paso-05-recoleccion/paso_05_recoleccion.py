import os, time
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

SEQUENCE_LENGTH = 30   # Número de frames por secuencia
NUM_FEATURES = 63      # 21 landmarks * 3 coordenadas (x, y, z)

def pedir_nombre_gesto():
    name_gesture_folder = input ('Nombre del gesto a crear:')
    normalized_folder = name_gesture_folder.strip().lower()
    gesture_folder = Path(f"gestos/{normalized_folder}")
    gesture_folder.mkdir(parents=True, exist_ok=True) #Creación de la carpeta si no existe
    if not gesture_folder.exists():
       return print(f"Error: No se pudo crear la carpeta: {gesture_folder}")
    else:
       return print('Carpeta creada')

pedir_nombre_gesto()
