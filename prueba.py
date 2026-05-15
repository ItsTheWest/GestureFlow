import cv2 
import mediapipe as mp

cap = cv2.VideoCapture(0) # 0 suele ser la cámara integrada

while cap.isOpened():
    success, frame = cap.read() # condicion de éxito y lectura de la imagen
    if not success:
        break

    # Invertir la imagen horizontalmente
    frame = cv2.flip(frame, 1)
    print("Tipo de frame:", type(frame))
    print(f"Resolución (Alto, Ancho, Canales): {frame.shape}")

    if cv2.waitKey(1) & 0xFF == ord('q'): # salir del bucle si se presiona la tecla 'q'
        break

cap.release()
cv2.destroyAllWindows()
