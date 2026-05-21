import cv2

cap = cv2.VideoCapture(0) # 0 es la camara default, si hay mas camaras, se puede cambiar por el numero de la camara

# Verificar si la camara se abrio correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit()

# Leer frames de la camara
while True:
    ret, frame = cap.read()
    
    # Verificar si se pudo leer el frame
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # Convertir el frame a escala de grises ya que el modelo de mediapipe necesita una imagen en escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Frame", gray)
    # Esperar a que se presione la tecla 'q' para salir
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la camara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()