import cv2

cap = cv2.VideoCapture(0)  # 0 = cámara por defecto; prueba 1 si no abre

if not cap.isOpened():
    print("Error: No se pudo abrir la camara")
    exit(1)

frame_count = 0
primer_frame_logeado = False

while True:
    ret, frame = cap.read()

    if not ret:
        print(f"Error: No se pudo leer el frame (tras {frame_count} frames OK)") # Log de error
        break # Salir del bucle si no se pudo leer el frame

    frame = cv2.flip(frame, 1)  # espejo horizontal (orientación natural)
    frame_count += 1 # Incrementar el contador de frames

    if not primer_frame_logeado:
        print(f"Primer frame OK: shape={frame.shape}, dtype={frame.dtype}") # Log de primer frame OK
        primer_frame_logeado = True
    elif frame_count % 100 == 0:
        print(f"Frames OK: {frame_count}") # Log de frames OK

    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Texto de frames OK
    cv2.putText(frame,
        "Pulsa Q para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Texto de salida

    cv2.imshow("Paso 01 - Camara", frame) # Mostrar el frame en la ventana

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break # Salir del bucle si se presiona la tecla 'q'

print(f"Total frames leidos: {frame_count}") # Log de total frames leidos
cap.release()
cv2.destroyAllWindows()
