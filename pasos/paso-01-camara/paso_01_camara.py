import cv2

cap = cv2.VideoCapture(0)  # 0 = default camera; try 1 if it doesn't open

if not cap.isOpened():
    print("Error: Could not open the camera")
    exit(1)

# Create the window with normal GUI to avoid the Qt toolbar
cv2.namedWindow("Paso 01 - Camara", cv2.WINDOW_GUI_NORMAL)

frame_count = 0
primer_frame_logeado = False

while True:
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame (after {frame_count} OK frames)") # Error log
        break # Exit the loop if the frame could not be read

    frame = cv2.flip(frame, 1)  # horizontal mirror (natural orientation)
    frame_count += 1 # Increment the frame counter

    if not primer_frame_logeado:
        print(f"First frame OK: shape={frame.shape}, dtype={frame.dtype}") # First frame OK log
        primer_frame_logeado = True
    elif frame_count % 100 == 0:
        print(f"Frames OK: {frame_count}") # Frames OK log

    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Frame count overlay
    cv2.putText(frame,
        "Press Q to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Exit instruction overlay

    cv2.imshow("Paso 01 - Camara", frame) # Show the frame in the window

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break # Exit the loop when 'q' is pressed

print(f"Total frames read: {frame_count}") # Total frames read log
cap.release()
cv2.destroyAllWindows()
