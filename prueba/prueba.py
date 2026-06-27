
# --- Vision and processing libraries ---
import cv2  # OpenCV: read images (BGR) and show windows
import mediapipe as mp  # MediaPipe: hand detection and drawing utilities
from mediapipe.tasks import python  # Base model configuration (.task)
from mediapipe.tasks.python import vision  # HandLandmarker and running modes
from pathlib import Path  # Cross-platform paths without depending on the current directory

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
mp_hands = mp.tasks.vision.HandLandmarksConnections


# --- Paths relative to the script (works regardless of the execution folder) ---

SCRIPT_DIR = Path(__file__).resolve().parent  # Folder where this .py lives (prueba/)
PROJECT_ROOT = SCRIPT_DIR.parent  # Project root (GestureFlow/)

image_path = PROJECT_ROOT / "assets" / "img_prueba" /  "image.png"  # Input image
model_path = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"  # Pre-trained MediaPipe model

# --- Load the image into memory (OpenCV BGR format) ---

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# --- HandLandmarker detector configuration ---
base_options = python.BaseOptions(
    model_asset_path=str(model_path)  # Path to the .task file on disk
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  # Single image (not video or live camera)
    num_hands=2,  # Maximum number of hands to look for in the frame

)

# The context manager releases the model resources when leaving the block
with vision.HandLandmarker.create_from_options(options) as landmarker:
    # MediaPipe expects RGB; OpenCV stores in BGR → mandatory conversion

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),

    )
    # Inference: returns normalised landmarks (x, y, z between 0 and 1 relative to image size)
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        # One entry per detected hand (up to num_hands)
        for hand_landmarks in results.hand_landmarks:
            # Draw circles on each landmark and lines according to HAND_CONNECTIONS
            mp_drawing.draw_landmarks(
                image,  # Modified in-place (same matrix as imread)
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,  # Which points to connect (hand skeleton)
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

# --- Display the result ---
# Create the window with normal GUI to avoid the Qt toolbar
cv2.namedWindow("Result", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("Result", image)  # Window with the image and drawn hands
cv2.waitKey(0)  # Pause until a key is pressed
cv2.destroyAllWindows()  # Close all OpenCV windows

