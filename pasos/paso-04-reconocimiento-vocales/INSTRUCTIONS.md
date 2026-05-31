# Step 4: Vowel Recognition (A, E, I, O, U)

In this step, you will create a new folder and write an algorithm to detect the vowels based on the hand landmarks provided by MediaPipe.

## 1. Folder Structure
Create a new folder named `paso-04-reconocimiento-vocales` inside the `pasos` directory. Inside this folder, create a new Python file named `vowel_recognition.py`.

## 2. The Algorithm (Hints & Steps)

To recognize a vowel, you need to analyze the relative positions of the hand landmarks (the 21 points MediaPipe gives you).

### Step 2.1: Understand the Landmarks
MediaPipe provides 21 landmarks for each hand. For example:
- `4` is the Thumb tip.
- `8` is the Index finger tip.
- `12` is the Middle finger tip.
- `16` is the Ring finger tip.
- `20` is the Pinky tip.

### Step 2.2: Define Finger States (Open/Closed)
A good starting point is to write a helper function that checks if a specific finger is "open" (extended) or "closed" (folded). You can do this by comparing the `y` coordinate of the finger tip with the `y` coordinate of the joint below it (e.g., the PIP joint).

**Hint:** In OpenCV, the `y` axis increases downwards. So a point is "higher" on the screen if its `y` value is *smaller*.

```python
# Hint: Checking if the index finger is open
if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
    # Index finger is open
    pass
```

### Step 2.3: Define the Vowel Rules (ASL Basics)
Based on American Sign Language (ASL), here are simple ways to think about the vowels:
- **A**: All fingers closed, thumb resting on the side.
- **E**: All fingers folded down tightly.
- **I**: Only the pinky finger is extended.
- **O**: All fingers curved, tips forming a circle with the thumb.
- **U**: Index and middle fingers extended (and close together), others closed.

### Step 2.4: Implement the Logic
Inside your main loop (where you process the frames), extract the landmarks and apply your rules. Use an `if-elif-else` structure to check which vowel is being formed.

```python
# Hint: Structure for your logic
def get_vowel(hand_landmarks):
    # Get tips and check states
    is_index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    is_pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    # ... get states for middle, ring, and thumb

    # Check for 'I' (simplest one: only pinky is up)
    if is_pinky_up and not is_index_up: # Add the rest of the fingers here
        return 'I'
    
    # Check for 'A', 'E', 'O', 'U'...
    
    return "Unknown"
```

## 3. Challenge
Integrate this logic into your existing real-time camera script (from Step 3) and use `cv2.putText` to display the recognized vowel on the screen!

Good luck! Once you finish this module, remember to commit your changes using the standard semantic commit format, for example:
`feat(vowels): implement real-time vowel recognition logic`
