from pathlib import Path
import sys

# Resolve project root and insert it into sys.path to find config/utils
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import config as tf_config

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical

import config
from utils import get_gesture_names

GESTOS_DIR: Path = config.GESTOS_DIR
SEQUENCE_LENGTH: int = config.SEQUENCE_LENGTH
NUM_FEATURES: int = config.NUM_FEATURES
TEST_SIZE: float = config.TEST_SIZE
RANDOM_STATE: int = config.RANDOM_STATE
EPOCHS: int = config.EPOCHS
BATCH_SIZE: int = config.BATCH_SIZE
MODEL_PATH: Path = config.MODEL_PATH

def verificar_entorno() -> None:
    """Print TF version and list available physical devices."""
    print(f"TensorFlow version: {tf.__version__}")
    dispositivos = tf_config.list_physical_devices()
    print(f"Physical devices found: {dispositivos}")

def cargar_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    gestos = get_gesture_names(GESTOS_DIR)  # validates existence and ≥ 2 classes
    X, Y = [], []

    for i, gesto in enumerate(gestos):
        gesto_path = GESTOS_DIR / gesto
        for npy_file in gesto_path.glob("*.npy"):
            secuencia = np.load(npy_file) # Load the .npy file
            
            # Adjust sequence length (frames)
            f_count = secuencia.shape[0]
            if f_count < SEQUENCE_LENGTH:
                # Pad with zeros at the end if frames are missing
                padding = np.zeros((SEQUENCE_LENGTH - f_count, secuencia.shape[1]), dtype=np.float32)
                secuencia = np.concatenate([secuencia, padding], axis=0)
            elif f_count > SEQUENCE_LENGTH:
                # Trim if there are too many frames
                secuencia = secuencia[:SEQUENCE_LENGTH, :]

            # Adjust number of features (coordinates)
            feat_count = secuencia.shape[1]
            if feat_count < NUM_FEATURES:
                # Pad with zeros if only one hand was detected (e.g. 63 features to 126)
                padding = np.zeros((secuencia.shape[0], NUM_FEATURES - feat_count), dtype=np.float32)
                secuencia = np.concatenate([secuencia, padding], axis=1)
            elif feat_count > NUM_FEATURES:
                # Trim if it exceeds the expected features
                secuencia = secuencia[:, :NUM_FEATURES]
                
            X.append(secuencia) # Append the sequence to the X list
            Y.append(i) # Append the gesture index to the Y list
    
    X = np.array(X, dtype=np.float32) # Convert X list to a numpy array
    Y = np.array(Y, dtype=np.int32) # Convert Y list to a numpy array

    # print(f"X:{X.shape}\nY:{Y.shape}")

    return X, Y, gestos # Return the X list, the Y list and the gestures list

def procesar(X:np.ndarray, Y:np.ndarray, num_clases:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=TEST_SIZE,  # reserve 20% of the data for testing and the rest for training
    random_state=RANDOM_STATE, 
    stratify=Y) # stratification to ensure equal class proportions in both sets

    Y_train_cat = to_categorical(Y_train, num_classes=num_clases) # represent data in one-hot encoding format readable by the model
    Y_test_cat  = to_categorical(Y_test, num_classes=num_clases) 

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_train_cat shape:", Y_train_cat.shape)
    print("Y_test_cat shape:", Y_test_cat.shape)
    return X_train, X_test, Y_train_cat, Y_test_cat

def construir_modelo_mejorado(input_shape: tuple[int, ...], num_classes: int) -> Sequential:
    model = Sequential() # Initialize the Keras sequential model
    
    # First LSTM layer: processes the sequence and returns sequences for the next LSTM layer
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, 
                   kernel_regularizer=l2(0.001))) # Add an LSTM layer with 128 neurons and L2 regularization
    model.add(BatchNormalization()) # Add a batch normalization layer
    model.add(Dropout(0.3)) # Add a dropout layer with a 30% dropout rate
    
    # Second LSTM layer: summarises the sequence into a single 64-dimensional vector (return_sequences=False)
    model.add(LSTM(64, return_sequences=False, 
                   kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization()) # Add a batch normalization layer
    model.add(Dropout(0.3)) # Add a dropout layer with a 30% dropout rate
    
    # Dense classification layer
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001))) # Intermediate hidden layer to refine features
    model.add(BatchNormalization()) # Add a batch normalization layer
    model.add(Dropout(0.3)) # Add a dropout layer with a 30% dropout rate
    
    # Output layer with softmax activation for class probabilities
    model.add(Dense(num_classes, activation='softmax')) # Output layer with softmax activation for class probabilities
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model defining the loss function and optimizer
    
    model.summary() # Display a model summary in the console
    return model

def entrenar_modelo(model: Sequential, X_train: np.ndarray, Y_train_cat: np.ndarray, X_test: np.ndarray, Y_test_cat: np.ndarray) -> None:
   callback = [] # Initialize the callbacks list
   callback.append(EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)) # Add the early stopping callback
   callback.append(ModelCheckpoint(
    filepath=MODEL_PATH, # Define the path where the model will be saved
    monitor='val_accuracy', # Define the metric to monitor
    verbose=1, # Enable verbose (shows training information)
    save_best_only=True, # Save only the best model
    save_weights_only=False, # Save the full model (not just weights)
    mode='auto', # Saving mode (auto means save when the monitored metric improves)
    save_freq='epoch' # Save frequency (once per epoch, i.e. each pass through the data)
   )) 
   history = model.fit(
    X_train, Y_train_cat, # Define the training data
    validation_data=(X_test, Y_test_cat), # Define the test data
    epochs=EPOCHS, # Define the number of epochs
    batch_size=BATCH_SIZE, # Define the batch size
    callbacks=callback, # Define the callbacks list
    verbose=2 # Enable verbose (level 2 shows training progress)
    )

def evaluar(model: Sequential, X_test: np.ndarray, Y_test_cat: np.ndarray) -> None:
   loss, accuracy = model.evaluate(X_test, Y_test_cat, verbose=0) # Evaluate the model on the test set
   print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}") # Display loss and accuracy
   

def evaluar_f1(model: Sequential, X_test: np.ndarray, Y_test_cat: np.ndarray, gestos: list[str]) -> None:
    predictions=model.predict(X_test) # Get the model predictions
    predicciones_clase = np.argmax(predictions, axis=1) # Get the predictions in class format
    y_true = np.argmax(Y_test_cat, axis=1) # Get the true labels in class format
    print(classification_report(y_true, predicciones_clase, target_names=gestos, labels=range(len(gestos)))) # Display the classification report

def guardar_modelo(model: Sequential) -> None:
    """Serialize the model to disk."""
    model.save(str(MODEL_PATH))
    print(f"Model saved to: {MODEL_PATH}")

def main() -> None:
    """Orchestrate the full training pipeline."""
    verificar_entorno()

    X, Y, gestos = cargar_dataset()
    print("--- Dataset loaded ---")
    
    num_classes = len(gestos)
    X_train, X_test, Y_train_cat, Y_test_cat = procesar(X, Y, num_classes)
    print("--- Dataset processed ---")
    
    input_shape = X_train.shape[1:]
    print("--- Building model ---")
    model = construir_modelo_mejorado(input_shape, num_classes)
    print("\n--- Model built ---\n")
    
    print("--- Training model ---")
    entrenar_modelo(model, X_train, Y_train_cat, X_test, Y_test_cat)
    print("\n--- Model trained ---\n")
    
    print("--- Evaluating model ---")
    evaluar(model, X_test, Y_test_cat)
    print("\n--- Model evaluated ---\n")
    
    print("--- Evaluating model F1 ---")
    evaluar_f1(model, X_test, Y_test_cat, gestos)
    print("--- Model F1 evaluated ---\n")

    guardar_modelo(model)

if __name__ == "__main__":
    main()