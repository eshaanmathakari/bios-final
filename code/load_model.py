import tensorflow as tf
import tensorflow.keras.backend as K
import os

# Clear backend
K.clear_session()

# Define model paths
MODEL_PATH = 'pneumonia_classifier_final.keras'
UPDATED_MODEL_PATH = 'pneumonia_classifier_updated.keras'

def load_model():
    """Load the latest model available."""
    if os.path.exists(UPDATED_MODEL_PATH):
        model_path = UPDATED_MODEL_PATH
    else:
        model_path = MODEL_PATH
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
        return model
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        exit(1)

# Load the trained model
model = load_model()

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define class labels
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
