import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('pneumonia_classifier_final.keras')

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define class labels
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
