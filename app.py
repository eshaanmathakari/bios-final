import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from datetime import datetime
import json

# Load the trained model
model = tf.keras.models.load_model('pneumonia_classifier_final.keras')

# Debugging 
print("App.py TensorFlow version:", tf.__version__)

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define class labels
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']

st.title("Medical Image Classifier")

# Allow the user to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Load the image
    image = Image.open(temp_file_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img = img.convert('RGB')  # Ensure image has 3 color channels
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = CLASS_LABELS[predicted_class[0]]
    confidence_scores = predictions[0]

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write("**Confidence Scores:**")
    for idx, score in enumerate(confidence_scores):
        st.write(f"{CLASS_LABELS[idx]}: {score:.4f}")

    # Feedback section
    st.write("### Is the prediction incorrect? Please provide the correct label:")
    correct_label = st.selectbox("Select the correct label:", CLASS_LABELS)
    submit_feedback = st.button("Submit Feedback")

    if submit_feedback:
        # Save the feedback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_dir = 'feedback_data'
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_data = {
            'timestamp': timestamp,
            'original_filename': uploaded_file.name,
            'predicted_label': predicted_label,
            'correct_label': correct_label,
            'confidence_scores': confidence_scores.tolist()
        }
        # Save the image and feedback
        feedback_image_dir = os.path.join(feedback_dir, correct_label)
        os.makedirs(feedback_image_dir, exist_ok=True)
        feedback_image_path = os.path.join(feedback_image_dir, f"{timestamp}_{uploaded_file.name}")
        image.save(feedback_image_path)
        feedback_json_path = os.path.join(feedback_dir, f"{timestamp}_{uploaded_file.name}.json")
        with open(feedback_json_path, 'w') as f:
            json.dump(feedback_data, f)
        st.write("Thank you for your feedback!")

    # Clean up the temporary file
    os.remove(temp_file_path)

