# from flask import Flask, request, render_template, redirect, url_for
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import shutil

# app = Flask(__name__, static_url_path='/static')

# # Load your trained model
# model = load_model('pneumonia_classifier_final.keras')

# # Define image dimensions and class labels
# img_height, img_width = 224, 224
# classes = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
# class_labels = classes

# # Ensure the 'uploads' and 'feedback_data' directories exist
# os.makedirs('static/uploads', exist_ok=True)
# os.makedirs('feedback_data', exist_ok=True)

# def predict_image(img_path):
#     # Load and preprocess the image
#     img = image.load_img(img_path, target_size=(img_height, img_width))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

#     # Make a prediction
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)
#     predicted_label = class_labels[predicted_class[0]]

#     # Get confidence scores
#     confidence_scores = predictions[0]

#     return predicted_label, confidence_scores

# def collect_feedback(img_path, correct_label):
#     # Create directory for the correct class if it doesn't exist
#     class_dir = os.path.join('feedback_data', correct_label)
#     os.makedirs(class_dir, exist_ok=True)

#     # Copy the image to the feedback directory
#     img_name = os.path.basename(img_path)
#     dest_path = os.path.join(class_dir, img_name)
#     shutil.copy(img_path, dest_path)

#     print(f"Feedback collected: {img_name} -> {correct_label}")

# def check_and_retrain(feedback_threshold=10):
#     # Count the total number of feedback images
#     total_feedback_images = sum(
#         [len(files) for r, d, files in os.walk('feedback_data')]
#     )

#     if total_feedback_images >= feedback_threshold:
#         # Include the feedback data in the training set
#         retrain_model()
#         # Clear feedback data after retraining
#         shutil.rmtree('feedback_data')
#         os.makedirs('feedback_data', exist_ok=True)

# def retrain_model():
#     # Implement retraining logic
#     # This includes data augmentation, updating the training generator,
#     # re-compiling the model, and fitting the model
#     # Remember to save the model after retraining
#     pass

# @app.route('/', methods=['GET', 'POST'])
# def upload_and_predict():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             return 'No file part'
#         file = request.files['file']
#         if file.filename == '':
#             return 'No selected file'
#         if file:
#             img_path = os.path.join('static', 'uploads', file.filename)
#             file.save(img_path)
#             predicted_label, confidence_scores = predict_image(img_path)
#             # Pass class_labels to the template
#             return render_template(
#                 'result.html',
#                 image=file.filename,
#                 predicted_label=predicted_label,
#                 confidence_scores=confidence_scores,
#                 class_labels=class_labels  # Ensure this is included
#             )
#     return render_template('index.html')

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     correct_label = request.form['correct_label']
#     img_filename = request.form['image']
#     img_path = os.path.join('static', 'uploads', img_filename)
#     collect_feedback(img_path, correct_label)
#     check_and_retrain(feedback_threshold=10)
#     return 'Feedback received. Thank you!'

# if __name__ == '__main__':
#     app.run(debug=True)


## Streamlit 

# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from datetime import datetime
import json

# Load the trained model
model = tf.keras.models.load_model('pneumonia_classifier_final.keras')

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

