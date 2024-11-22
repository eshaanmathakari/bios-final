from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil

app = Flask(__name__, static_url_path='/static')

# Load your trained model
model = load_model('pneumonia_classifier_final.keras')

# Define image dimensions and class labels
img_height, img_width = 224, 224
classes = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
class_labels = classes

# Ensure the 'uploads' and 'feedback_data' directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('feedback_data', exist_ok=True)

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_labels[predicted_class[0]]

    # Get confidence scores
    confidence_scores = predictions[0]

    return predicted_label, confidence_scores

def collect_feedback(img_path, correct_label):
    # Create directory for the correct class if it doesn't exist
    class_dir = os.path.join('feedback_data', correct_label)
    os.makedirs(class_dir, exist_ok=True)

    # Copy the image to the feedback directory
    img_name = os.path.basename(img_path)
    dest_path = os.path.join(class_dir, img_name)
    shutil.copy(img_path, dest_path)

    print(f"Feedback collected: {img_name} -> {correct_label}")

def check_and_retrain(feedback_threshold=10):
    # Count the total number of feedback images
    total_feedback_images = sum(
        [len(files) for r, d, files in os.walk('feedback_data')]
    )

    if total_feedback_images >= feedback_threshold:
        # Include the feedback data in the training set
        retrain_model()
        # Clear feedback data after retraining
        shutil.rmtree('feedback_data')
        os.makedirs('feedback_data', exist_ok=True)

def retrain_model():
    # Implement retraining logic
    # This includes data augmentation, updating the training generator,
    # re-compiling the model, and fitting the model
    # Remember to save the model after retraining
    pass

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img_path = os.path.join('static', 'uploads', file.filename)
            file.save(img_path)
            predicted_label, confidence_scores = predict_image(img_path)
            # Render the result template
            return render_template(
                'result.html',
                image=file.filename,
                predicted_label=predicted_label,
                confidence_scores=dict(zip(class_labels, confidence_scores)),
                class_labels=class_labels
            )
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    correct_label = request.form['correct_label']
    img_filename = request.form['image']
    img_path = os.path.join('static', 'uploads', img_filename)
    collect_feedback(img_path, correct_label)
    check_and_retrain(feedback_threshold=10)
    return 'Feedback received. Thank you!'

if __name__ == '__main__':
    app.run(debug=True)
