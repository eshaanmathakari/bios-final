# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from datetime import datetime
import json
import pandas as pd
import base64
import threading
import openai
from retraining import retrain_model, count_feedback_images, RETRAIN_THRESHOLD

# Clear Streamlit cache
st.cache_resource.clear()
st.cache_data.clear()

# Set page configuration
st.set_page_config(page_title="PneumoScan", page_icon="🩺")

# Constants
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
IMG_HEIGHT, IMG_WIDTH = 224, 224
FEEDBACK_DIR = 'feedback_data'

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

# Function to set background
def set_background(image_path, opacity=0.5):
    if os.path.exists(image_path):
        with open(image_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), 
                            url('data:image/png;base64,{encoded_string}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Call the background setup
set_background("header_image.png", opacity=0.5)

# Load the model
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
        st.write(f"Model loaded successfully from '{model_path}'.")
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        st.stop()

model = load_model()

# Track retraining status
retraining_status = {"in_progress": False}

# Function to start retraining
def start_retraining():
    if not retraining_status["in_progress"]:
        retraining_status["in_progress"] = True
        st.write("Retraining the model with feedback data...")
        def retrain_and_update():
            retrain_model()
            retraining_status["in_progress"] = False
            # Reload the updated model
            global model
            model = load_model()
            st.write("Model retrained and updated.")
        threading.Thread(target=retrain_and_update).start()
    else:
        st.write("Retraining is already in progress. Please wait.")

# Tabs for the app
tab1, tab2, tab3 = st.tabs(["Home", "Results", "Chatbot"])

# Tab 1: Home
with tab1:
    st.markdown('<h1 style="color:black;">Pneumonia Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:black;">This application assists in detecting pneumonia from medical imaging data.</p>', unsafe_allow_html=True)
    
    # Upload Section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Save uploaded file
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and preprocess the image
        image = Image.open(temp_path).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = CLASS_LABELS[predicted_class[0]]
        confidence_scores = predictions[0]

        # Display the image and results
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"**Predicted Class:** {predicted_label}")
        for idx, score in enumerate(confidence_scores):
            st.write(f"{CLASS_LABELS[idx]}: {score:.4f}")

        # Feedback Section
        st.write("### Is the prediction incorrect? Please provide the correct label:")
        correct_label = st.selectbox("Select the correct label:", CLASS_LABELS)
        submit_feedback = st.button("Submit Feedback")

        if submit_feedback:
            # Save feedback
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feedback_class_dir = os.path.join(FEEDBACK_DIR, correct_label)
            os.makedirs(feedback_class_dir, exist_ok=True)
            feedback_path = os.path.join(feedback_class_dir, f"{timestamp}_{uploaded_file.name}")
            image.save(feedback_path)
            st.write("Feedback submitted successfully.")

            # Check feedback count
            feedback_count = count_feedback_images()
            st.write(f"Total feedback entries: {feedback_count}")

            if feedback_count >= RETRAIN_THRESHOLD:
                start_retraining()
            else:
                st.write(f"Need at least {RETRAIN_THRESHOLD} feedback images to retrain the model.")

# Tab 2: Results
with tab2:
    st.header("Prediction Results")
    if uploaded_file:
        confidence_df = pd.DataFrame({
            'Class': CLASS_LABELS,
            'Confidence': confidence_scores
        })
        st.bar_chart(confidence_df.set_index('Class'))
    else:
        st.markdown('<p style="color:black;">Please upload an image to see prediction results.</p>', unsafe_allow_html=True)

# Tab 3: Chatbot
def chatbot_response(user_input):
    faq = {
        'What does my result mean?': 'Your result indicates the model\'s prediction based on the uploaded image.',
        'How accurate is the model?': 'The model has an accuracy of XX% on the test dataset.',
        'What should I do next?': 'Please consult a medical professional for further advice.',
    }
    return faq.get(user_input.strip(), 'I\'m sorry, I do not have an answer to that question.')

def gpt_chatbot(user_input):
    try:
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=f"You are a helpful assistant specialized in medical imaging. {user_input}",
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception:
        return "Sorry, I am unable to process your request at the moment."

with tab3:
    st.header("Chat with PneumoBot")
    chatbot_type = st.radio("Choose a chatbot:", ("FAQ Bot", "GPT-3 Bot"))
    user_input = st.text_input("Ask a question:")
    if user_input:
        if chatbot_type == "FAQ Bot":
            st.markdown(f"<p style='color:black;'>PneumoBot: {chatbot_response(user_input)}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:black;'>PneumoBot: {gpt_chatbot(user_input)}</p>", unsafe_allow_html=True)
