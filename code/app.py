# # Import necessary libraries
# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import os
# from datetime import datetime
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import openai
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from sklearn.preprocessing import label_binarize
# import base64  # For encoding the background image

# # Set page configuration (must be first Streamlit command)
# st.set_page_config(page_title="PneumoScan", page_icon="🩺")

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set
# # Function to set the background image with reduced opacity
# def set_background(image_path, opacity=0.5):
#     with open(image_path, "rb") as file:
#         encoded_string = base64.b64encode(file.read()).decode("utf-8")  # Corrected encoding
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), 
#                         url('data:image/png;base64,{encoded_string}');
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Call the function with the relative path to your image
# set_background("header_image.png", opacity=0.5)


# # Define constants
# CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
# IMG_HEIGHT, IMG_WIDTH = 224, 224

# # Load the model once
# model = tf.keras.models.load_model('pneumonia_classifier_final.keras')

# # Create tabs
# tab1, tab2, tab3 = st.tabs(["Home", "Results", "Chatbot"])

# # Tab 1: Home (Image Upload and Prediction)
# with tab1:
#     # st.title("Pneumonia Detection")
#     # st.write("This application is designed to assist in detecting pneumonia from medical imaging data.")
    
#     # st.header("Upload an Image")
#     # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     # Custom CSS for black text color
#     st.markdown(
#         """
#         <style>
#         .black-text {
#             color: black;
#             font-size: 20px; /* Optional: Adjust text size */
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#         # Applying the custom style to text
#     st.markdown('<h1 class="black-text">Pneumonia Detection</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="black-text">This application is designed to assist in detecting pneumonia from medical imaging data.</p>', unsafe_allow_html=True)

#     st.markdown('<h2 class="black-text">Upload an Image</h2>', unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Save the uploaded file to a temporary directory
#         temp_dir = 'temp_uploads'
#         os.makedirs(temp_dir, exist_ok=True)
#         temp_file_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(temp_file_path, 'wb') as f:
#             f.write(uploaded_file.getbuffer())

#         # Load the image
#         image = Image.open(temp_file_path)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Preprocess the image
#         img = image.resize((IMG_WIDTH, IMG_HEIGHT))
#         img = img.convert('RGB')  # Ensure image has 3 color channels
#         img_array = np.array(img) / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make a prediction
#         predictions = model.predict(img_array)
#         predicted_class = np.argmax(predictions, axis=1)
#         predicted_label = CLASS_LABELS[predicted_class[0]]
#         confidence_scores = predictions[0]

#         # Display the prediction results
#         st.write(f"**Predicted Class:** {predicted_label}")
#         st.write("**Confidence Scores:**")
#         for idx, score in enumerate(confidence_scores):
#             st.write(f"{CLASS_LABELS[idx]}: {score:.4f}")

#         # Feedback section
#         st.write("### Is the prediction incorrect? Please provide the correct label:")
#         correct_label = st.selectbox("Select the correct label:", CLASS_LABELS)
#         submit_feedback = st.button("Submit Feedback")

#         if submit_feedback:
#             # Save the feedback
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             feedback_dir = 'feedback_data'
#             os.makedirs(feedback_dir, exist_ok=True)
#             feedback_data = {
#                 'timestamp': timestamp,
#                 'original_filename': uploaded_file.name,
#                 'predicted_label': predicted_label,
#                 'correct_label': correct_label,
#                 'confidence_scores': confidence_scores.tolist()
#             }
#             # Save the image and feedback
#             feedback_image_dir = os.path.join(feedback_dir, correct_label)
#             os.makedirs(feedback_image_dir, exist_ok=True)
#             feedback_image_path = os.path.join(feedback_image_dir, f"{timestamp}_{uploaded_file.name}")
#             image.save(feedback_image_path)
#             feedback_json_path = os.path.join(feedback_dir, f"{timestamp}_{uploaded_file.name}.json")
#             with open(feedback_json_path, 'w') as f:
#                 json.dump(feedback_data, f)
#             st.write("Thank you for your feedback!")

#         # Clean up the temporary file
#         os.remove(temp_file_path)

# # Tab 2: Results (Confidence Scores and Visualization)
# with tab2:
#     st.header("Prediction Results")
#     if uploaded_file is not None:
#         # Display confidence scores as a bar chart
#         confidence_df = pd.DataFrame({
#             'Class': CLASS_LABELS,
#             'Confidence': confidence_scores
#         })
#         st.bar_chart(confidence_df.set_index('Class'))

#         # Note: The following code requires test data which is not defined.
#         # If you have test data and a test generator, you can uncomment and adjust the code accordingly.

#         # # Generate predictions on test data
#         # y_pred = model.predict(test_generator)
#         # y_true = test_generator.classes
#         # cm = confusion_matrix(y_true, y_pred.argmax(axis=1))

#         # # Plot confusion matrix
#         # fig, ax = plt.subplots()
#         # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
#         # ax.set_xlabel('Predicted')
#         # ax.set_ylabel('Actual')
#         # st.pyplot(fig)

#         # # Binarize the labels for multi-class ROC
#         # y_true_binarized = label_binarize(y_true, classes=range(len(CLASS_LABELS)))

#         # # Compute ROC curve and ROC area for each class
#         # fig, ax = plt.subplots()
#         # for i in range(len(CLASS_LABELS)):
#         #     fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
#         #     roc_auc = auc(fpr, tpr)
#         #     ax.plot(fpr, tpr, label=f'{CLASS_LABELS[i]} (AUC = {roc_auc:.2f})')

#         # ax.plot([0, 1], [0, 1], 'k--')
#         # ax.set_xlabel('False Positive Rate')
#         # ax.set_ylabel('True Positive Rate')
#         # ax.set_title('Receiver Operating Characteristic')
#         # ax.legend(loc='lower right')
#         # st.pyplot(fig)
#     else:
#         st.write("Please upload an image to see prediction results.")

# # Tab 3: Chatbot
# # Define the FAQ chatbot function
# def chatbot_response(user_input):
#     faq = {
#         'What does my result mean?': 'Your result indicates the model\'s prediction based on the uploaded image.',
#         'How accurate is the model?': 'The model has an accuracy of XX% on the test dataset.',
#         'What should I do next?': 'Please consult a medical professional for further advice.',
#         # Add more FAQs as needed
#     }
#     return faq.get(user_input.strip(), 'I\'m sorry, I do not have an answer to that question.')

# # Define the GPT-3 chatbot function
# def gpt_chatbot(user_input):
#     try:
#         response = openai.Completion.create(
#             engine='text-davinci-003',
#             prompt=f"You are a helpful assistant specialized in medical imaging. {user_input}",
#             max_tokens=150,
#             n=1,
#             stop=None,
#             temperature=0.7,
#         )
#         return response.choices[0].text.strip()
#     except Exception as e:
#         return "Sorry, I am unable to process your request at the moment."

# # Create the tab for the chatbot in Streamlit
# with tab3:
#     st.header("Chat with PneumoBot")

#     # Option to select chatbot type
#     chatbot_type = st.radio("Choose a chatbot:", ("FAQ Bot", "GPT-3 Bot"))

#     user_input = st.text_input("Ask a question:")
#     if user_input:
#         if chatbot_type == "FAQ Bot":
#             response = chatbot_response(user_input)
#         else:
#             response = gpt_chatbot(user_input)
#         st.write(f"PneumoBot: {response}")
# app.py

# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import base64
import subprocess  # For running retraining.py

# Set page configuration (must be first Streamlit command)
st.set_page_config(page_title="PneumoScan", page_icon="🩺")

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

# Function to set the background image with reduced opacity
def set_background(image_path, opacity=0.5):
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

# Call the function with the relative path to your image
set_background("header_image.png", opacity=0.5)

# Define constants
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load the model once
model = tf.keras.models.load_model('pneumonia_classifier_final.keras')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Home", "Results", "Chatbot"])

# Tab 1: Home (Image Upload and Prediction)
with tab1:
    # Custom CSS for black text color
    st.markdown(
        """
        <style>
        .black-text {
            color: black;
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Applying the custom style to text
    st.markdown('<h1 class="black-text">Pneumonia Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="black-text">This application is designed to assist in detecting pneumonia from medical imaging data.</p>', unsafe_allow_html=True)

    st.markdown('<h2 class="black-text">Upload an Image</h2>', unsafe_allow_html=True)
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

            # Count the number of feedback entries
            feedback_count = sum([len(files) for r, d, files in os.walk(feedback_dir)])
            st.write(f"Feedback entries received: {feedback_count}")

            # Trigger retraining if 10 feedback entries are collected
            if feedback_count >= 10:
                st.write("Retraining the model with new feedback data...")
                # Run retraining.py
                result = subprocess.run(['python', 'retraining.py'], capture_output=True, text=True)
                if result.returncode == 0:
                    st.write("Model retrained successfully.")
                    # Reload the updated model
                    model = tf.keras.models.load_model('pneumonia_classifier_updated.keras')
                    st.write("Updated model loaded.")
                    # Clear feedback data
                    shutil.rmtree(feedback_dir)
                    st.write("Feedback data cleared.")
                else:
                    st.write("Error during retraining:")
                    st.write(result.stderr)

        # Clean up the temporary file
        os.remove(temp_file_path)

# Tab 2: Results (Confidence Scores and Visualization)
with tab2:
    st.header("Prediction Results")
    if uploaded_file is not None:
        # Display confidence scores as a bar chart
        confidence_df = pd.DataFrame({
            'Class': CLASS_LABELS,
            'Confidence': confidence_scores
        })
        st.bar_chart(confidence_df.set_index('Class'))
    else:
        st.write("Please upload an image to see prediction results.")

# Tab 3: Chatbot
# Define the FAQ chatbot function
def chatbot_response(user_input):
    faq = {
        'What does my result mean?': 'Your result indicates the model\'s prediction based on the uploaded image.',
        'How accurate is the model?': 'The model has an accuracy of XX% on the test dataset.',
        'What should I do next?': 'Please consult a medical professional for further advice.',
        # Add more FAQs as needed
    }
    return faq.get(user_input.strip(), 'I\'m sorry, I do not have an answer to that question.')

# Define the GPT-3 chatbot function
def gpt_chatbot(user_input):
    try:
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=f"You are a helpful assistant specialized in medical imaging. {user_input}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return "Sorry, I am unable to process your request at the moment."

# Create the tab for the chatbot in Streamlit
with tab3:
    st.header("Chat with PneumoBot")

    # Option to select chatbot type
    chatbot_type = st.radio("Choose a chatbot:", ("FAQ Bot", "GPT-3 Bot"))

    user_input = st.text_input("Ask a question:")
    if user_input:
        if chatbot_type == "FAQ Bot":
            response = chatbot_response(user_input)
        else:
            response = gpt_chatbot(user_input)
        st.write(f"PneumoBot: {response}")
