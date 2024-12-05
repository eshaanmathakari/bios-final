# *PneumoScan: Pneumonia Detection App*
## *Table of Contents*

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Retraining Mechanism](#retraining-mechanism)
- [Chatbot Integration](#chatbot-integration)
- [Contributing](#contributing)
- [License](#license)


## *Features*

- *Image Upload*: Users can upload chest X-ray images in JPG or PNG format.
- *Pneumonia Detection*: The app predicts whether the uploaded image indicates Normal lungs, Pneumonia (Bacterial or Viral), or COVID-19 infection.
- *Confidence Scores*: Displays confidence scores for each class in a bar chart.
- *User Feedback*: Allows users to provide feedback if the prediction is incorrect, improving the model over time.
- *Automated Retraining*: The model retrains automatically after receiving a certain number of feedback entries.
- *Interactive Chatbot*: Integrated chatbot (PneumoBot) to answer user queries using both an FAQ and GPT-3 powered responses.
- *Visualizations*: Provides visual insights like Grad-CAM heatmaps (if implemented), confusion matrices, and ROC curves.
- *Customizable Theme*: The app has a professional and user-friendly interface with customizable themes.
## *Installation*
### *Clone the Repository*
```bash
git clone https://github.com/yourusername/pneumoscan.git
cd pneumoscan
```

```bash
conda create -n pneumoscan_env python=3.8
conda activate pneumoscan_env
```

```bash
python -m venv pneumoscan_env
source pneumoscan_env/bin/activate  # On Windows use pneumoscan_env\Scripts\activate
```

### *Create a Virtual Environment*

Using Anaconda:

```bash
conda create -n pneumoscan_env python=3.8
conda activate pneumoscan_env
```

Or using venv:

```bash
python -m venv pneumoscan_env
source pneumoscan_env/bin/activate  # On Windows use pneumoscan_env\Scripts\activate
```
### Install Dependencies

```bash
Copy code
pip install -r requirements.txt
```
### Set Up OpenAI API Key

```bash
set OPENAI_API_KEY='your-openai-api-key-here'
```
## Usage
### Running the App
```bash
streamlit run app.py
```
The app will open in your default web browser at http://localhost:8501.

### Using the App

Upload an Image: Click on the "Browse files" button or drag and drop an image.
View Predictions: The app will display the predicted class and confidence scores.
Provide Feedback: If the prediction is incorrect, select the correct label and submit feedback.
Interact with PneumoBot: Navigate to the "Chatbot" tab to ask questions.

### Model Architecture

The app uses a convolutional neural network based on the DenseNet121 architecture:

- Base Model: DenseNet121 with pre-trained weights.
- Custom Layers: Global Average Pooling, Dropout, and a Dense output layer with softmax activation.
- Training: The model is trained on labeled chest X-ray images.

### Retraining Mechanism

- Feedback Collection: User-provided feedback is saved in feedback_data/.
- Automatic Retraining: When feedback entries reach a threshold (e.g., 10), retraining.py is triggered.
- Model Update: The model retrains with the augmented dataset and updates the saved model.
- Model Loading: The app automatically loads the updated model after retraining.

### Chatbot Integration
PneumoBot assists users with their queries:

- FAQ Bot: Provides answers to common questions.
- GPT-3 Bot: Offers detailed responses powered by OpenAI's GPT-3.
- Usage: Accessible under the "Chatbot" tab within the app.

### Contributing
Contributions are welcome! Please follow these steps:

- Fork the Repository
- Create a Feature Branch
```bash
git checkout -b feature/YourFeature
```
- Commit Your Changes
```bash
Copy code
git commit -m "Add your message here"
```
- Push to the Branch
```bash
Copy code
git push origin feature/YourFeature
```
- Open a Pull Request

## License
This project is licensed under the MIT License.

### Disclaimer:
This application is intended for educational purposes and should not be used as a substitute for professional medical advice. Always consult a healthcare professional for medical diagnoses and treatment.





