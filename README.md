# *PneumoScan: Pneumonia Detection App*
## *Table of Contents*

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Retraining Mechanism](#retraining-mechanism)
- [Chatbot Integration](#chatbot-integration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## *Features*

- *Image Upload*: Users can upload chest X-ray images in JPG or PNG format.
- *Pneumonia Detection*: The app predicts whether the uploaded image indicates Normal lungs, Pneumonia (Bacterial or Viral), or COVID-19 infection.
- *Confidence Scores*: Displays confidence scores for each class in a bar chart.
- *User Feedback*: Allows users to provide feedback if the prediction is incorrect, improving the model over time.
- *Automated Retraining*: The model retrains automatically after receiving a certain number of feedback entries.
- *Interactive Chatbot*: Integrated chatbot (PneumoBot) to answer user queries using both an FAQ and GPT-3 powered responses.
- *Visualizations*: Provides visual insights like Grad-CAM heatmaps (if implemented), confusion matrices, and ROC curves.
- *Customizable Theme*: The app has a professional and user-friendly interface with customizable themes.

