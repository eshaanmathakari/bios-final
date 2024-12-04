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
### Running the App
```bash
streamlit run app.py
```
