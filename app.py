from flask import Flask, request, render_template, redirect, url_for
import os


app = Flask(__name__, static_url_path='/static')

# Update the image save path
img_path = os.path.join('static', 'uploads', file.filename)
file.save(img_path)


# Ensure the 'uploads' and 'feedback_data' directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('feedback_data', exist_ok=True)

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
            img_path = os.path.join('uploads', file.filename)
            file.save(img_path)
            predicted_label, confidence_scores = predict_image(img_path)
            # Render the result template
            return render_template('result.html', image=file.filename, predicted_label=predicted_label, confidence_scores=confidence_scores)
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    correct_label = request.form['correct_label']
    img_filename = request.form['image']
    img_path = os.path.join('uploads', img_filename)
    collect_feedback(img_path, correct_label)
    check_and_retrain(feedback_threshold=10)
    return 'Feedback received. Thank you!'

if __name__ == '__main__':
    app.run(debug=True)
