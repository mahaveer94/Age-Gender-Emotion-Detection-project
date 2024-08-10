from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained models
age_model = load_model('age_model_pretrained.h5')
gender_model = load_model('gender_model_pretrained.h5')
emotion_model = load_model('emotion_model_pretrained.h5')

# Function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to postprocess prediction
def postprocess_prediction(prediction):
    return int(np.argmax(prediction, axis=1)[0])

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']

    try:
        img = Image.open(io.BytesIO(file.read()))
    except:
        return jsonify({"error": "Invalid image format"})

    # Preprocess image for each model
    age_img = preprocess_image(img, target_size=(64, 64))  # Example target size
    gender_img = preprocess_image(img, target_size=(64, 64))
    emotion_img = preprocess_image(img, target_size=(48, 48))

    # Make predictions
    age_prediction = age_model.predict(age_img)
    gender_prediction = gender_model.predict(gender_img)
    emotion_prediction = emotion_model.predict(emotion_img)

    # Post-process predictions
    age_result = postprocess_prediction(age_prediction)
    gender_result = postprocess_prediction(gender_prediction)
    emotion_result = postprocess_prediction(emotion_prediction)

    # Prepare the response
    result = {
        'age': age_result,
        'gender': 'Male' if gender_result == 0 else 'Female',
        'emotion': ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised'][emotion_result]
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
