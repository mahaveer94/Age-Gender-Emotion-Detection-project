from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load models
age_model = load_model('age_model_pretrained.h5')
gender_model = load_model('gender_model_pretrained.h5')
emotion_model = load_model('emotion_model_pretrained.h5')

# Define mappings
age_classes = ["0-2", "3-5", "6-12", "13-19", "20-35", "36-40", "41-60"]
gender_classes = ["Male", "Female"]
emotion_classes = ["Angry", "Sad", "Happy", "Surprise", "Neutral"]

def prepare_image(image, target_size):
    """Resize and preprocess the image according to the target size."""
    image = image.resize(target_size)  # Resize the image
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part', 400

        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400

        if file:
            img = Image.open(file)

            # Prepare the image for each model
            age_img = prepare_image(img, (200, 200))  # For age model
            gender_img = prepare_image(img, (100, 100))  # For gender model
            emotion_img = prepare_image(img, (48, 48))  # For emotion model

            # Predict using the models
            age_probs = age_model.predict(age_img)[0]
            gender_probs = gender_model.predict(gender_img)[0]
            emotion_probs = emotion_model.predict(emotion_img)[0]

            # Convert predictions to readable text
            predicted_age = age_classes[np.argmax(age_probs)]
            predicted_gender = gender_classes[np.argmax(gender_probs)]
            predicted_emotion = emotion_classes[np.argmax(emotion_probs)]

            # Process and display the result
            return render_template('result.html', age=predicted_age, gender=predicted_gender, emotion=predicted_emotion)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) #for local 
    #app.run(host='0.0.0.0', port=5000) #for deployment
