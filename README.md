# Age, Gender, and Emotion Detection Using CNN
![Python](https://img.shields.io/badge/python-v3.8-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.5-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview
This project utilizes a Convolutional Neural Network (CNN) to detect age, gender, and emotions from human facial images. It is trained on the **UTK Face Dataset**, a widely-used dataset for age, gender, and race prediction.

## Features
- **Age Prediction**: Predicts the age range from facial images.
- **Gender Classification**: Distinguishes between male and female.
- **Emotion Detection**: Classifies common emotions (happy, sad, neutral, etc.).

## Dataset
- Facial-age & UTKFace for Age Detection
- UTK Faces for Gender Detection
- CT+ for Emotion Detection
- **Preprocessing**: Images resized to 48x48 pixels, converted to grayscale for CNN input.

## Model Architecture
The CNN model is composed of:
- Input Conv2D layer with 32 filters, followed by MaxPooling
- 3 Conv2D layers (64, 128, 256 filters) with MaxPooling layers
- Dense layer with 128 nodes
- Output layer with softmax activation for emotion classification

## Performance Matrics
- **Age Prediction Accuracy: 82%
- **Gender Classification Accuracy: 90%
- **Emotion Detection Accuracy: 97%

