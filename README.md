Project Overview (STAR Method)
S – Situation

Medical image analysis and image classification tasks require automated systems to reduce manual effort and improve accuracy. Traditional manual analysis is time-consuming and prone to human error, especially when handling large volumes of images.

T – Task

The objective of this project was to design and implement a CNN-based image classification system capable of classifying images accurately and to deploy the trained model using a web interface so users can upload images and get predictions in real time.

A – Action

Collected and organized image datasets for classification

Preprocessed images using resizing, normalization, and augmentation

Designed and trained a Convolutional Neural Network (CNN) using TensorFlow & Keras

Evaluated the model using accuracy metrics

Saved the trained model (model.h5) for reuse

Integrated the model with a Flask web application

Built a frontend interface to upload images and display predictions

R – Result

Successfully built a CNN model with strong classification accuracy

Enabled real-time predictions through a web interface

Created an end-to-end deep learning application combining model training + deployment

Improved usability by allowing non-technical users to interact with the model

🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy

Flask

HTML

🚀 How to Run the Project
pip install -r requirements.txt
python app.py


Open in browser:

http://127.0.0.1:5000/

