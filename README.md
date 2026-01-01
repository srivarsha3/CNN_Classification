🐱🐶 Cat vs Dog Image Classification using CNN (TensorFlow & Keras)

This project implements a Convolutional Neural Network (CNN) to classify images as cats or dogs using TensorFlow and Keras. The model is trained on image data stored in CSV format, reshaped into image tensors, normalized, and enhanced using data augmentation for better generalization.

📌 Project Overview

Goal: Build a deep learning model that can accurately classify images into two categories: Cat or Dog

Approach: Convolutional Neural Network (CNN)

Framework: TensorFlow (Keras API)

Image Size: 100 × 100 RGB images

Output: Binary classification (0 → Dog, 1 → Cat)

🧠 Model Architecture

The CNN model consists of:

Convolutional Layers (Conv2D):

32, 64, and 128 filters

Kernel size: 3×3

Activation: ReLU

MaxPooling Layers: Reduce spatial dimensions

Dropout Layers: Prevent overfitting

Fully Connected Layers (Dense):

Hidden layer with ReLU

Output layer with Sigmoid activation

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

🔄 Data Preprocessing

Data loaded from CSV files:

input.csv, labels.csv

input_test.csv, labels_test.csv

Reshaped into (samples, 100, 100, 3)

Normalized pixel values to range [0, 1]

Labels reshaped for binary classification

🔁 Data Augmentation

To improve model robustness, ImageDataGenerator is used with:

Rotation

Width & height shifts

Zooming

Horizontal flipping

This helps the model generalize better to unseen images.

🚀 Model Training

Batch Size: 32

Epochs: 30

Training Method: datagen.flow()

Validation: Separate test dataset

The model learns from augmented images while validating on test data.

📊 Evaluation & Prediction

Model performance evaluated on test data

Random test image displayed with prediction

Prediction threshold:

0 → Dog

1 → Cat

💾 Model Saving

After training, the model is saved as:

model.h5


This allows reuse of the trained model without retraining.

🛠 Technologies Used

Python

NumPy

Matplotlib

TensorFlow / Keras

CNN (Deep Learning)

📌 Conclusion

This project demonstrates how to build an end-to-end image classification pipeline using CNNs, including preprocessing, augmentation, training, evaluation, prediction, and model persistence. It’s ideal for beginners learning deep learning and computer vision.
