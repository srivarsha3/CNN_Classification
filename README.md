Project Overview (STAR Method)
S – Situation

Image classification is a fundamental computer vision task, but handling large image datasets efficiently can be challenging. In many cases, image data is stored in non-traditional formats such as CSV files, requiring additional preprocessing steps to convert the data into a usable form for deep learning models.

T – Task

The objective of this project was to build a Convolutional Neural Network (CNN) capable of accurately classifying images as cats or dogs, using image data stored in CSV format, and to improve the model’s generalization performance through proper preprocessing and data augmentation techniques.

A – Action

Loaded image data stored in CSV files and reshaped it into image tensors

Normalized pixel values to improve training stability

Applied data augmentation techniques to reduce overfitting and enhance generalization

Designed and trained a CNN model using TensorFlow and Keras

Evaluated the model’s performance on unseen data

Saved the trained model for reuse and deployment

Integrated the model with a Flask-based web interface that allows users to upload an image, preview it, and receive real-time predictions

R – Result

Successfully developed a CNN model that accurately classifies images as cats or dogs

Improved model generalization through normalization and data augmentation

Demonstrated the ability to handle image data stored in CSV format

Delivered an end-to-end solution combining model training, preprocessing, and deployment

🛠️ Technologies Used

Python

TensorFlow & Keras

NumPy

Flask

HTML / CSS
