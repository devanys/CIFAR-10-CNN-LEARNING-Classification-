# CIFAR-10 Image Classification with CNN
![image](https://github.com/user-attachments/assets/95f04797-9940-4f5a-9279-e48661e5a14e)

This project implements a **Convolutional Neural Network (CNN)** to classify images from the CIFAR-10 dataset into one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The project is written in **Python** using **TensorFlow** and includes visualization of training progress and predictions.

## Features

- **Data Preprocessing:** Normalizes image pixel values to the range [0, 1].
- **Visualization:** Displays sample images and their labels from the CIFAR-10 dataset.
- **Model Architecture:** A simple CNN with two convolutional layers, max-pooling, and fully connected layers.
- **Training:** Trains the CNN with training data and validates with test data.
- **Evaluation:** Displays accuracy and loss metrics during training and testing.
- **Prediction:** Makes predictions on test images and visualizes the results.

## Dataset

**CIFAR-10** is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is split into 50,000 training images and 10,000 test images.

## Requirements

- **Python 3.7+**
- **TensorFlow 2.x**
- **NumPy**
- **Matplotlib**
