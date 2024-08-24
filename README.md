# Digit-Recognizer with Neural Networks

This repository contains a machine learning model for recognizing handwritten digits using a Feed Forward Neural Network implemented in Python. The model achieves an accuracy of 97% on the Digit Recognizer competition dataset from Kaggle.

## Overview

The model uses a neural network with multiple hidden layers to classify images of handwritten digits from the MNIST dataset. It employs activation functions like ReLU and Tanh, and uses Adam optimization for training.

## Dataset

The dataset used for training and testing the model is the Digit Recognizer dataset from Kaggle. It includes:

`train.csv`: Training data with labels.
 `test.csv`: Test data without labels.
`sample_submission.csv`: Example submission file.

## Features
**Feed Forward Neural Network** with multiple hidden layers.
**Activation Functions**: ReLU for hidden layers, Sigmoid or Softmax for output layer.
**Optimization**: Adam optimizer with learning rate and other hyperparameters.
**Regularization**: L2 regularization to prevent overfitting.

## Installation

1. Clone the repository:
   bash
   git clone https://github.com/yourusername/digit-recognizer.git

   Navigate to the project directory:
   cd digit-recognizer

   Install required packages:
   pip install -r requirements.txt
 # Usage
Upload kaggle.json containing your Kaggle API credentials to the Colab environment.
Execute the provided Colab notebook or script to download and preprocess the data.
Run the training function to train the model.
Use the test_prediction function to visualize predictions on test data.

# Example
To test the model on a specific image, use the following code:
i = 1
test_prediction(i * 5, parameters)
test_prediction(i * 5+1, parameters)
test_prediction(i * 5+2, parameters)
test_prediction(i * 5+3, parameters)
test_prediction(i * 5+4, parameters)
test_prediction(i * 5+5, parameters)

# Results
The model achieves an accuracy of 97% on the test dataset, demonstrating its effectiveness in digit recognition.

