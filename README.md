# Table of Contents

1. [MNIST Digit Classification with CNN](#mnist-digit-classification-with-cnn)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Preprocessing](#data-preprocessing)
5. [Training](#training)
6. [Results](#results)
7. [Usage](#usage)
8. [Dependencies](#dependencies)

# MNIST Digit Classification with CNN
This is a Convolutional Neural Network (CNN) model for digit classification using the MNIST dataset. The model was built using TensorFlow and Keras.

# Dataset
The MNIST dataset is a collection of handwritten digits, commonly used as a benchmark for image classification tasks. The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels.

# Model Architecture
The CNN model has the following architecture:

* Input layer of shape (28, 28, 1)
* Convolutional layer with 32 filters, kernel size of (3,3), and ReLU activation
* MaxPooling layer with pool size of (2,2)
* Convolutional layer with 64 filters, kernel size of (3,3), and ReLU activation
* MaxPooling layer with pool size of (2,2)
* Convolutional layer with 64 filters, kernel size of (3,3), and ReLU activation
* Flatten layer
* Dense layer with 64 units and ReLU activation
* Output layer with 10 units and Softmax activation

# Data Preprocessing
The input images were normalized by dividing each pixel value by 255.0 to scale the values between 0 and 1. The training labels were one-hot encoded using the to_categorical function from Keras.

# Training
The model was trained for 5 epochs using the Adam optimizer and categorical cross-entropy loss. The training accuracy was around 99%, while the test accuracy was around 99%.

# Results
The training and validation accuracy were plotted against the number of epochs using Matplotlib. The model was then used to make predictions on 10 test images, which were also plotted alongside their predicted labels.

# Usage
To use the model for prediction, you can load it using Keras' load_model function, preprocess your image using the same steps as the training data, and then use the predict method of the loaded model to obtain the predicted label.

# Dependencies
* TensorFlow 2.x
* Keras
* NumPy
* Matplotlib
* Pillow
