# American-Sign-Language-Detection-with-PyTorch
This project is an implementation of a Convolutional Neural Network (CNN) for American Sign Language detection using PyTorch. The goal of this project is to train a CNN model that can accurately detect American Sign Language gestures from images.

## Dataset
The dataset used in this project is the American Sign Language Hand Dataset from kaggle. This dataset contains over 27,000 images of American Sign Language gestures from 24 different classes. The dataset is split into training and testing sets.

## Model Architecture
The CNN model used in this project consists of 5 convolutional layers followed by a classifier layer. The model was trained using the Adam optimizer and the cross-entropy loss function.

## Usage
To run the project, you can clone the repository and run the main.py script. The script will train the model on the training set and evaluate its performance on the testing set. You can modify the number of epochs, batch size, and other hyperparameters in the config.py file.

## Results
After training the model for 5 epochs, we achieved a testing accuracy of 93%. The model is able to accurately classify American Sign Language gestures from images with a high degree of accuracy.

## Credits
This project was developed by [Sayyad Abdul Bari]. The code is based on the PyTorch tutorial on CNNs and adapted to the American Sign Language Hand Dataset.

References
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
