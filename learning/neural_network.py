# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  
# Author: B. Gregorutti
# Email: baptiste.gregorutti@gmail.com

"""
    Wrappers for neural network architectures.

    Features:
    =========
        - Fully connected
        - Convolutional NN
        - LSTM
"""

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential

def dense_nn(input_shape, nrClasses):
    """
        Fully-connected neural network with four layers
    """

    # model architecture
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nrClasses, activation='softmax'))

    # Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def conv_nn(input_shape, nrClasses):
    """
        Convolutional neural network with two convolutional layers and two dense layers
    """

    # model architecture
    model = Sequential()

    # First convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())

    # Second convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())

    # Max pooling and dropout
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    # Dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Last layer
    model.add(Dense(nrClasses, activation='softmax'))

    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    X, y = load_iris(True)
    nrClasses = len(np.unique(y))
    n, d = X.shape

    model = dense_nn((d,), nrClasses)
    model.fit(X, y, epochs=50, batch_size=128)
    print(model.predict(X[:3]))

