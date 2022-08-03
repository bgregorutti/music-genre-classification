"""
Define the Keras model objects
"""

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model

def convolutional(input_shape, nr_classes, padding="valid"):
    """
    TODO
    """
    input_layer = Input(input_shape)

    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding=padding)(input_layer)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding=padding)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(.3)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    output_layer = Dense(nr_classes, activation="softmax")(x)

    model = Model(input_layer, output_layer)
    model.summary()
    return model

def deep_convolutional(input_shape, nr_classes, filters=16, nr_layers=2):
    """
    TODO
    """
    input_layer = Input(input_shape)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(.2)(x)
    for _ in range(nr_layers):
        filters *= 2
        x = Conv2D(filters=filters, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Dropout(.2)(x)

    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(.5)(x)
    output_layer = Dense(nr_classes, activation="softmax")(x)

    model = Model(input_layer, output_layer)
    model.summary()
    return model
