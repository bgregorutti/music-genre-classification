"""
Train MLP model on well choesen audio features
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from genre.evaluate import evaluate
from genre.audio_features import static_features


def get_model(nr_classes):
    """
    Get the Keras model

    Args:
        nr_classes: number of classes
    
    Returns:
        An object of class tf.keras.Model
    """

    units = 512
    nr_layers = 4

    input_layer = Input((X_train.shape[1],))
    x = Dense(units, activation='relu')(input_layer)
    x = Dropout(.2)(x)
    for _ in range(nr_layers):
        units /= 2
        x = Dense(units, activation='relu')(x)
        x = Dropout(.2)(x)

    output_layer = Dense(nr_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)
    model.summary()
    return model

def get_features():
    """
    Compute the audio features
    """

    path = Path("data.csv")

    if path.exists():
        data = pd.read_csv('data.csv')
    else:
        data = static_features()
        data.to_csv('data.csv')
        
    # Dropping unneccesary columns
    data = data.drop(['filename'],axis=1)

    # Label encoder
    y_data = data.label
    labels = np.unique(y_data)
    y_data = LabelEncoder().fit_transform(y_data)

    # Standard scaling
    scaler = StandardScaler()
    X_data = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    print(f"Train data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    print(f"Test data: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test, X_val, y_val, labels

def run(model, X_train, y_train, X_test, y_test, X_val, y_val, labels):
    """
    Main training function, evaluate and plot the results

    Args:
        model: a Keras model
        X_train, y_train, X_test, y_test, X_val, y_val: Train, Test and Validation data
        labels: the actual labels
    """

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss")
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[reduce_lr, early_stopping], batch_size=10)
    evaluate(model, X_test, y_test, labels, history)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_val, y_val, labels = get_features()
    model = get_model(nr_classes=len(labels))
    run(model, X_train, y_train, X_test, y_test, X_val, y_val, labels)
