"""
Train a CNN from the spectrogram images
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

from genre.evaluate import evaluate
from genre.features_extraction import multichannel_spectrogram, label_mapping

def get_features(wav_data_path, np_data_path, precompute=False):
    """
    Get the features. If precompute is True, compute the spectrograms features and store the 'images'.

    Args:
        wav_data_path: folder containing the WAV files
        np_data_path: folder containing the resulted numpy arrays
        precompute: if True, compute the spectrograms. Default: False
    
    Returns:
        Train, test and validation data
        The labels
    """
    if precompute:
        # Load the wav files
        dataset, targets = multichannel_spectrogram(wav_data_path)

        try:
            n, img_rows, img_cols, n_channels = dataset.shape
        except:
            n, img_rows, img_cols = dataset.shape
            dataset = dataset.reshape(n, img_rows, img_cols, 1)

        input_shape = (img_rows, img_cols, n_channels)
        print('data shape: {}'.format(dataset.shape))
        print('input shape: {}'.format(input_shape))

        # Get the targets
        labels = np.unique(targets)

        # categorize the target
        targets = to_categorical(targets, num_classes=len(label_mapping))

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.2)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        np.save(Path(np_data_path, "x_train.npy"), X_train)
        np.save(Path(np_data_path, "y_train.npy"), y_train)
        np.save(Path(np_data_path, "x_test.npy"), X_test)
        np.save(Path(np_data_path, "y_test.npy"), y_test)
        np.save(Path(np_data_path, "x_val.npy"), X_val)
        np.save(Path(np_data_path, "y_val.npy"), y_val)
        np.save(Path(np_data_path, "labels.npy"), labels)

    else:
        X_train = np.load(Path(np_data_path, "x_train.npy"))
        y_train = np.load(Path(np_data_path, "y_train.npy"))
        X_test = np.load(Path(np_data_path, "x_test.npy"))
        y_test = np.load(Path(np_data_path, "y_test.npy"))
        X_val = np.load(Path(np_data_path, "x_val.npy"))
        y_val = np.load(Path(np_data_path, "y_val.npy"))
        labels = np.load(Path(np_data_path, "labels.npy"))
        
    print(f"Train data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    print(f"Test data: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test, X_val, y_val, labels

def run_generator(model, train_generator, val_generator, X_test, y_test):
    """
    Run the model training using the data generator

    Args:
        model: a Keras model
        train_generator, val_generator: two object of class genre.data_generator.Generator
        X_test, y_test: numpy arrays
    """
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss")
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[reduce_lr, early_stopping])
    evaluate(model, X_test, y_test, label_mapping.keys(), history)

def run(model, X_train, y_train, X_test, y_test, X_val, y_val, labels):
    """
    Run the model training

    Args:
        model: a Keras model
        X_train, y_train, X_test, y_test, X_val, y_val, labels: numpy arrays
    """
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # TODO check the loss function. Should be categorical_crossentropy
    reduce_lr = ReduceLROnPlateau(monitor="val_loss")
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[reduce_lr, early_stopping])
    evaluate(model, X_test, y_test, labels, history)
