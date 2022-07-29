from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import math
import librosa

import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

label_mapping = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}


def evaluate(model, X_test, y_test, labels, history):
    
    print(pd.Series(model.evaluate(X_test, y_test), index=model.metrics_names))
        
    plt.figure(figsize=(12,8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    
    probabilities = model.predict(X_test)
    predicted_classes = np.argmax(probabilities, axis=1)
    if y_test.shape[1] == len(labels):
        y_test = np.argmax(y_test, axis=1)

    confMat = pd.DataFrame(confusion_matrix(y_test, predicted_classes), index=labels, columns=labels)
    confMat /= np.sum(confMat, axis=1)

    plt.figure(figsize=(12,8))
    sns.heatmap(confMat, cmap=plt.cm.Blues, annot=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title('Confusion matrix')

    plt.show()

def get_model(input_shape, nr_classes, padding="valid"):
    input_layer = Input(input_shape)

    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding=padding)(input_layer)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding=padding)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(.3)(x)

    # x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding=padding)(x)
    # x = MaxPooling2D((2, 2), padding="same")(x)
    # x = Dropout(.3)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    output_layer = Dense(nr_classes, activation="softmax")(x)

    model = Model(input_layer, output_layer)
    model.summary()
    return model

def compute_mfcc(path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10, amplitude_to_db=False):
    features = []
    labels = []
    sample_rate = 22050
    sample_per_segment =int(sample_rate * 30 / num_segment)
    
    for file_path in Path(path).glob("*.wav"):
        print(f"Track name: {file_path.name}")
        y, sample_rate = librosa.load(file_path, sr=sample_rate)
        label = file_path.stem.split("_")[0]

        h, p = hpss(y)

        for n in range(num_segment):
            segment = y[sample_per_segment*n:sample_per_segment*(n+1)]
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

            if amplitude_to_db:
                mfcc = librosa.amplitude_to_db(np.abs(mfcc))

            mfcc = mfcc.T
            if len(mfcc) == math.ceil(sample_per_segment / hop_length):
                features.append(mfcc.tolist())
                labels.append(label_mapping[label])
    
    return np.array(features), np.array(labels)

def compute_mfcc2(path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10, amplitude_to_db=False):
    features = []
    labels = []
    sample_rate = 22050
    sample_per_segment =int(sample_rate * 30 / num_segment)
    
    for file_path in Path(path).glob("*.wav"):

        print(f"Track name: {file_path.name}")
        y, sample_rate = librosa.load(file_path, sr=sample_rate)
        label = file_path.stem.split("_")[0]

        for n in range(num_segment):
            segment = y[sample_per_segment*n:sample_per_segment*(n+1)]
            
            # Melspectrogram, Harmonic and percusive features, MFCC
            mel = librosa.feature.melspectrogram(y=segment, n_mels=40)
            harmonic, percusive = librosa.decompose.hpss(mel)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

            if amplitude_to_db:
                mfcc = librosa.amplitude_to_db(np.abs(mfcc))

            spectro = np.stack((mfcc.T, harmonic.T, percusive.T), axis=-1)

            if len(spectro) == math.ceil(sample_per_segment / hop_length):
                features.append(spectro.tolist())
                labels.append(label_mapping[label])
    
    return np.array(features), np.array(labels)

def get_features(wav_data_path, np_data_path, precompute=False):
    if precompute:
        
        # Load the wav files
        dataset, targets = compute_mfcc2(wav_data_path)

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

def run(model, X_train, y_train, X_test, y_test, X_val, y_val, labels):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss")
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[reduce_lr, early_stopping])
    evaluate(model, X_test, y_test, labels, history)


if __name__ == "__main__":
    compute_features = False
    wav_data_path = "data/genres_wav"
    np_data_path = "data/genres_np/multi_channel"

    # config()
    X_train, y_train, X_test, y_test, X_val, y_val, labels = get_features(wav_data_path, np_data_path, compute_features)
    model = get_model(input_shape=X_train.shape[1:], nr_classes=len(labels), padding="same")
    run(model, X_train, y_train, X_test, y_test, X_val, y_val, labels)
    model.save("model")