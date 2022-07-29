from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from extract_spectrogram import spectrogram
from Generator import DataGenerator, label_mapping, load_data

import tensorflow as tf

def config():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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

def get_model(input_shape, nr_classes, filters=16, nr_layers=2):
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

def get_features(wav_data_path, np_data_path, precompute=False):
    if precompute:
        
        # Load the wav files
        dataset = np.array([spectrogram(path) for path in Path(wav_data_path).glob("*.wav")])
        n, img_rows, img_cols = dataset.shape
        dataset = dataset.reshape(n, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        print('data shape: {}'.format(dataset.shape))
        print('input shape: {}'.format(input_shape))

        # Get the targets
        targets = np.array([path.stem.split("_")[0] for path in  Path(wav_data_path).glob("*.wav")])
        labels = np.unique(targets)

        # categorize the target
        targets = LabelEncoder().fit_transform(targets)
        #targets = to_categorical(targets, num_classes=nr_classes)

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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss")
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val),
                        callbacks=[reduce_lr, early_stopping])
    evaluate(model, X_test, y_test, labels, history)

def run_generator(model, train_generator, val_generator, X_test, y_test):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss")
    early_stopping = EarlyStopping(monitor="loss", patience=3)
    history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[reduce_lr, early_stopping])
    evaluate(model, X_test, y_test, label_mapping.keys(), history)


if __name__ == "__main__":
    compute_features = False
    wav_data_path = "data/genres_wav"
    np_data_path = "data/genres_np"
    filters = 16
    nr_layers = 4
    batch_size = 1
    input_shape = (1025, 1292, 1)

    # config()
    # X_train, y_train, X_test, y_test, X_val, y_val, labels = get_features(wav_data_path, np_data_path, compute_features)
    # model = get_model(input_shape=X_train.shape[1:], nr_classes=len(labels), filters=filters, nr_layers=nr_layers)
    # run(model, X_train, y_train, X_test, y_test, X_val, y_val, labels)

    # Instanciate the generator
    list_of_files = list(Path("data/genres_wav").glob("*.wav"))
    train_files, test_files = train_test_split(list_of_files, test_size=0.2)
    val_files, test_files = train_test_split(test_files, test_size=0.5)
    print(f"Train data: {len(train_files)}")
    print(f"Validation data: {len(val_files)}")
    print(f"Test data: {len(test_files)}")
    
    train_generator = DataGenerator(list_IDs=train_files, batch_size=batch_size)
    val_generator = DataGenerator(list_IDs=val_files, batch_size=batch_size)
    x_test, y_test = load_data(test_files)

    model = get_model(input_shape=input_shape, nr_classes=len(label_mapping.keys()), filters=filters, nr_layers=nr_layers)
    run_generator(model, train_generator, val_generator, x_test, y_test)
