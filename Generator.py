from pathlib import Path
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
from extract_spectrogram import spectrogram

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

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x_data, targets = self.__data_generation(list_IDs_temp)

        return x_data, targets

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        x_data = np.array([spectrogram(path) for path in list_IDs_temp])
        n, img_rows, img_cols = x_data.shape
        x_data = x_data.reshape(n, img_rows, img_cols, 1)
        targets = np.array([label_mapping[path.stem.split("_")[0]] for path in list_IDs_temp])
        return x_data, to_categorical(targets, num_classes=len(label_mapping))

def load_data(list_of_files):
    # Manually load the data
    x_data = np.array([spectrogram(path) for path in list_of_files])
    x_data = np.expand_dims(x_data, -1)
    targets = np.array([label_mapping[path.stem.split("_")[0]] for path in list_of_files])
    return x_data, to_categorical(targets, num_classes=len(label_mapping))

if __name__ == "__main__":
    list_of_files = list(Path("data/genres_wav").glob("*.wav"))
    gen = DataGenerator(list_IDs=list_of_files, batch_size=100)
    print(len(gen))
    for (x_data, targets) in gen:
        print(x_data.shape, targets.shape)