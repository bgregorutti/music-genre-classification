"""
MAIN PROGRAM
"""
from pathlib import Path
from sklearn.model_selection import train_test_split
from genre.train_cnn import run_generator
from genre.models import deep_convolutional
from genre.features_extraction import label_mapping
from genre.data_generator import DataGenerator

compute_features = False
wav_data_path = "data/genres_wav"
np_data_path = "data/genres_np"
filters = 16
nr_layers = 4
batch_size = 1
input_shape = (1025, 1292, 1)

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

model = deep_convolutional(input_shape=input_shape, nr_classes=len(label_mapping.keys()), filters=filters, nr_layers=nr_layers)
run_generator(model, train_generator, val_generator, x_test, y_test)
