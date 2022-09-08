"""
MAIN PROGRAM
"""
from pandas import DataFrame
from genre.features_extraction import label_mapping
from genre.train_cnn import get_features, run
from genre.models import convolutional

compute_features = False
wav_data_path = "data/genres_wav"
np_data_path = "data/genres_np/multi_channel"
n_repeat = 10

if n_repeat:
    metrics = []
    for k in range(n_repeat):
        print(
f"""
***
*** RUN {k}
***
""")
        X_train, y_train, X_test, y_test, X_val, y_val, labels = get_features(wav_data_path, np_data_path, compute_features)
        model = convolutional(input_shape=X_train.shape[1:], nr_classes=len(labels), padding="same")
        metrics.append(run(model, X_train, y_train, X_test, y_test, X_val, y_val, label_mapping.keys()))
    
    metrics = DataFrame(metrics, columns=["loss", "accuracy"])
    print(metrics)
    print(metrics.describe())
else:
    X_train, y_train, X_test, y_test, X_val, y_val, labels = get_features(wav_data_path, np_data_path, compute_features)
    model = convolutional(input_shape=X_train.shape[1:], nr_classes=len(labels), padding="same")
    run(model, X_train, y_train, X_test, y_test, X_val, y_val, label_mapping.keys())
    model.save("model")
