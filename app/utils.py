"""
Utils functions: data management, prediction APIs
"""

import base64
import math
import requests

import librosa
import numpy as np
import pandas as pd

def split_data(data, sr):
    """
    Split into segments of 3 seconds

    Args:
        data: a ndarray
        sr: sample_rate
    
    Returns:
        A numpy array of the spectrograms
    """
    features = []
    num_mfcc = 40
    n_fft = 2048
    hop_length = 512

    total_seconds = data.size / sr
    nr_segments = int(total_seconds / 3)
    if not nr_segments:
        nr_segments = 1
    
    sample_per_segment = int(data.size / nr_segments)
    for n in range(nr_segments):
        segment = data[sample_per_segment*n:sample_per_segment*(n+1)]
        mel = librosa.feature.melspectrogram(y=segment, n_mels=40)
        harmonic, percusive = librosa.decompose.hpss(mel)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        spectro = np.stack((mfcc.T, harmonic.T, percusive.T), axis=-1)
        if len(spectro) == math.ceil(sample_per_segment / hop_length):
            features.append(spectro.tolist())
    return np.array(features)

def predict_genre(features, host, port):
    """
    Predict the genre of one sequence

    Args:
        features: numpy array of the spectrograms
        host, port: the host and port of the prediction app

    Returns:
        A string of the predicted label and the probability
    """
    response = requests.post(f"http://{host}:{port}/classify", json=features.tolist())
    predicted_label = response.json().get("predicted_label")
    probability = round(response.json().get("probability") * 100, 2)
    return f"{predicted_label} ({probability}%)"

def predict_genre_overall(features, host, port):
    """
    Predict the genre of the all song

    Args:
        features: numpy array of the spectrograms
        host, port: the host and port of the prediction app

    Returns:
        A dictionary of the predicted genres with corresponding probabilities
    """
    response = requests.post(f"http://{host}:{port}/classify_overall", json=features.tolist())
    if response.status_code != 200:
        return "N/A"

    predicted_labels = response.json().get("predicted_labels")
    probabilities = response.json().get("probabilities")
    genres = [{"Predicted label": predicted_labels[k], "Probability": f"{probabilities[k]*100:.2f}%"} for k in np.argsort(probabilities)[::-1]]
    return genres

def read_data(file_name):
    """
    Read a WAV file

    Args:
        file_name: name of the WAV file

    Returns:
        The encoded data, the sample_rate and two DataFrames (the raw signal and the resampled signal)
    """
    # Open the default file and process it
    encoded_sound = base64.b64encode(open(str(file_name), "rb").read())
    np_data, sr = librosa.load(str(file_name))
    df_raw = pd.DataFrame({
        "time": np.array([t / sr for t in range(len(np_data))]),
        "data": np_data
    })
    df_raw["time"] = pd.to_datetime(df_raw["time"], unit="s")

    target_sr = 1000
    sample = librosa.resample(np_data, orig_sr=sr, target_sr=target_sr)

    df = pd.DataFrame({
        "time": np.array([t / target_sr for t in range(len(sample))]),
        "data": sample
    })
    df["time"] = pd.to_datetime(df["time"], unit="s")

    return encoded_sound, sr, df_raw, df
