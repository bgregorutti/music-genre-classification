"""
Some routines for extracting the audio features from the raw data
"""

import math
import os
from pathlib import Path

import librosa
import numpy as np
from pandas import DataFrame

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

def static_features(folder):
    """
    Extract 'static' features from the audio files

    Args:
        folder: containing WAV files
    
    Returns:
        A DataFrame object
    """

    # Headers
    header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += ' mfcc{}'.format(i)
    header += ' label'
    header = header.split()

    features = []
    for k, filename in enumerate(os.listdir(folder)):

        if not k % 100:
            print('{} songs imported'.format(k))
        
        songname = os.path.join(folder, filename)
        genre = filename.split('_')[0]
        
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        current_song = [filename, np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
        for e in mfcc:
            current_song.append(np.mean(e))
        
        # Add the label
        current_song.append(genre)
        
        features.append(current_song)
    
    return DataFrame(features, columns=header)

def mfcc_features(path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10, amplitude_to_db=False):
    """
    The MFCC 2D features

    Args:
        path: path of the folder containing the WAV files
        num_mfcc: number of MFCCs to return. Default: 40
        n_fft: length of the FFT window. Default: 2048
        hop_length: number of samples between successive frames. Default: 512
        num_segment: number of time segment to extract. Default: 10
        amplitude_to_db: Convert an amplitude spectrogram to dB-scaled spectrogram. Default: False
    
    Returns:
        Two numpy arrays, the features and the labels
    """
    features = []
    labels = []
    sample_rate = 22050
    sample_per_segment = int(sample_rate * 30 / num_segment)
    
    for file_path in Path(path).glob("*.wav"):
        print(f"Track name: {file_path.name}")
        y, sample_rate = librosa.load(file_path, sr=sample_rate)
        label = file_path.stem.split("_")[0]

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

def multichannel_spectrogram(path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10, amplitude_to_db=False):
    """
    Multichannel spectrograms. Stack MFCC, Harmonic & Percusive features

    Args:
        path: path of the folder containing the WAV files
        num_mfcc: number of MFCCs to return. Default: 40
        n_fft: length of the FFT window. Default: 2048
        hop_length: number of samples between successive frames. Default: 512
        num_segment: number of time segment to extract. Default: 10
        amplitude_to_db: Convert an amplitude spectrogram to dB-scaled spectrogram. Default: False
    
    Returns:
        Two numpy arrays, the features and the labels
    """
    features = []
    labels = []
    sample_rate = 22050
    sample_per_segment = int(sample_rate * 30 / num_segment)
    
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
