# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#  
# Author: B. Gregorutti
# Email: baptiste.gregorutti@gmail.com

"""
    Extract the spectrograms from the audio files
"""

import os
from random import sample

import librosa
import numpy as np
from tqdm import tqdm

def load_from_file(fileName, representation, db_scale=True):
    """
        Extract the Mel-frequency Cepstral Coefficients
    """    

    # Load
    y, sr = librosa.load(fileName, mono=True, duration=30)

    # Return the spectrogram features
    if representation == 'cepstral':
        features = librosa.feature.mfcc(y=y, sr=sr)
    elif representation == 'stft':
        features = librosa.stft(y=y)

    # To db scale
    if db_scale:
        features = librosa.amplitude_to_db(features, ref=np.max)

    return features


def cepstral(nrFiles, folder):
    """
        Extract the Mel-frequency Cepstral Coefficients for all files in a folder
    """    

    files = os.listdir(folder)
    if nrFiles > 0 and nrFiles < len(files):
        files = sample(files, nrFiles)

    # Get the features
    features = []
    for filename in tqdm(files):

        # Load MFCC and store
        mfcc = load_from_file(os.path.join(folder, filename), 'cepstral')
        
        # There is one corrupted file in the list
        # if mfcc.shape != (20, 1292):
        #     print(k, filename)
        #     raise ValueError('Found you!')

        features.append(mfcc)
    
    # Get the targets
    targets = [filename.split('_')[0] for filename in tqdm(files)]
    
    return np.array(features), np.array(targets)


def stft(nrFiles, folder):
    """
        Extract the Short-time Fourier transform (STFT) for all files in a folder
    """    

    files = os.listdir(folder)
    if nrFiles > 0 and nrFiles < len(files):
        files = sample(files, nrFiles)

    # Get the features
    features = []
    for filename in tqdm(files):

        # Load MFCC and store
        mfcc = load_from_file(os.path.join(folder, filename), 'stft')
        
        # There is one corrupted file in the list
        # if mfcc.shape != (20, 1292):
        #     print(k, filename)
        #     raise ValueError('Found you!')

        features.append(mfcc)
    
    # Get the targets
    targets = [filename.split('_')[0] for filename in tqdm(files)]
    
    return np.array(features), np.array(targets)

if __name__ == '__main__':

    # features, targets = cepstral(0, '/Users/bgregorutti/Documents/Audio_processing/python/GTZAN_dataset/data/genres_wav')
    # np.save('preprocessed_data/spectrograms.npy', features)
    # np.save('preprocessed_data/targets.npy', targets)

    features, targets = stft(0, '/Users/bgregorutti/Documents/Audio_processing/python/GTZAN_dataset/data/genres_wav')
    np.save('preprocessed_data/fourier_features.npy', features)
    
    print(features.shape)
