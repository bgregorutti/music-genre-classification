
import os
import librosa
from random import sample
import numpy as np
# from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

def featureExtraction():
    
    # Headers
    header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += ' mfcc{}'.format(i)
    header += ' label'
    header = header.split()

    folder = 'data/genres_wav/'
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
        
        currentSong = [filename, np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
        for e in mfcc:
            currentSong.append(np.mean(e))
        
        # Add the label
        currentSong.append(genre)
        
        features.append(currentSong)
    
    return pd.DataFrame(features, columns=header)

def compute_mfcc(file_name):
    """
    Extract the Mel-frequency Cepstral Coefficients
    """    
    # Load
    y, sr = librosa.load(file_name, mono=True, duration=30)

    # Return the spectrogram features
    return librosa.feature.mfcc(y=y, sr=sr)

def mel_spectrogram(file_name):
    y, sr = librosa.load(file_name, mono=True, duration=30)
    spectro = librosa.feature.melspectrogram(y, sr=sr)
    return librosa.amplitude_to_db(spectro, ref=np.max)

def spectrogram(file_name):
    y, sr = librosa.load(file_name, mono=True, duration=30)
    spectro = np.abs(librosa.stft(y))
    return librosa.amplitude_to_db(spectro, ref=np.max)

def load_spec_data(fileName, outFile):
    """
    Extract the Mel-frequency Cepstral Coefficients
    """    
    # Load
    y, sr = librosa.load(fileName, mono=True, duration=30)

    # Return the spectrogram features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # To NPY
    np.save(outFile, mfcc)

    # return mfcc


def specImgs(nrFiles):
    
    folder = 'data/genres_wav/'
    
    files = os.listdir(folder)
    if nrFiles:
        files = sample(files, nrFiles)

    features = []
    for k, filename in enumerate(files):
        
        if not (k+1) % 100:
            print('{} songs imported'.format(k))

        # Load MFCC and store
        mfcc = loadSpecData(os.path.join(folder, filename))
        if mfcc.shape != (20, 1292):
            print(k, filename)
            raise ValueError('Found you!')
        features.append(mfcc)
    
    targets = [filename.split('_')[0] for filename in tqdm(files)]

    features = np.array(features)
    targets = np.array(targets)
    
    return features, targets



# 
# Parallel execution
# 

def changeExtension(fileName, ext):

    tmp = fileName.split('.')[0]
    return tmp + ext

def loadSpecDataParallel(nrFiles):
    """
        Load the wav, extract the spectrograms and export to CSV
    """

    folder = 'data/genres_wav/'
    folder_csv = 'data/spectrograms_csv/'
    
    files = os.listdir(folder)
    if nrFiles:
        files = sample(files, nrFiles)

    Parallel(n_jobs=-1)(delayed(load_spec_data)(os.path.join(folder, filename), os.path.join(folder_csv, changeExtension(filename, '.csv'))) for filename in tqdm(files))


def importCSV(nrFiles):
    folder = 'data/spectrograms_csv/'
    
    files = os.listdir(folder)
    if nrFiles:
        files = sample(files, nrFiles)

    print('Get the spectrogram features')
    # features = [wrapper(folder, fileName) for fileName in tqdm(files)]
    features = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(wrapper)(folder, filename) for filename in files))
    
    print('Get the targets')
    genres = np.array([filename.split('_')[0] for filename in files])

    return features, genres


def wrapper(folder, fileName): 
    return pd.read_csv(os.path.join(folder, fileName)).values

if __name__ == '__main__':
    # loadSpecDataParallel(0)
    # loadSpecData('data/genres_wav/metal_00056.wav', 'aaaa.npy')
    features, targets = importCSV(0)
    np.save('spectrograms.npy', features)
    np.save('targets.npy', targets)

