"""
Record audio from an existing input device (a microphone) and request the model for the genre prediction
"""
from argparse import ArgumentParser
import librosa
import numpy as np
import requests
import sounddevice as sd

def run(duration_sec, sample_rate, host, port):
    """
    Main program

    Args:
        duration_sec: the duration in seconds of the sound to be recorded
        sample_rate: the sample rate, should be 44100 or 22050
        host: the host of the prediction service
        port: the port of the prediction service
    """
    print("This program stream the output audio and predict the music genre")
    print("Press Ctrl+C for exit.")
    print(f"Listening the following device: {get_device(sd.query_devices())}.")

    while True:
        myrecording = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1).ravel()
        sd.wait()
        features = audio_to_spec(myrecording, int(sample_rate))
        print("Done.", features.shape)

        response = requests.post(f"http://{host}:{port}/classify", json=features.tolist())
        if response.status_code != 200:
            raise SystemExit(f"Unable to request the predapp service. Status code: {response.status_code}. Reason: {response.reason}")

        predicted_label = response.json().get("predicted_label")
        probability = round(response.json().get("probability") * 100, 2)
        print(f"{predicted_label} ({probability}%)")

def get_device(devices):
    """
    Get the device name from the result of 'sd.query_devices()'

    Args:
        devices: an object of class sounddevice.DeviceList
        idx: the indices of the choosen device
    
    Returns:
        A string of the device name
    """
    device_list = str(devices).split("\n")
    for device in device_list:
        if ">" in device:
            break
    return device.split("> ")[1]

def audio_to_spec(audio_array, sample_rate=44100, num_mfcc=40, n_fft=2048, hop_length=512, n_mels=40):
    """
    Compute the spectrograms from the audio array
    """
    mel = librosa.feature.melspectrogram(y=audio_array, n_mels=n_mels)
    harmonic, percusive = librosa.decompose.hpss(mel)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    features = np.stack((mfcc.T, harmonic.T, percusive.T), axis=-1)
    return np.expand_dims(features, axis=0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--sample_rate", default=44100, type=int)
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8080)
    args = parser.parse_args()
    
    try:
        run(duration_sec=args.duration,
            sample_rate=args.sample_rate,
            host=args.host,
            port=args.port)
    except (KeyboardInterrupt, SystemExit) as err:
        print("Ending the program")
        if err:
            print(err)
