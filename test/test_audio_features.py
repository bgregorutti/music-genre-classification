from genre.audio_features import compute_mfcc, mel_spectrogram, spectrogram

def test_compute_mfcc():
    path = "test/resources/track_1.wav"
    data = compute_mfcc(path)
    print(data.shape)


def test_mel_spectrogram():
    path = "test/resources/track_1.wav"
    data = mel_spectrogram(path)
    print(data.shape)


def test_spectrogram():
    path = "test/resources/track_1.wav"
    data = spectrogram(path)
    print(data.shape)


if __name__ == "__main__":
    test_compute_mfcc()
    test_mel_spectrogram()
    test_spectrogram()
