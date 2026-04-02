import librosa
import librosa.display
import numpy as np
import os


def extract_spectrogram(file_path):

    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)


    S_dB = librosa.power_to_db(S, ref=np.max)

    if S_dB.shape[1] > 128:
        S_dB = S_dB[:, :128]
    else:
        S_dB = np.pad(S_dB, ((0, 0), (0, 128 - S_dB.shape[1])), mode='constant')

    return S_dB

