__author__ = "Vitaly Butoma"

import librosa
import librosa.display
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import sys
import sox


HIGH_NOISE_DATA = './data/background_noise_high'


def load_files(folder=HIGH_NOISE_DATA):
    res = []

    return res


def load_file(file_name, folder):
    file_path = os.path.join(folder, file_name)
    wav, sr = librosa.load(file_path, sr=None)
    return wav, sr


def get_spectrogram(wav):
    D = librosa.stft(wav, n_fft=480, hop_length=160,
                     win_length=480, window='hamming')
    spect, phase = librosa.magphase(D)
    return spect


def noise_red(file_name, folder):
    file_path = os.path.join(folder, file_name)
    profile_path = 'audio.prof'
    s = sox.Transformer()
    s.noiseprof(file_path, profile_path)
    s.noisered(profile_path, 0.3)
    s.build(file_path, 'clear.wav')


def noisered(input_file, out_file, delta=0.0):
    profile_path = 'audio.prof'
    s = sox.Transformer()
    s.noiseprof(input_file, profile_path)
    s.noisered(profile_path, delta)
    s.build(input_file, out_file)


def clear_folder(folder):
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, .5]
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        output_folder = os.path.join("{}_{}".format(folder, 'processed'), file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for _ in thresholds:
            output_file_path = os.path.join(output_folder, "delta{}{}".format(_, file))
            noisered(file_path, output_file_path, delta=_)


if __name__ == "__main__":
    name = 'Sample_1_ Female.wav'
    # wav, sr = load_file(file_name=name, folder=HIGH_NOISE_DATA)
    # print(wav.shape, wav.max(), wav.min())
    # noise_red(file_name=name, folder=HIGH_NOISE_DATA)

    clear_folder(HIGH_NOISE_DATA)
