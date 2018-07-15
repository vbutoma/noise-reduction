__author__ = "Vitaly Butoma"

import librosa
import librosa.display
import os
import numpy as np
import sox


HIGH_NOISE_DATA = './data/background_noise_high'
LOW_NOISE_DATA = './data/background_noise_low'


def save_file(wav, sr):
    librosa.output.write_wav('test.wav', wav, sr, norm=False)
    librosa.output.write_wav('test_nrm.wav', wav, sr, norm=True)


def smooth(x, window_len=11, window='hanning'):
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def load_file(file_name, folder):
    file_path = os.path.join(folder, file_name)
    wav, sr = librosa.load(file_path, sr=None)
    return wav, sr


def post_process(input_file, processed_file):
    wav_i, sr_i = librosa.load(input_file, sr=None)
    wav_p, sr_p = librosa.load(processed_file, sr=None)
    smoothed = smooth(wav_p, window_len=5, window='blackman')
    librosa.output.write_wav('smoothed.wav', smoothed, sr)


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


def solve(file_name):
    temp_file = '{}_clear_temp.wav'.format(file_name)
    reduction_file = '{}_clear_red.wav'.format(file_name)
    post_processed = '{}_clear_post.wav'.format(file_name)
    # pre process
    wav, sr = librosa.load(file_name, sr=None)
    smoothed = smooth(wav, window_len=7, window='blackman')
    librosa.output.write_wav(temp_file, smoothed, sr, norm=False)

    # noise reduction
    noisered(temp_file, reduction_file, delta=0.1)

    # post processed
    wav, sr = librosa.load(reduction_file, sr=None)
    smoothed = smooth(wav, window_len=7, window='blackman')
    librosa.output.write_wav(post_processed, smoothed, sr, norm=False)


def cut_audio(file_path, delta=1000000, left_shift=300000, right_shift=500000):
    w, s = librosa.load(file_path)
    print(len(w))
    librosa.output.write_wav(
        'cutted.wav',
        w[delta + left_shift: delta + right_shift],
        s
    )


if __name__ == "__main__":
    name = 'Sample_1_Female.wav'
    # clear_folder(LOW_NOISE_DATA)
    file_path = os.path.join(LOW_NOISE_DATA, name)
    file_path = 'music_over_2.wav'
    # solve(file_path)
    cut_audio(file_path, delta=0)

