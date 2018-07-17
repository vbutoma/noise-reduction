import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
from os.path import basename
from math import ceil
import argparse
import console


def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path)
    return audio, sample_rate


def save_audio(audio, file_path, sample_rate):
    librosa.output.write_wav(file_path, audio, sample_rate, norm=True)
    console.info("Wrote audio file to", file_path)


def expand_to_grid(spectrogram, grid_size):
    # crop along both axes
    new_Y = ceil(spectrogram.shape[1] / grid_size) * grid_size
    new_X = ceil(spectrogram.shape[0] / grid_size) * grid_size
    new_spectrogram = np.zeros((new_X, new_Y))
    new_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
    return new_spectrogram


# todo: need research
def isolate_vocal_simple(audio, fft_window_size, sr):
    s_full, phase = librosa.magphase(librosa.stft(audio, fft_window_size))
    s_filter = librosa.decompose.nn_filter(
        s_full,
        aggregate=np.median,
        metric='cosine',
        width=int(librosa.time_to_frames(2, sr=sr))
    )
    s_filter = np.minimum(s_full, s_filter)
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(s_filter,
                                   margin_i * (s_full - s_filter),
                                   power=power)

    mask_v = librosa.util.softmask(
        s_full - s_filter,
        margin_v * s_filter,
        power=power
    )
    s_foreground = mask_v * s_full
    s_background = mask_i * s_full
    return s_foreground, phase


# Return a 2d numpy array of the spectrogram
def audio_to_spectrogram(audio, fft_window_size, sr):
    spectrogram = librosa.stft(audio, fft_window_size)
    phase = np.imag(spectrogram)
    amplitude = np.log1p(np.abs(spectrogram))
    return amplitude, phase


# This is the nutty one
def spectrogram_to_audio(spectrogram, fft_window_size, phase_iterations=10, phase=None):
    if phase is not None:
        # reconstructing the new complex matrix
        squared_amplitude = np.power(spectrogram, 2)
        squared_phase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squared_amplitude - squared_phase, 0))
        amplitude = np.expm1(unexpd)
        stft_matrix = amplitude + phase * 1j
        audio = librosa.istft(stft_matrix)
    else:
        # phase reconstruction with successive approximation
        # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
        # for the algorithm used
        amplitude = np.exp(spectrogram) - 1
        for i in range(phase_iterations):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * \
                                 (2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, fft_window_size)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio


def load_spectrogram(file_path):
    filename = basename(file_path)
    sample_rate = 22050
    if file_path.index("sampleRate") >= 0:
        sample_rate = int(filename[filename.index("sampleRate=") + 11:filename.index(").png")])
    console.info("Using sample rate : " + str(sample_rate))
    image = io.imread(file_path, as_grey=True)
    return image / np.max(image), sample_rate


def save_spectrogram(spectrogram, file_path):
    spectrum = spectrogram
    console.info("Range of spectrum is " + str(np.min(spectrum)) + " -> " + str(np.max(spectrum)))
    image = np.clip((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)), 0, 1)
    console.info("Shape of spectrum is", image.shape)
    # Low-contrast image warnings are not helpful, tyvm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(file_path, image)
    console.log("Saved image to", file_path)


def file_suffix(title, **kwargs):
    return " (" + title + "".join(sorted([", " + i + "=" + str(kwargs[i]) for i in kwargs])) + ")"


def handle_audio(file_path, args):
    console.h1("Creating Spectrogram")
    INPUT_FILE = file_path
    INPUT_FILENAME = basename(INPUT_FILE)

    console.info("Attempting to read from " + INPUT_FILE)
    audio, sample_rate = load_audio(INPUT_FILE)
    console.info("Max of audio file is " + str(np.max(audio)))
    spectrogram, phase = audio_to_spectrogram(audio, fft_window_size=args.fft, sr=sample_rate)
    SPECTROGRAM_FILENAME = INPUT_FILENAME + file_suffix(
        "Input Spectrogram", fft=args.fft, iter=args.iter, sampleRate=sample_rate) + ".png"

    save_spectrogram(spectrogram, SPECTROGRAM_FILENAME)

    print()
    console.wait("Saved Spectrogram; press Enter to continue...")
    print()

    handle_image(SPECTROGRAM_FILENAME, args, phase)


def handle_image(file_name, args, phase=None):
    console.h1("Reconstructing Audio from Spectrogram")

    spectrogram, sample_rate = load_spectrogram(file_name)
    audio = spectrogram_to_audio(spectrogram, fft_window_size=args.fft, phase_iterations=args.iter)

    sanity_check, phase = audio_to_spectrogram(audio, fft_window_size=args.fft, sr=sample_rate)
    save_spectrogram(sanity_check, file_name + file_suffix(
        "Output Spectrogram", fft=args.fft, iter=args.iter, sample_rate=sample_rate) + ".png")

    save_audio(audio, file_name + file_suffix("Output", fft=args.fft, iter=args.iter) + ".wav", sample_rate)


if __name__ == "__main__":
    # Test code for experimenting with modifying acapellas in image processors (and generally
    # testing the reconstruction pipeline)
    parser = argparse.ArgumentParser(description="Convert image files to audio and audio files to images")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--iter", default=10, type=int, help="Number of iterations to use for phase reconstruction")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    for f in args.files:
        if f.endswith(".mp3") or f.endswith(".wav"):
            handle_audio(f, args)
        elif f.endswith(".png"):
            handle_image(f, args)
