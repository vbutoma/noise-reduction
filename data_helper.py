"""
Loads and stores mashup data given a folder full of acapellas and instrumentals
Assumes that all audio clips (wav, mp3) in the folder
a) have their Camelot key as the first token in the filename
b) are in the same BPM
c) have "acapella" somewhere in the filename if they're an acapella, and are otherwise instrumental
d) all have identical arrangements
e) have the same sample rate
"""
import sys
import os
import numpy as np
import h5py

import console
import conversion


# Modify these functions if your data is in a different format
def key_of_file(fileName):
    first_token = int(fileName.split()[0])
    if 0 < first_token <= NUMBER_OF_KEYS:
        return first_token
    console.warn("File", fileName, "doesn't specify its key, ignoring..")
    return None


def file_is_acapella(file_name):
    return "acapella" in file_name.lower()


NUMBER_OF_KEYS = 12  # number of keys to iterate over
SLICE_SIZE = 128    # size of spectrogram slices to use


# Slice up matrices into squares so the neural net gets a consistent size for training (doesn't matter for inference)
def chop(matrix, scale):
    slices = []
    for time in range(0, matrix.shape[1] // scale):
        for freq in range(0, matrix.shape[0] // scale):
            s = matrix[freq * scale : (freq + 1) * scale,
                       time * scale : (time + 1) * scale]
            slices.append(s)
    return slices


class Data:

    def __init__(self, in_path, fft_window_size=1536, training_split=0.9):
        self.inPath = in_path
        self.fft_window_size = fft_window_size
        self.training_split = training_split
        self.x = []
        self.y = []
        self.load()

    def train(self):
        return self.x[:int(len(self.x) * self.training_split)], self.y[:int(len(self.y) * self.training_split)]

    def valid(self):
        return self.x[int(len(self.x) * self.training_split):], self.y[int(len(self.y) * self.training_split):]

    def load(self, as_h5=False):
        h5_path = os.path.join(self.inPath, "data.h5")
        if os.path.isfile(h5_path):
            h5f = h5py.File(h5_path, "r")
            self.x = h5f["x"][:]
            self.y = h5f["y"][:]
        else:
            acapellas = {}
            instrumentals = {}
            # Hash bins for each camelot key so we can merge
            # in the future, this should be a generator w/ yields in order to eat less memory
            for i in range(NUMBER_OF_KEYS):
                key = i + 1
                acapellas[key] = []
                instrumentals[key] = []
            for dir_path, dir_names, file_names in os.walk(self.inPath):
                for file_name in filter(lambda f: (f.endswith(".mp3") or f.endswith(".wav")) and not f.startswith("."),
                                        file_names):
                    key = key_of_file(file_name)
                    if key:
                        target_path_map = acapellas if file_is_acapella(file_name) else instrumentals
                        tag = "[Acapella]" if file_is_acapella(file_name) else "[Instrumental]"
                        audio, sample_rate = conversion.load_audio(os.path.join(self.inPath, file_name))
                        spectrogram, phase = conversion.audio_to_spectrogram(
                            audio,
                            self.fft_window_size,
                            sr=sample_rate
                        )
                        target_path_map[key].append(spectrogram)
                        console.info(tag, "Created spectrogram for", file_name, "in key",
                                     key, "with shape", spectrogram.shape)
            # Merge mashups
            for k in range(NUMBER_OF_KEYS):
                acapellas_in_key = acapellas[k + 1]
                instrumentals_in_key = instrumentals[k + 1]
                count = 0
                for acapella in acapellas_in_key:
                    for instrumental in instrumentals_in_key:
                        # Pad if smaller
                        if instrumental.shape[1] < acapella.shape[1]:
                            new_instrumental = np.zeros(acapella.shape)
                            new_instrumental[:instrumental.shape[0], :instrumental.shape[1]] = instrumental
                            instrumental = new_instrumental
                        elif acapella.shape[1] < instrumental.shape[1]:
                            new_acapella = np.zeros(instrumental.shape)
                            new_acapella[:acapella.shape[0], :acapella.shape[1]] = acapella
                            acapella = new_acapella
                        # simulate a limiter/low mixing (loses info, but that's the point)
                        # I've tested this against making the same mashups in Logic and it's pretty close
                        mashup = np.maximum(acapella, instrumental)
                        # chop into slices so everything's the same size in a batch
                        dim = SLICE_SIZE
                        mashup_slices = chop(mashup, dim)
                        acapella_slices = chop(acapella, dim)
                        count += 1
                        self.x.extend(mashup_slices)
                        self.y.extend(acapella_slices)
                console.info("Created", count, "mashups for key", k, "with", len(self.x), "total slices so far")
            # Add a "channels" channel to please the network
            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]
            # Save to file if asked
            if as_h5:
                h5f = h5py.File(h5_path, "w")
                h5f.create_dataset("x", data=self.x)
                h5f.create_dataset("y", data=self.y)
                h5f.close()


if __name__ == "__main__":
    # Simple testing code to use while developing
    console.h1("Loading Data")
    d = Data(sys.argv[1], 1536)
    console.h1("Writing Sample Data")
    conversion.save_spectrogram(d.x[0], "x_sample_0.png")
    conversion.save_spectrogram(d.y[0], "y_sample_0.png")
    audio = conversion.spectrogram_to_audio(d.x[0], 1536)
    conversion.save_audio(audio, "x_sample.wav", 22050)
