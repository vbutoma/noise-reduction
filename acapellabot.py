"""
Acapella extraction with a CNN

Typical usage:
    python acapellabot.py song.wav
    => Extracts acapella from <song.wav> to <song (Acapella Attempt).wav> using default weights

    python acapellabot.py --data input_folder --batch 32 --weights new_model_iteration.h5
    => Trains a new model based on song/acapella pairs in the folder <input_folder>
       and saves weights to <new_model_iteration.h5> once complete.
       See data.py for data specifications.
"""

import argparse
import random, string
import os, sys, time
os.environ['MKL_NUM_THREADS'] = '7'
os.environ['GOTO_NUM_THREADS'] = '7'
os.environ['OMP_NUM_THREADS'] = '7'
import librosa
import numpy as np
# import keras as kr
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model

import console
import conversion
from data_helper import Data


class Acapella:

    def __init__(self):
        mashup = Input(shape=(None, None, 1), name='input')
        convA = Conv2D(64, 3, activation='relu', padding='same')(mashup)
        conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convA)
        conv = BatchNormalization()(conv)

        convB = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convB)
        conv = BatchNormalization()(conv)

        conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(128, 3, activation='relu', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convB])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convA])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(1, 3, activation='relu', padding='same')(conv)
        acapella = conv
        m = Model(inputs=mashup, outputs=acapella)
        console.log("Model has", m.count_params(), "params")
        m.compile(loss='mean_squared_error', optimizer='adam')
        self.model = m
        # need to know so that we can avoid rounding errors with spectrogram
        # this should represent how much the input gets downscaled
        # in the middle of the network
        self.peak_downscale_factor = 4

    def train(self, data, epochs, batch=8):
        x_train, y_train = data.train()
        x_test, y_test = data.valid()
        while epochs > 0:
            console.log("Training for", epochs, "epochs on", len(x_train), "examples")
            self.model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_data=(x_test, y_test))
            console.notify(str(epochs) + " Epochs Complete!", "Training on", data.inPath, "with size", batch)
            while True:
                try:
                    epochs = int(input("How many more epochs should we train for? "))
                    break
                except ValueError:
                    console.warn("Oops, number parse failed. Try again, I guess?")
            if epochs > 0:
                save = input("Should we save intermediate weights [y/n]? ")
                if not save.lower().startswith("n"):
                    weight_path = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
                    console.log("Saving intermediate weights to", weight_path)
                    self.save_weights(weight_path)

    def save_weights(self, path):
        self.model.save_weights(path, overwrite=True)

    def load_weights(self, path):
        self.model.load_weights(path)

    def isolate_vocals(self, path, fft_window_size, phase_iterations=10):
        console.log("Attempting to isolate vocals from", path)
        start_time = time.time()
        audio, sample_rate = conversion.load_audio(path)
        spectrogram, phase = conversion.audio_to_spectrogram(audio, fft_window_size=fft_window_size, sr=sample_rate)
        # spectrogram, phase = conversion.isolate_vocal_simple(audio, fft_window_size=fft_window_size, sr=sample_rate)
        console.log("Retrieved spectrogram; processing...")

        expanded_spectrogram = conversion.expand_to_grid(spectrogram, self.peak_downscale_factor)
        expanded_spectrogram_with_batch_channels = expanded_spectrogram[np.newaxis, :, :, np.newaxis]
        predicted_spectrogram_with_batch_channels = self.model.predict(expanded_spectrogram_with_batch_channels)
        predicted_spectrogram = predicted_spectrogram_with_batch_channels[0, :, :, 0] # o /// o
        new_spectrogram = predicted_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]]
        console.log("Processed spectrogram; reconverting to audio")

        new_audio = conversion.spectrogram_to_audio(
            new_spectrogram,
            fft_window_size=fft_window_size,
            phase_iterations=phase_iterations
        )
        path_parts = os.path.split(path)
        filename_parts = os.path.splitext(path_parts[1])
        output_filename_base = os.path.join(path_parts[0], filename_parts[0] + "_acapella")
        console.log("Converted to audio; writing to", output_filename_base)

        conversion.save_audio(new_audio, output_filename_base + ".wav", sample_rate)
        conversion.save_spectrogram(new_spectrogram, output_filename_base + ".png")
        conversion.save_spectrogram(spectrogram, os.path.join(path_parts[0], filename_parts[0]) + ".png")
        # console.log("Vocal isolation complete 👌")
        print('execution time: {}'.format(time.time() - start_time))
        return new_audio


if __name__ == "__main__":
    # if data folder is specified, create a new data object and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="weights.h5", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int, help="Phase iterations for reconstruction")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    acapellabot = Acapella()

    if len(args.files) == 0 and args.data:
        console.log("No files provided; attempting to train on " + args.data + "...")
        if args.load:
            console.h1("Loading Weights")
            acapellabot.load_weights(args.weights)
        console.h1("Loading Data")
        data = Data(args.data, args.fft, args.split)
        console.h1("Training Model")
        acapellabot.train(data, args.epochs, args.batch)
        acapellabot.save_weights(args.weights)
    elif len(args.files) > 0:
        console.log("Weights provided; performing inference on " + str(args.files) + "...")
        console.h1("Loading weights")
        acapellabot.load_weights(args.weights)
        for f in args.files:
            acapellabot.isolate_vocals(f, args.fft, args.phase)
    else:
        console.error("Please provide data to train on (--data) or files to infer on")
