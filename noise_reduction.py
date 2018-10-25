import librosa
import librosa.display
import os
import numpy as np
import sox
import time

from acapellabot import Acapella


class NoiseReduction:

    def __init__(self, weights='weights/weights.h5'):
        # init model
        self.acapella = Acapella()
        self.acapella.load_weights(weights)

    def solve_big_file(self, file_path, output_file, use_acapella=True, use_harmonic=True):
        # make chunks
        chunks = self._make_chunks(file_path)
        for chunk in chunks:
            self.solve(chunk, use_acapella=use_acapella, use_harmonic=use_harmonic)
        result_y, sr = self._concat_chunks(chunks)
        librosa.output.write_wav(output_file, result_y, sr)

    def _make_chunks(self, file_name):
        max_chunk_size = 2 * 10**5
        y, sr = librosa.load(file_name, sr=None)
        n = len(y)
        max_chunk_size = n  # :))))
        i_count = n // max_chunk_size + (1 if (n % max_chunk_size) else 0)
        print('Len of audio: {}. Chunk size: {}. Chunks count: {}'.format(n, max_chunk_size, i_count))
        chunk_names = []
        for i in range(i_count):
            left = i * max_chunk_size
            right = min(n, (i + 1) * max_chunk_size)
            chunk = y[left: right]
            chunk_name = '{}_{}'.format(file_name, i)
            chunk_names.append(chunk_name)
            librosa.output.write_wav(chunk_name, chunk, sr)
        return chunk_names

    def _concat_chunks(self, chunk_names):
        res = np.ndarray((0, ))
        for chunk_name in chunk_names:
            name = '{}_result.wav'.format(chunk_name)
            y, sr = librosa.load(name, sr=None)
            # res = np.concatenate(res, y)  # todo
            res = y
        return res, sr

    def solve(self, file_path, use_acapella=True, use_harmonic=True, fft=1536, phase=10):
        """
        1: preprocess audio: mb smooth ??
        2. acapella model
        3. noise reduction
        4. smoothing
        5. todo: increase the timbre
        6. todo: reduce the echo
        """
        # init const names
        ISOLATED = '{}_isolated.wav'.format(file_path)
        PREPROCESSED = '{}_pre.wav'.format(file_path)
        N_REDUCTIONED= '{}_red.wav'.format(file_path)
        RESULT= '{}_result.wav'.format(file_path)
        # =========================
        start_time = time.time()
        audio, sr = self.load_and_preprocess(file_path, smooth=False, harmonic=use_harmonic)
        librosa.output.write_wav(PREPROCESSED, audio, sr, norm=True)
        if use_acapella:
            audio = self.isolate_vocal(PREPROCESSED, fft, phase)
        else:
            # audio = self.isolate_vso(PREPROCESSED)
            pass
        # save temp file
        self._save_temp(audio, sr, ISOLATED)
        self.noise_reduction(ISOLATED, N_REDUCTIONED, delta=0.05)
        self.post_process(N_REDUCTIONED, RESULT, smooth=False)
        # todo 5th step: increase the timbre

        print('Time to reduce: ', time.time() - start_time)

    def _save_temp(self, x, sr, name):
        print('Save temp to:', name)
        librosa.output.write_wav(name, x, sr, norm=True)

    def load_and_preprocess(self, file_path, smooth=False, harmonic=True):
        audio, sr = librosa.load(file_path, sr=None)
        if smooth:
            audio = self._smooth(audio, window_len=5, window='flat')
        if harmonic:
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            audio = y_harmonic
        return audio, sr

    def post_process(self, inp, out, smooth=False):
        wav, sr = librosa.load(inp, sr=None)
        y_harmonic, y_percussive = librosa.effects.hpss(wav)
        # if smooth:
        #     wav = self._smooth(wav, window_len=7, window='hamming')
        librosa.output.write_wav('harmonic.wav', y_harmonic, sr, norm=True)
        librosa.output.write_wav(out, wav, sr, norm=True)
        librosa.output.write_wav('percussive.wav', y_percussive, sr, norm=True)

    def _smooth(self, x, window_len=11, window='hanning'):
        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def isolate_vocal(self, audio_path, fft, phase):
        isolated = self.acapella.isolate_vocals(audio_path, fft, phase)

        return isolated

    def isolate_vso(self, file_name):
        audio, sr = librosa.load(file_name)
        s_full, phase = librosa.magphase(librosa.stft(audio))
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
        return s_foreground

    def noise_reduction(self, input_file, out_file, delta=0.25):
        profile_path = 'audio.prof'
        s = sox.Transformer()
        s.noiseprof(input_file, profile_path)
        s.noisered(profile_path, delta)
        s.build(input_file, out_file)


if __name__ == "__main__":
    v = NoiseReduction()
    # v.solve(file_path='test/2/1.wav', use_acapella=True)
    v.solve_big_file(file_path='test/7/1.wav', output_file='test/7/FUll_RESULT.wav',
                     use_harmonic=True, use_acapella=True)
