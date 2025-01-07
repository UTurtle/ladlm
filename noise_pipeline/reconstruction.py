# noise_pipeline/reconstruction.py

import librosa
import numpy as np


def reconstruct_with_griffinlim(
        spectrogram_db, 
        hop_length,
        n_iter=32, 
        window='hann'
    ):
    amplitude = librosa.db_to_amplitude(spectrogram_db, ref=1.0)
    new_audio = librosa.griffinlim(
        amplitude,
        n_iter=n_iter,
        hop_length=hop_length,
        window=window
    )
    return new_audio


def reconstruct_audio_from_final_spectrogram(
        spectro_mod, 
    ):
    stft_orig = librosa.stft(
        spectro_mod.signal_with_noise,
        n_fft=spectro_mod.n_fft,
        hop_length=spectro_mod.hop_length,
        window=spectro_mod.window
    )
    phase = np.angle(stft_orig)
    magnitude = librosa.db_to_amplitude(spectro_mod.S_db, ref=1.0)
    new_stft = magnitude * np.exp(1j * phase)

    new_audio = librosa.istft(
        new_stft,
        hop_length=spectro_mod.hop_length,
        window=spectro_mod.window
    )
    return new_audio
