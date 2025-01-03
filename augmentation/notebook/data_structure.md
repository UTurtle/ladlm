
### Dataset Sturcture

> [audio][image][stft_maker_parameter][text]

```json
{
    "audio": "audio_file_path_or_base64_encoded",
    "image": "spectrogram_image_path_or_base64_encoded",
    "stft_maker_parameter": {
        "sample_rate": 16000,
        "duration": 12,
        "n_samples": 192000,
        "n_fft": 1024,
        "hop_length": null,
        "noise_strength": 5,
        "horizontal_patterns": {
            "frequencies": [
                {"freq": 500, "strength_dB": 5},
                {"freq": 1000, "strength_dB": 2},
                {"freq": 2000, "strength_dB": -5}
            ],
            "frequency_ranges": [
                {"freq_min": 300, "freq_max": 400, "strength_dB": 30, "distribution": "gaussian", "distribution_params": {"sigma": 50}},
                {"freq_min": 4000, "freq_max": 6000, "strength_dB": -10, "distribution": "gaussian", "distribution_params": {"sigma": 500}}
            ],
            "random_frequency_ranges": [
                {"freq_min": 500, "freq_max": 1000, "strength_dB": 40, "n": 10, "mode": "random"},
                {"freq_min": 2000, "freq_max": 3000, "strength_dB": -20, "n": 10, "mode": "random"}
            ]
        },
        "vertical_patterns": {
            "times": [
                {"time": 1, "strength_dB": 5},
                {"time": 2.7, "strength_dB": 10},
                {"time": 1.5, "strength_dB": 15}
            ],
            "time_ranges": [
                {"time_min": 3, "time_max": 5, "strength_dB": 20, "distribution": "gaussian", "distribution_params": {"sigma": 0.2}},
                {"time_min": 7, "time_max": 9, "strength_dB": -15, "distribution": "gaussian", "distribution_params": {"sigma": 0.5}}
            ]
        }
    },
    "eda_text_explain": "text"
}

```