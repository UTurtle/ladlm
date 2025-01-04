# main.py

import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from noise_pipeline import (
    SpectrogramModifier,
    ShapeFactory,
    PatternFactory,
    create_random_noise_pipeline,
    reconstruct_audio_from_final_spectrogram
)


def main():
    # Parameters
    sr = 16000
    duration = 12.0
    n_samples = int(sr * duration)
    signal = np.random.normal(-40, 1, n_samples)  # Adjusted mean

    # Randomly select SpectrogramModifier parameters
    noise_types = ['normal', 'uniform', 'perlin']
    noise_type = random.choice(noise_types)
    noise_strength = random.uniform(5, 15)
    noise_params = {}
    if noise_type == 'normal':
        noise_params = {'mean': 0.0, 'std': 1.0}
    elif noise_type == 'uniform':
        noise_params = {'low': -1.0, 'high': 1.0}
    elif noise_type == 'perlin':
        noise_params = {'seed': random.randint(0, 1000), 'scale': random.uniform(20.0, 100.0)}

    spectro_mod = SpectrogramModifier(
        sample_rate=sr,
        noise_strength=noise_strength,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Define shape and pattern ratios
    ratio_shape = {
        "circle": 0,
        "rectangle": 0,
        "ellipse": 0,
        "horizontal_spike": 1,
        "vertical_spike": 1,
        "fog": 0,
        "pillar": 0,
        "horizontal_line": 1,
        "vertical_line": 1,
        "horizontal_range_dist_db": 1,
        "vertical_range_dist_db": 1,
        "trapezoid": 0,
        "hill": 0,
        "wave_pattern": 0,
        "polygon": 0
    }

    ratio_pattern = {
        "linear": 0,
        "random": 0,
        "n_linear_repeat_t_sleep": 1,
        "convex": 0
    }

    # Create NoisePipeline with fixed number of shapes and patterns
    pipeline = create_random_noise_pipeline(
        spectro_mod,
        max_shapes=10,   # Fixed number of shapes
        max_patterns=2,  # Fixed number of patterns
        apply_blur=True,
        blur_sigma=1.0,
        duration=duration,
        sr=sr / 2,  # As per instruction
        freq_min=0,
        min_float_value=0.001,
        alpha=1.0,
        ratio_shape=ratio_shape,
        ratio_pattern=ratio_pattern,
        max_db_power=40,
        min_db_power=20
    )

    # Generate the final spectrogram with noise
    result_db = pipeline.generate(signal)
    print("NoisePipeline applied successfully.")

    # Create output directories
    output_dirs = {
        "audio": "output/audio",
        "spectrogram_with_axes": "output/linear_spectrogram_with_axes",
        "spectrogram_no_axes": "output/linear_spectrogram_no_axes",
        "json": "output/json"
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Define file identifiers
    file_id = "000000001"
    audio_file_path = os.path.join(output_dirs["audio"], f"automated_audio_{file_id}.wav")
    spectrogram_with_axes_path = os.path.join(
        output_dirs["spectrogram_with_axes"],
        f"automated_linear_spectrogram_with_axes_{file_id}.png"
    )
    spectrogram_no_axes_path = os.path.join(
        output_dirs["spectrogram_no_axes"],
        f"automated_linear_spectrogram_no_axes_{file_id}.png"
    )
    json_file_path = os.path.join(output_dirs["json"], f"automated_audio_{file_id}.json")

    # Plot and save spectrogram with axes
    fig_with_axes, _ = spectro_mod.plot_spectrogram(
        show_labels=True,
        title='Final Spectrogram with Full Random'
    )
    fig_with_axes.savefig(spectrogram_with_axes_path)
    plt.show()
    plt.close(fig_with_axes)
    print(f"Spectrogram with axes saved to {spectrogram_with_axes_path}")

    # Plot and save spectrogram without axes
    fig_no_axes, _ = spectro_mod.plot_spectrogram(
        show_labels=False,
        title='Final Spectrogram without Axes'
    )
    fig_no_axes.savefig(spectrogram_no_axes_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_no_axes)
    print(f"Spectrogram without axes saved to {spectrogram_no_axes_path}")

    # Reconstruct audio from the final spectrogram
    reconstructed = reconstruct_audio_from_final_spectrogram(spectro_mod, hop_length=512)

    # Save the reconstructed audio using soundfile
    sf.write(audio_file_path, reconstructed, int(sr))
    print(f"Reconstructed audio saved to {audio_file_path}")

    # Prepare JSON metadata
    extracted_data = {
        "file_path": audio_file_path,
        "file_name": os.path.basename(audio_file_path),
        "spectrogram_with_axes": spectrogram_with_axes_path,
        "spectrogram_no_axes": spectrogram_no_axes_path,
        "spectrogram_base": {
            "sample_rate": spectro_mod.sample_rate,
            "n_fft": spectro_mod.n_fft,
            "hop_length": spectro_mod.hop_length,
            "noise_strength": spectro_mod.noise_strength,
            "noise_type": spectro_mod.noise_type,
            "noise_params": spectro_mod.noise_params
        },
        "shapes": [
            {
                "type": shape.__class__.__name__,
                "parameters": shape.__dict__
            } for shape in pipeline.shapes
        ],
        "patterns": [
            {
                "type": pattern.__class__.__name__,
                "parameters": pattern.__dict__
            } for pattern in pipeline.patterns
        ],
    }

    # Save JSON metadata
    with open(json_file_path, 'w') as json_file:
        json.dump(extracted_data, json_file, indent=4)
    print(f"JSON metadata saved to {json_file_path}")

    # Optionally, play audio in Jupyter environment
    try:
        import IPython.display as ipd
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython:
            audio_widget = ipd.Audio(data=reconstructed, rate=int(sr))
            display(audio_widget)
    except (ImportError, NameError):
        print(
            "Jupyter environment not detected. Audio has been saved as a WAV file."
        )


if __name__ == "__main__":
    main()
