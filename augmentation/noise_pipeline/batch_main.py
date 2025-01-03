# batch_main.py

import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

from noise_pipeline import (
    SpectrogramModifier,
    ShapeFactory,
    PatternFactory,
    create_random_noise_pipeline,
    reconstruct_audio_from_final_spectrogram
)


def batch_main(
    total_files=100000,
    batch_size=1000,  # Number of files per subdirectory (not utilized in this script)
    output_base_dir="output",
    sr=16000,
    duration=12.0
):
    """
    Generates multiple audio files with modified spectrograms.

    Parameters:
    - total_files (int): Total number of files to generate.
    - batch_size (int): Number of files per subdirectory.
    - output_base_dir (str): Base directory for all outputs.
    - sr (int): Sample rate.
    - duration (float): Duration of each audio file in seconds.
    """
    # Ensure necessary output subdirectories exist
    output_dirs = {
        "audio": os.path.join(output_base_dir, "audio"),
        "spectrogram_with_axes": os.path.join(output_base_dir, "linear_spectrogram_with_axes"),
        "spectrogram_no_axes": os.path.join(output_base_dir, "linear_spectrogram_no_axes"),
        "json": os.path.join(output_base_dir, "json")
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

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

    shape_factory = ShapeFactory()
    pattern_factory = PatternFactory()

    # Loop over the total number of files
    for i in tqdm(range(1, total_files + 1), desc="Generating Files"):
        # Generate a zero-padded file_id (e.g., '000000001' to '100000')
        file_id = f"{i:09d}"

        # Parameters for noise generation
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

        # Create NoisePipeline
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
        try:
            result_db = pipeline.generate(np.random.normal(-40, 1, int(sr * duration)))
        except Exception as e:
            print(f"Error generating spectrogram for file_id {file_id}: {e}")
            continue

        # Define file paths
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
        try:
            fig_with_axes, _ = spectro_mod.plot_spectrogram(
                show_labels=True,
                title='Final Spectrogram with Full Random'
            )
            fig_with_axes.savefig(spectrogram_with_axes_path)
            plt.close(fig_with_axes)
        except Exception as e:
            print(f"Error saving spectrogram with axes for file_id {file_id}: {e}")

        # Plot and save spectrogram without axes
        try:
            fig_no_axes, _ = spectro_mod.plot_spectrogram(
                show_labels=False,
                title='Final Spectrogram without Axes'
            )
            fig_no_axes.savefig(spectrogram_no_axes_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig_no_axes)
        except Exception as e:
            print(f"Error saving spectrogram without axes for file_id {file_id}: {e}")

        # Reconstruct audio from the final spectrogram
        try:
            reconstructed = reconstruct_audio_from_final_spectrogram(spectro_mod, hop_length=512)
            # Save the reconstructed audio using soundfile
            sf.write(audio_file_path, reconstructed, int(sr))
        except Exception as e:
            print(f"Error reconstructing/saving audio for file_id {file_id}: {e}")

        # Prepare JSON metadata
        try:
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
        except Exception as e:
            print(f"Error saving JSON metadata for file_id {file_id}: {e}")

    print("Batch generation completed successfully.")


if __name__ == "__main__":
    batch_main(
        total_files=10,      # Total number of files to generate
        batch_size=1,        # Not used in this script but can be utilized for advanced batching
        output_base_dir="output",
        sr=16000,
        duration=12.0
    )
