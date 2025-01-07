# main.py

import os
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import logging
from tqdm import tqdm
import librosa
import librosa.display

# for import noise_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from noise_pipeline.constants import (
    RATIO_SHAPE_BASE,
    RATIO_PATTERN_BASE,
    SHAPE_TYPE_MAPPING,
    PATTERN_TYPE_MAPPING
)

from noise_pipeline import (
    SpectrogramModifier,
    ShapeFactory,
    PatternFactory,
    create_random_noise_pipeline,
    reconstruct_audio_from_final_spectrogram,
    calculate_complexity_level
)
from noise_pipeline.utils import pick_item_from_ratio, generate_shape_params

def setup_logging(disable_logging=False):
    """
    Configures the logging settings.
    - If disable_logging is True, disables all logging.
    - Else, sets up logging with DEBUG level for file and INFO level for console.
    - Specific modules can have their logging levels adjusted to reduce verbosity.
    """
    if disable_logging:
        logging.disable(logging.CRITICAL)
    else:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create a custom logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Set the root logger level to DEBUG

        # Formatter for logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler for detailed logs
        file_handler = logging.FileHandler(os.path.join(log_dir, 'batch_level_main.log'))
        file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler for console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Log INFO and above to the console
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Adjust specific module loggers to reduce verbosity
        # Set 'noise_pipeline.utils' logger to WARNING to exclude its DEBUG and INFO logs
        utils_logger = logging.getLogger('noise_pipeline.utils')
        utils_logger.setLevel(logging.WARNING)


def batch_level_main(
    max_level=5,
    total_files=1000,
    output_base_dir="output",
    sr=16000,
    duration=3.0,
    n_fft=256,
    hop_length=256,
    window='hann'
):
    """
    Generates multiple audio files with modified spectrograms, evenly distributed across complexity levels 1-5.

    Parameters:
    - total_files (int): Total number of files to generate.
    - output_base_dir (str): Base directory for all outputs.
    - sr (int): Sample rate.
    - duration (float): Duration of each audio file in seconds.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch generation of {total_files} files.")

    # Validate input parameters
    if total_files < 1:
        logger.error("total_files must be at least 1.")
        raise ValueError("total_files must be at least 1.")
    if sr <= 0:
        logger.error("Sample rate (sr) must be positive.")
        raise ValueError("Sample rate (sr) must be positive.")
    if duration <= 0:
        logger.error("Duration must be positive.")
        raise ValueError("Duration must be positive.")

    # Ensure necessary output subdirectories exist
    output_dirs = {
        "audio": os.path.join(output_base_dir, "audio"),
        "spectrogram_with_axes": os.path.join(output_base_dir, "linear_spectrogram_with_axes"),
        "spectrogram_no_axes": os.path.join(output_base_dir, "linear_spectrogram_no_axes"),
        "json": os.path.join(output_base_dir, "json")
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured existence of directory: {dir_path}")

    # Define shape and pattern ratios
    # These ratios are generalized; adjust as needed to match desired complexity distribution
    ratio_shape_base = RATIO_SHAPE_BASE.copy()
    ratio_pattern_base = RATIO_PATTERN_BASE.copy()

    shape_factory = ShapeFactory()
    pattern_factory = PatternFactory()

    # Calculate the number of files per level
    files_per_level = total_files // max_level
    remaining_files = total_files % max_level
    level_counts = {level: files_per_level for level in range(1, max_level+1)}
    # Distribute remaining files
    for level in range(1, max_level+1):
        if remaining_files > 0:
            level_counts[level] += 1
            remaining_files -= 1
        else:
            break

    logger.info(f"Files distribution across levels: {level_counts}")

    file_idx = 1
    # Loop over each level and generate the corresponding number of files
    for level, count in level_counts.items():
        logger.info(f"Generating {count} files for Level {level}.")
        for i in tqdm(range(1, count + 1), desc=f"Generating Level {level} Files"):
            # Generate a zero-padded file_id based on the current index within the level
            # Format: 'level_{level}_{index:09d}' e.g., 'level_1_000000001'
            file_id = f"level_{level}_{file_idx:09d}"
            file_idx += 1

            # Parameters for noise generation
            noise_types = ['normal', 'uniform', 'perlin']
            noise_type = random.choice(noise_types)
            noise_strength = random.uniform(5, 15)
            noise_params = {}
            seed = random.randint(0, int(1e9))
            if noise_type == 'normal':
                noise_params = {'mean': 0.0, 'std': 1.0, 'seed': seed}
            elif noise_type == 'uniform':
                noise_params = {'low': -1.0, 'high': 1.0, 'seed': seed}
            elif noise_type == 'perlin':
                noise_params = {'seed': seed, 'scale': random.uniform(20.0, 100.0)}

            logger.debug(f"File ID: {file_id} - Noise type: {noise_type}, Strength: {noise_strength}")
            logger.debug(f"Noise parameters: {noise_params}")

            spectro_mod = SpectrogramModifier(
                sample_rate=sr,
                noise_strength=noise_strength,
                noise_type=noise_type,
                noise_params=noise_params,
                window=window,
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Define shape and pattern ratios based on level
            # Higher levels have more shapes and patterns
            ratio_shape = {k: v if v else 0 for k, v in ratio_shape_base.items()}
            ratio_pattern = {k: v if v else 0 for k, v in ratio_pattern_base.items()}

            # Adjust ratios based on the desired level
            # For simplicity, incrementally add more shapes and patterns as the level increases
            if level >= 1:
                ratio_shape["horizontal_spike"] = 1
                ratio_shape["vertical_spike"] = 1
            if level >= 2:
                ratio_shape["horizontal_line"] = 1
                ratio_shape["vertical_line"] = 1
                ratio_pattern["linear"] = 1
            if level >= 3:
                ratio_shape["horizontal_range_dist_db"] = 1
                ratio_shape["vertical_range_dist_db"] = 1
                ratio_pattern["random"] = 1
            if level >= 4:
                ratio_shape["trapezoid"] = 1
                ratio_shape["hill"] = 1
                ratio_pattern["n_linear_repeat_t_sleep"] = 1
            if level >= 5:
                ratio_shape["fog"] = 1
                ratio_shape["pillar"] = 1
                ratio_shape["wave_pattern"] = 1
                ratio_shape["polygon"] = 1
                ratio_pattern["convex"] = 1

            # Determine the number of shapes and patterns based on the level
            max_shapes = level  # Number of unique shapes corresponds to level
            max_patterns = level - 1  # One fewer pattern than level

            logger.debug(f"Creating NoisePipeline with max_shapes={max_shapes}, max_patterns={max_patterns}")
            try:
                pipeline = create_random_noise_pipeline(
                    spectro_mod,
                    max_shapes=max_shapes,
                    max_patterns=max_patterns,
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
                logger.debug(f"Created NoisePipeline for file_id {file_id}.")
            except Exception as e:
                logger.error(f"Error creating NoisePipeline for file_id {file_id}: {e}")
                continue

            # Generate the final spectrogram with noise
            try:
                result_db = pipeline.generate(np.random.normal(-40, 1, int(sr * duration)))
                logger.debug(f"Spectrogram generated for file_id {file_id}.")
            except Exception as e:
                logger.error(f"Error generating spectrogram for file_id {file_id}: {e}")
                continue

            # Calculate complexity level to ensure it matches the desired level
            actual_level = calculate_complexity_level(pipeline.shapes, pipeline.patterns)
            if actual_level != level:
                logger.warning(f"Desired level: {level}, but achieved level: {actual_level} for file_id {file_id}")
            else:
                logger.info(f"Achieved desired complexity level: {actual_level} for file_id {file_id}")

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
                    title=f'Final Spectrogram Level {actual_level}'
                )
                fig_with_axes.savefig(spectrogram_with_axes_path)
                plt.close(fig_with_axes)
            except Exception as e:
                logger.error(f"Error saving spectrogram with axes for file_id {file_id}: {e}")

            # Plot and save spectrogram without axes
            try:
                fig_no_axes, _ = spectro_mod.plot_spectrogram(
                    show_labels=False,
                    title=f'Final Spectrogram Level {actual_level}'
                )
                fig_no_axes.savefig(spectrogram_no_axes_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig_no_axes)
            except Exception as e:
                logger.error(f"Error saving spectrogram without axes for file_id {file_id}: {e}")

            # Reconstruct audio from the final spectrogram
            try:
                reconstructed = reconstruct_audio_from_final_spectrogram(spectro_mod)
                # Save the reconstructed audio using soundfile
                sf.write(audio_file_path, reconstructed, int(sr))
            except Exception as e:
                logger.error(f"Error reconstructing/saving audio for file_id {file_id}: {e}")

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
                            "type": SHAPE_TYPE_MAPPING[shape.__class__.__name__],
                            "parameters": shape.__dict__
                        } for shape in pipeline.shapes
                    ],
                    "patterns": [
                        {
                            "type": PATTERN_TYPE_MAPPING[pattern.__class__.__name__],
                            "parameters": pattern.__dict__
                        } for pattern in pipeline.patterns
                    ],
                    "complexity_level": actual_level,  # Include complexity level
                    "shape_count": len(set([shape.__class__.__name__ for shape in pipeline.shapes])),  # Number of unique shapes
                    "pattern_count": len(pipeline.patterns),  # Number of patterns
                    "duration": duration 
                }

                # Save JSON metadata
                with open(json_file_path, 'w') as json_file:
                    json.dump(extracted_data, json_file, indent=4)
            except Exception as e:
                logger.error(f"Error saving JSON metadata for file_id {file_id}: {e}")

            # 통합된 로그 메시지
            try:
                if all([
                    os.path.exists(spectrogram_with_axes_path),
                    os.path.exists(spectrogram_no_axes_path),
                    os.path.exists(audio_file_path),
                    os.path.exists(json_file_path)
                ]):
                    logger.info(
                        f"Saved spectrogram with axes to {spectrogram_with_axes_path}, "
                        f"spectrogram without axes to {spectrogram_no_axes_path}, "
                        f"reconstructed audio to {audio_file_path}, "
                        f"and JSON metadata to {json_file_path}."
                    )
                else:
                    logger.warning(f"Some files were not saved correctly for file_id {file_id}.")
            except Exception as e:
                logger.error(f"Error during logging combined save paths for file_id {file_id}: {e}")


def main():
    max_level = 3
    total_files = 10000
    output_dir = "output"
    sr = 16000
    duration = 3.0
    disable_logging = False  

    # librosa 파라미터
    n_fft = 256
    hop_length = 256
    window = 'hann'

    # 로깅 설정
    setup_logging(disable_logging=disable_logging)

    # batch_level_main 함수 호출
    batch_level_main(
        max_level=max_level,
        total_files=total_files,
        output_base_dir=output_dir,
        sr=sr,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )

if __name__ == "__main__":
    main()
