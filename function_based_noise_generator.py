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

from noise_pipeline.constants import (
    RATIO_SHAPE_BASE,
    RATIO_PATTERN_BASE,
    SHAPE_TYPE_MAPPING,
    PATTERN_TYPE_MAPPING
)
from noise_pipeline import (
    SpectrogramModifier,
    NoisePipeline,
    ShapeFactory,
    PatternFactory,
    create_random_noise_pipeline,
    reconstruct_audio_from_final_spectrogram,
    calculate_complexity_level
)
from noise_pipeline.utils import calculate_complexity_level, pick_item_from_ratio
from noise_pipeline.shape_params import generate_shape_params
from noise_pipeline.random_noise_pipeline import create_random_noise_pipeline


def setup_logging(disable_logging=False):
    if disable_logging:
        logging.disable(logging.CRITICAL)
    else:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'batch_level_main.log')
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        utils_logger = logging.getLogger('noise_pipeline.utils')
        utils_logger.setLevel(logging.WARNING)


def validate_parameters(total_files, sr, duration):
    logger = logging.getLogger(__name__)
    if total_files < 1:
        logger.error("total_files must be at least 1.")
        raise ValueError("total_files must be at least 1.")
    if sr <= 0:
        logger.error("Sample rate (sr) must be positive.")
        raise ValueError("Sample rate (sr) must be positive.")
    if duration <= 0:
        logger.error("Duration must be positive.")
        raise ValueError("Duration must be positive.")


def create_output_dirs(output_base_dir):
    logger = logging.getLogger(__name__)
    output_dirs = {
        "audio": os.path.join(output_base_dir, "audio"),
        "spectrogram_with_axes": os.path.join(
            output_base_dir, "linear_spectrogram_with_axes"
        ),
        "spectrogram_no_axes": os.path.join(
            output_base_dir, "linear_spectrogram_no_axes"
        ),
        "json": os.path.join(output_base_dir, "json")
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured existence of directory: {dir_path}")
    return output_dirs


def distribute_files_across_levels(total_files, max_level):
    files_per_level = total_files // max_level
    remaining_files = total_files % max_level
    level_counts = {level: files_per_level for level in range(1, max_level + 1)}
    for level in range(1, max_level + 1):
        if remaining_files > 0:
            level_counts[level] += 1
            remaining_files -= 1
        else:
            break
    return level_counts


def generate_noise_parameters(noise_types):
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
    return noise_type, noise_strength, noise_params


def initialize_spectrogram_modifier(sr, noise_strength, noise_type,
                                   noise_params, window, n_fft, hop_length):
    return SpectrogramModifier(
        sample_rate=sr,
        noise_strength=noise_strength,
        noise_type=noise_type,
        noise_params=noise_params,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length
    )


def create_noise_pipeline(spectro_mod, max_shapes, max_patterns, duration,
                         sr, ratio_shape, ratio_pattern):
    return create_random_noise_pipeline(
        spectro_mod,
        max_shapes=max_shapes,
        max_patterns=max_patterns,
        apply_blur=True,
        blur_sigma=1.0,
        duration=duration,
        sr=sr // 2,
        freq_min=0,
        min_float_value=0.001,
        alpha=1.0,
        ratio_shape=ratio_shape,
        ratio_pattern=ratio_pattern,
        max_db_power=40,
        min_db_power=20
    )


def generate_spectrogram(pipeline, sr, duration):
    return pipeline.generate(
        np.random.normal(-40, 1, int(sr * duration))
    )


def save_spectrogram(spectro_mod, show_labels, title, save_path):
    fig, _ = spectro_mod.plot_spectrogram(
        show_labels=show_labels, title=title
    )
    if not show_labels:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        fig.savefig(save_path)
    plt.close(fig)


def reconstruct_and_save_audio(spectro_mod, audio_file_path, sr):
    reconstructed = reconstruct_audio_from_final_spectrogram(spectro_mod)
    sf.write(audio_file_path, reconstructed, int(sr))


def save_json_metadata(json_file_path, audio_file_path,
                       spectrogram_with_axes_path,
                       spectrogram_no_axes_path, spectro_mod, pipeline,
                       actual_level, duration):
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
        "complexity_level": actual_level,
        "shape_count": len(
            set(shape.__class__.__name__ for shape in pipeline.shapes)
        ),
        "pattern_count": len(pipeline.patterns),
        "duration": duration
    }
    with open(json_file_path, 'w') as json_file:
        json.dump(extracted_data, json_file, indent=4)


def log_save_paths(logger, spectrogram_with_axes_path,
                   spectrogram_no_axes_path, audio_file_path,
                   json_file_path, file_id):
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
        logger.warning(
            f"Some files were not saved correctly for file_id {file_id}."
        )


def batch_level_main(max_level=5, total_files=1000,
                    output_base_dir="output", sr=16000,
                    duration=3.0, n_fft=256, hop_length=256,
                    window='hann'):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch generation of {total_files} files.")

    validate_parameters(total_files, sr, duration)
    output_dirs = create_output_dirs(output_base_dir)

    ratio_shape_base = RATIO_SHAPE_BASE.copy()
    ratio_pattern_base = RATIO_PATTERN_BASE.copy()

    shape_factory = ShapeFactory()
    pattern_factory = PatternFactory()

    level_counts = distribute_files_across_levels(total_files, max_level)
    logger.info(f"Files distribution across levels: {level_counts}")

    noise_types = ['normal', 'uniform', 'perlin']
    file_idx = 1

    for level, count in level_counts.items():
        logger.info(f"Generating {count} files for Level {level}.")
        for _ in tqdm(range(1, count + 1),
                    desc=f"Generating Level {level} Files"):
            file_id = f"level_{level}_{file_idx:09d}"
            file_idx += 1

            noise_type, noise_strength, noise_params = (
                generate_noise_parameters(noise_types)
            )
            logger.debug(
                f"File ID: {file_id} - Noise type: {noise_type}, "
                f"Strength: {noise_strength}"
            )
            logger.debug(f"Noise parameters: {noise_params}")

            spectro_mod = initialize_spectrogram_modifier(
                sr, noise_strength, noise_type, noise_params,
                window, n_fft, hop_length
            )

            ratio_shape = {
                k: v if v else 0 for k, v in ratio_shape_base.items()
            }
            ratio_pattern = {
                k: v if v else 0 for k, v in ratio_pattern_base.items()
            }

            max_shapes = level
            max_patterns = level

            logger.debug(
                f"Creating NoisePipeline with max_shapes={max_shapes}, "
                f"max_patterns={max_patterns}"
            )
            try:
                pipeline = create_noise_pipeline(
                    spectro_mod, max_shapes, max_patterns, duration,
                    sr, ratio_shape, ratio_pattern
                )
                logger.debug(f"Created NoisePipeline for file_id {file_id}.")
            except Exception as e:
                logger.error(
                    f"Error creating NoisePipeline for file_id {file_id}: {e}"
                )
                continue

            try:
                result_db = generate_spectrogram(pipeline, sr, duration)
                logger.debug(f"Spectrogram generated for file_id {file_id}.")
            except Exception as e:
                logger.error(
                    f"Error generating spectrogram for file_id {file_id}: {e}"
                )
                continue

            actual_level = calculate_complexity_level(
                pipeline.shapes, pipeline.patterns
            )
            if actual_level != level:
                logger.warning(
                    f"Desired level: {level}, but achieved level: "
                    f"{actual_level} for file_id {file_id}"
                )
            else:
                logger.info(
                    f"Achieved desired complexity level: {actual_level} "
                    f"for file_id {file_id}"
                )

            audio_file_path = os.path.join(
                output_dirs["audio"], f"automated_audio_{file_id}.wav"
            )
            spectrogram_with_axes_path = os.path.join(
                output_dirs["spectrogram_with_axes"],
                f"automated_linear_spectrogram_with_axes_{file_id}.png"
            )
            spectrogram_no_axes_path = os.path.join(
                output_dirs["spectrogram_no_axes"],
                f"automated_linear_spectrogram_no_axes_{file_id}.png"
            )
            json_file_path = os.path.join(
                output_dirs["json"], f"automated_audio_{file_id}.json"
            )

            try:
                save_spectrogram(
                    spectro_mod,
                    show_labels=True,
                    title=f'Final Spectrogram Level {actual_level}',
                    save_path=spectrogram_with_axes_path
                )
                logger.debug(
                    f"Saved spectrogram with axes for file_id {file_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error saving spectrogram with axes for file_id "
                    f"{file_id}: {e}"
                )

            try:
                save_spectrogram(
                    spectro_mod,
                    show_labels=False,
                    title=f'Final Spectrogram Level {actual_level}',
                    save_path=spectrogram_no_axes_path
                )
                logger.debug(
                    f"Saved spectrogram without axes for file_id {file_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error saving spectrogram without axes for file_id "
                    f"{file_id}: {e}"
                )

            try:
                reconstruct_and_save_audio(
                    spectro_mod, audio_file_path, sr
                )
                logger.debug(
                    f"Reconstructed and saved audio for file_id {file_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error reconstructing/saving audio for file_id "
                    f"{file_id}: {e}"
                )

            try:
                save_json_metadata(
                    json_file_path, audio_file_path,
                    spectrogram_with_axes_path, spectrogram_no_axes_path,
                    spectro_mod, pipeline, actual_level, duration
                )
                logger.debug(
                    f"Saved JSON metadata for file_id {file_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error saving JSON metadata for file_id {file_id}: {e}"
                )

            try:
                log_save_paths(
                    logger, spectrogram_with_axes_path,
                    spectrogram_no_axes_path, audio_file_path,
                    json_file_path, file_id
                )
            except Exception as e:
                logger.error(
                    f"Error during logging combined save paths for file_id "
                    f"{file_id}: {e}"
                )


def handle_existing_output_dir(output_dir):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        response = input(
            f"The output directory '{output_dir}' already exists and is not "
            "empty. Do you want to remove its contents? (y/n): "
        ).strip().lower()
        if response == 'y':
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            print(f"Contents of '{output_dir}' have been removed.")
        else:
            print("Existing files will be retained.")


def main():
    disable_logging = True

    max_level = 1
    total_files = 1000
    output_dir = "augmentation/output"

    sr = 16000
    duration = 3.0
    n_fft = 256
    hop_length = 256
    window = 'hann'

    setup_logging(disable_logging=disable_logging)
    handle_existing_output_dir(output_dir)
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
