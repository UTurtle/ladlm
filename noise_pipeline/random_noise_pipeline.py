# noise_pipeline/random_noise_pipeline.py

import logging

from .utils import pick_item_from_ratio
from .shape_params import generate_shape_params
from .constants import DEFAULT_SHAPES, DEFAULT_PATTERNS
from .noise_pipeline import NoisePipeline
from .factories import ShapeFactory, PatternFactory

logger = logging.getLogger(__name__)

def create_random_noise_pipeline(
    spectro_mod,
    max_shapes=5,
    max_patterns=3,
    apply_blur=False,
    blur_sigma=1.0,
    duration=8.0,
    sr=16000,
    freq_min=20,
    min_float_value=0.001,
    alpha=1.0,
    ratio_shape=None,
    ratio_pattern=None,
    max_db_power=20,
    min_db_power=10
):
    """
    Create a NoisePipeline with randomly selected shapes and patterns based on provided ratios.

    Args:
        spectro_mod (SpectrogramModifier): The spectrogram modifier instance.
        max_shapes (int, optional): Maximum number of shapes to add. Defaults to 5.
        max_patterns (int, optional): Maximum number of patterns to add. Defaults to 3.
        apply_blur (bool, optional): Whether to apply Gaussian blur. Defaults to False.
        blur_sigma (float, optional): Sigma value for Gaussian blur. Defaults to 1.0.
        duration (float, optional): Duration of the audio in seconds. Defaults to 8.0.
        sr (int, optional): Sample rate. Defaults to 16000.
        freq_min (float, optional): Minimum frequency value. Defaults to 20.
        min_float_value (float, optional): Minimum float value for certain parameters. Defaults to 0.001.
        alpha (float, optional): Scaling factor. Defaults to 1.0.
        ratio_shape (dict, optional): Selection ratio for shapes. Defaults to None.
        ratio_pattern (dict, optional): Selection ratio for patterns. Defaults to None.
        max_db_power (float, optional): Maximum dB power for shapes. Defaults to 20.
        min_db_power (float, optional): Minimum dB power for shapes. Defaults to 10.

    Returns:
        NoisePipeline: Configured NoisePipeline instance.
    """
    logger.debug("Creating random NoisePipeline.")

    pipeline = NoisePipeline(
        spectro_mod,
        apply_blur=apply_blur,
        blur_sigma=blur_sigma
    )

    ratio_shape = ratio_shape or {s: 1 for s in DEFAULT_SHAPES}
    ratio_pattern = ratio_pattern or {p: 1 for p in DEFAULT_PATTERNS}

    shape_factory = ShapeFactory()
    pattern_factory = PatternFactory()

    # Add Shapes
    for _ in range(max_shapes):
        shape_name = pick_item_from_ratio(ratio_shape)
        try:
            shape_params = generate_shape_params(
                shape_name=shape_name,
                duration=duration,
                sr=sr,
                freq_min=freq_min,
                time_min=0.0,
                min_float_value=min_float_value,
                alpha=alpha,
                max_db_power=max_db_power,
                min_db_power=min_db_power
            )
            shape = shape_factory.create(shape_name, **shape_params)
            pipeline.add_shape(shape)
            logger.info(f"Added Shape: {shape_name} -> {shape}")
        except Exception as e:
            logger.error(f"Error configuring shape '{shape_name}': {e}.")

    # Add Patterns
    for _ in range(max_patterns):
        pattern_name = pick_item_from_ratio(ratio_pattern)
        try:
            # Select suitable shape names for the pattern
            if pattern_name == "n_linear_repeat_t_sleep":
                possible_shape_names = [
                    s for s in ratio_shape.keys()
                    if not s.endswith('_range_dist_db')
                ]
            else:
                possible_shape_names = [
                    s for s in ratio_shape.keys()
                    if s.startswith('horizontal_') or
                    s.startswith('vertical_') or
                    s.endswith('_range_dist_db')
                ]

            if not possible_shape_names:
                logger.warning("No suitable shapes available for the selected pattern.")
                continue

            shape_name = pick_item_from_ratio(
                {s: ratio_shape[s] for s in possible_shape_names}
            )

            shape_params = generate_shape_params(
                shape_name=shape_name,
                duration=duration,
                sr=sr,
                freq_min=freq_min,
                time_min=0.0,
                min_float_value=min_float_value,
                alpha=alpha,
                max_db_power=max_db_power,
                min_db_power=min_db_power
            )

            # Determine direction based on shape name
            if shape_name.startswith('horizontal_') or shape_name.endswith('_range_dist_db'):
                direction = 'freq'
            elif shape_name.startswith('vertical_') or shape_name.endswith('_range_dist_db'):
                direction = 'time'
            else:
                direction = 'time'

            pattern_kwargs = {
                "shape_name": shape_name,
                "shape_params": shape_params
            }

            if pattern_name == "linear":
                pattern_kwargs.update({
                    "direction": direction,
                    "repeat": 3,
                    "spacing": 1.0
                })
            elif pattern_name == "random":
                pattern_kwargs.update({
                    "n": 5,
                    "freq_range": (freq_min, sr),
                    "time_range": (0.0, duration)
                })
            elif pattern_name == "n_linear_repeat_t_sleep":
                pattern_kwargs.update({
                    "repeat": 3,
                    "repeat_time": 0.5,
                    "sleep_time": 1.0,
                    "start_time": 0.0,
                    "direction": direction
                })
            elif pattern_name == "convex":
                pattern_kwargs.update({
                    "freq_min": freq_min,
                    "freq_max": sr,
                    "time_min": 0.0,
                    "time_max": duration,
                    "n": 5
                })

            pattern = pattern_factory.create(pattern_name, pattern_kwargs)
            if pattern:
                pipeline.add_pattern(pattern)
                logger.info(f"Added Pattern: {pattern_name} -> {pattern}")
        except Exception as e:
            logger.error(f"Error configuring pattern '{pattern_name}': {e}.")

    logger.debug("Random NoisePipeline creation completed.")
    return pipeline
