# noise_pipeline/utils.py

import random
import numpy as np
import logging

from noise_pipeline.noise_pipeline import NoisePipeline 
from noise_pipeline import ShapeFactory, PatternFactory

from noise_pipeline.constants import DEFAULT_SHAPES, DEFAULT_PATTERNS

# Initialize logger for this module
logger = logging.getLogger(__name__)

def calculate_complexity_level(shapes, patterns):
    """
    Calculate the complexity level based on the number of unique shapes and patterns.
    
    Levels:
    1. 1 shape, 0 patterns
    2. 2 shapes, 1 pattern
    3. 3 shapes, 2 patterns
    4. 4 shapes, 3 patterns
    5. 5+ shapes, 4+ patterns
    """
    unique_shapes = set([shape.__class__.__name__ for shape in shapes])
    num_shapes = len(unique_shapes)
    num_patterns = len(patterns)
    
    logger.debug(f"Calculating complexity level: {num_shapes} unique shapes, {num_patterns} patterns.")
    
    if num_shapes >= 5 or num_patterns >= 4:
        level = 5
    elif num_shapes == 4 or num_patterns == 3:
        level = 4
    elif num_shapes == 3 or num_patterns == 2:
        level = 3
    elif num_shapes == 2 or num_patterns == 1:
        level = 2
    else:
        level = 1
    
    logger.info(f"Determined complexity level: {level}")
    return level

def pick_item_from_ratio(ratio_dict):
    """
    Picks an item from the ratio_dict based on the weights.
    
    Parameters:
    - ratio_dict (dict): Dictionary with items as keys and their corresponding weights as values.
    
    Returns:
    - selected_item: The chosen item based on the defined weights.
    """
    logger.debug(f"Picking item from ratio_dict: {ratio_dict}")
    items = list(ratio_dict.keys())
    weights = list(ratio_dict.values())
    total = sum(weights)
    if total == 0:
        logger.error("Total weight is zero in pick_item_from_ratio.")
        raise ValueError("Total weight cannot be zero.")
    r = random.uniform(0, total)
    s = 0
    for item, w in zip(items, weights):
        s += w
        if r <= s:
            logger.debug(f"Selected item: {item} with weight: {w}")
            return item
    selected_item = items[-1]
    logger.debug(f"Selected last item by default: {selected_item}")
    return selected_item

def generate_shape_params(
    shape_name,
    duration,
    sr,
    freq_min=20,
    time_min=0.0,
    min_float_value=0.001,
    alpha=1.0,
    max_db_power=20,
    min_db_power=10
):
    """
    Generates all parameters randomly based on the shape_name.
    Clamps time and frequency values within their specified boundaries.
    """
    logger.debug(f"Generating shape parameters for shape: {shape_name}")

    params = {
        "strength_dB": random.uniform(min_db_power, max_db_power)
    }

    # Generate frequency bounds
    random_freq_min = random.uniform(freq_min, sr)
    random_freq_max = random.uniform(freq_min, sr)
    if random_freq_min > random_freq_max:
        random_freq_min, random_freq_max = random_freq_max, random_freq_min

    # Generate time bounds
    random_time_min = random.uniform(time_min, duration)
    random_time_max = random.uniform(time_min, duration)
    if random_time_min > random_time_max:
        random_time_min, random_time_max = random_time_max, random_time_min

    # Define a helper function to clamp values
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))

    # Shape-specific parameter generation
    if shape_name == "horizontal_spike":
        params.update({
            "center_freq": clamp(
                random.uniform(random_freq_min, random_freq_max),
                0,
                sr
            ),
            "center_time": clamp(
                random.uniform(random_time_min, random_time_max),
                0,
                duration
            ),
            "radius_freq": clamp(
                random.uniform(30.0, 100.0),
                15.0,
                sr / 2
            ),
            "radius_time": clamp(
                random.uniform(random_time_max * 0.001, random_time_max * 0.1),
                0.05,
                duration * alpha
            )
        })
    elif shape_name == "vertical_spike":
        params.update({
            "center_freq": clamp(
                random.uniform(random_freq_min, random_freq_max) * 0.001,
                0,
                sr
            ),
            "center_time": clamp(
                random.uniform(random_time_min, random_time_max),
                0,
                duration
            ),
            "radius_freq": clamp(
                random.uniform(random_freq_max * 0.1, random_freq_max),
                0.1 * sr,
                sr
            ),
            "radius_time": clamp(
                random.uniform(random_time_max * 0.01, random_time_max * 0.1),
                0.05,
                duration * alpha
            )
        })
    elif shape_name == "trapezoid":
        params.update({
            "freq_min": clamp(random_freq_min, freq_min, sr),
            "freq_max": clamp(random_freq_max, freq_min, sr),
            "time_min": clamp(random_time_min, time_min, duration),
            "time_max": clamp(random_time_max, time_min, duration),
            "slope_freq": random.uniform(0.0, 2.0),
            "slope_time": random.uniform(0.0, 2.0)
        })
    elif shape_name == "fog":
        params.update({
            "coverage": clamp(random.uniform(0, 1), 0, 1)
        })
    elif shape_name == "hill":
        freq_center = random.uniform(random_freq_min, random_freq_max)
        center_time = random.uniform(random_time_min, random_time_max)
        freq_width = random.uniform(min_float_value, (random_freq_max - random_freq_min) * alpha)
        time_width = random.uniform(min_float_value, duration * alpha)

        params.update({
            "freq_center": clamp(freq_center, 0, sr),
            "time_center": clamp(center_time, 0, duration),
            "freq_width": clamp(freq_width, min_float_value, sr),
            "time_width": clamp(time_width, min_float_value, duration)
        })
    elif shape_name == "wave_pattern":
        params.update({
            "axis": random.choice(["time", "freq"]),
            "frequency": clamp(random.uniform(0.1, 10), 0.1, 10)
        })
    elif shape_name == "polygon":
        num_vertices = random.randint(3, 8)
        vertices = []
        for _ in range(num_vertices):
            f_rand = random.uniform(random_freq_min, random_freq_max)
            t_rand = random.uniform(random_time_min, random_time_max)
            f_rand = clamp(f_rand, 0, sr)
            t_rand = clamp(t_rand, 0, duration)
            vertices.append([f_rand, t_rand])
        params.update({
            "vertices": vertices
        })
    elif shape_name == "rectangle":
        params.update({
            "freq_min": clamp(random_freq_min, freq_min, sr),
            "freq_max": clamp(random_freq_max, freq_min, sr),
            "time_min": clamp(random_time_min, time_min, duration),
            "time_max": clamp(random_time_max, time_min, duration)
        })
    elif shape_name == "pillar":
        params.update({
            "freq_min": clamp(random_freq_min, freq_min, sr),
            "freq_max": clamp(random_freq_max, freq_min, sr)
        })
    elif shape_name == "horizontal_line":
        center_freq = random.uniform(random_freq_min, random_freq_max)
        params.update({
            "center_freq": clamp(center_freq, 0, sr),
            "thickness": random.randint(1, 3)
        })
    elif shape_name == "vertical_line":
        center_time = random.uniform(random_time_min, random_time_max)
        params.update({
            "center_time": clamp(center_time, 0, duration),
            "thickness": random.randint(1, 3)
        })
    elif shape_name == "horizontal_range_dist_db":
        dist_options = ["gaussian"]
        chosen_dist = random.choice(dist_options)
        freq_min_range = clamp(
            random_freq_min - 0.1,  # Adjusted threshold
            0,
            sr
        )
        freq_max_range = clamp(
            random_freq_min + 0.1,  # Adjusted threshold
            0,
            sr
        )
        sigma = clamp(
            random.uniform(1000.0 * alpha, 1100.0 * alpha),
            1000.0 * alpha,
            1100.0 * alpha
        )

        params.update({
            "freq_min": freq_min_range,
            "freq_max": freq_max_range,
            "distribution": chosen_dist,
            "distribution_params": {"sigma": sigma}
        })
    elif shape_name == "vertical_range_dist_db":
        dist_options = ["gaussian"]
        chosen_dist = random.choice(dist_options)
        time_min_range = clamp(
            random_time_min - 0.1,  # Adjusted threshold
            0,
            duration
        )
        time_max_range = clamp(
            random_time_max + 0.1,  # Adjusted threshold
            0,
            duration
        )
        sigma = clamp(
            random.uniform(1.0 * alpha, 2.0 * alpha),
            1.0 * alpha,
            2.0 * alpha
        )

        params.update({
            "time_min": time_min_range,
            "time_max": time_max_range,
            "distribution": chosen_dist,
            "distribution_params": {"sigma": sigma}
        })
    elif shape_name == "ellipse":
        center_freq = random.uniform(random_freq_min, random_freq_max)
        center_time = random.uniform(random_time_min, random_time_max)
        radius_freq = random.uniform(10, (random_freq_max - random_freq_min) * alpha)
        radius_time = random.uniform(min_float_value, duration * alpha)

        params.update({
            "center_freq": clamp(center_freq, 0, sr),
            "center_time": clamp(center_time, 0, duration),
            "radius_freq": clamp(radius_freq, 10, sr),
            "radius_time": clamp(radius_time, min_float_value, duration)
        })
    else:
        # Default handling for shapes like circle
        center_freq = random.uniform(random_freq_min, random_freq_max)
        center_time = random.uniform(random_time_min, random_time_max)
        radius_freq = random.uniform(10, (random_freq_max - random_freq_min) * alpha)
        radius_time = random.uniform(min_float_value, duration * alpha)

        params.update({
            "center_freq": clamp(center_freq, 0, sr),
            "center_time": clamp(center_time, 0, duration),
            "radius_freq": clamp(radius_freq, 10, sr),
            "radius_time": clamp(radius_time, min_float_value, duration)
        })

    logger.debug(f"Generated parameters for shape '{shape_name}': {params}")

    return params

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
    
    Parameters:
    - spectro_mod (SpectrogramModifier): The spectrogram modifier instance.
    - max_shapes (int): Maximum number of shapes to add.
    - max_patterns (int): Maximum number of patterns to add.
    - apply_blur (bool): Whether to apply Gaussian blur.
    - blur_sigma (float): Sigma value for Gaussian blur.
    - duration (float): Duration of the audio in seconds.
    - sr (int): Sample rate.
    - freq_min (float): Minimum frequency value.
    - min_float_value (float): Minimum float value for certain parameters.
    - alpha (float): Scaling factor.
    - ratio_shape (dict): Dictionary defining the selection ratio for shapes.
    - ratio_pattern (dict): Dictionary defining the selection ratio for patterns.
    - max_db_power (float): Maximum dB power for shapes.
    - min_db_power (float): Minimum dB power for shapes.
    
    Returns:
    - pipeline (NoisePipeline): Configured NoisePipeline instance.
    """
    logger.debug("Creating random NoisePipeline.")

    pipeline = NoisePipeline(
        spectro_mod,
        apply_blur=apply_blur,
        blur_sigma=blur_sigma
    )

    if ratio_shape is None:
        ratio_shape = {s: 1 for s in DEFAULT_SHAPES}
    if ratio_pattern is None:
        ratio_pattern = {p: 1 for p in DEFAULT_PATTERNS}

    shape_factory = pipeline.shape_factory
    pattern_factory = pipeline.pattern_factory

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
                    "repeat": 3,  # Fixed number
                    "spacing": 1.0  # Fixed spacing
                })
            elif pattern_name == "random":
                pattern_kwargs.update({
                    "n": 5,  # Fixed number
                    "freq_range": (freq_min, sr),
                    "time_range": (0.0, duration)
                })
            elif pattern_name == "n_linear_repeat_t_sleep":
                pattern_kwargs.update({
                    "repeat": 3,  # Fixed number
                    "repeat_time": 0.5,  # Fixed repeat time
                    "sleep_time": 1.0,  # Fixed sleep time
                    "start_time": 0.0,  # Fixed start time
                    "direction": direction
                })
            elif pattern_name == "convex":
                pattern_kwargs.update({
                    "freq_min": freq_min,
                    "freq_max": sr,
                    "time_min": 0.0,
                    "time_max": duration,
                    "n": 5  # Fixed number
                })

            pattern = pattern_factory.create(pattern_name, pattern_kwargs)
            if pattern:
                pipeline.add_pattern(pattern)
                logger.info(f"Added Pattern: {pattern_name} -> {pattern}")
        except Exception as e:
            logger.error(f"Error configuring pattern '{pattern_name}': {e}.")

    logger.debug("Random NoisePipeline creation completed.")
    return pipeline
