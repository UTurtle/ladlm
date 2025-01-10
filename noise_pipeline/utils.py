# noise_pipeline/utils.py

import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

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
