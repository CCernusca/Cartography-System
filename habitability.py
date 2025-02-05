
"""
Habitability calculation

Separate scores are calculated for each factor, and then combined

0 is a perfect score, as the score measures how far the data strays from the ideal values
"""

import json
import numpy as np
import helper_functions as hf
from area_mapping import get_water_mask, get_tree_mask

CONFIG = json.load(open("habitability_config.json", "r"))
# Perfect ratio of water
GOLDEN_WATER_RATIO = CONFIG["golden_water_ratio"]
# Perfect ratio of trees
GOLDEN_TREE_RATIO = CONFIG["golden_tree_ratio"]

def get_water_score(water_mask: np.ndarray) -> float:
    """
    Calculate the water score based on the difference between the actual water 
    coverage in the given mask and the ideal water-to-land ratio.

    Parameters:
    water_mask (numpy.ndarray): A binary mask indicating water areas.

    Returns:
    float: The difference between the actual water coverage percentage and the 
           golden water ratio.
    """
    return hf.mask_percentage_difference(water_mask, GOLDEN_WATER_RATIO)

def get_tree_score(tree_mask: np.ndarray) -> float:
    """
    Calculate the tree score based on the difference between the actual tree 
    coverage in the given mask and the golden tree ratio.

    Parameters:
    tree_mask (numpy.ndarray): A binary mask indicating tree areas.

    Returns:
    float: The difference between the actual tree coverage percentage and the 
           golden tree ratio.
    """
    return hf.mask_percentage_difference(tree_mask, GOLDEN_TREE_RATIO)

def get_score(scores: list, weights: list = None) -> float:
    """
    Calculate a weighted average score from a list of scores.

    Parameters:
    scores (list): A list of numerical scores to be averaged.
    weights (list, optional): A list of weights corresponding to each score. 
                              Defaults to equal weighting if not provided. 
                              Must be the same length as scores.

    Returns:
    float: The weighted average of the provided scores.

    Raises:
    ValueError: If the number of weights does not match the number of scores.
    """
    if weights is None:
        weights = [1] * len(scores)
    elif len(weights) != len(scores):
        raise ValueError("The number of weights must match the number of scores.")

    weighted_scores = [score * weight for score, weight in zip(scores, weights)]
    return np.mean(weighted_scores)

if __name__ == "__main__":
    image_input_path = "./mocking_examples/main2.png"

    water_mask = get_water_mask(image_input_path)
    water_score = get_water_score(water_mask)
    print(f"Water Score: {water_score}")

    tree_mask = get_tree_mask(image_input_path)
    tree_score = get_tree_score(tree_mask)
    print(f"Tree Score: {tree_score}")

    score = get_score([water_score, tree_score])
    print(f"Total Score: {score}")
