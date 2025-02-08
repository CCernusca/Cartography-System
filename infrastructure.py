
"""
Infrastructure generation

This module contains functions to generate infrastructure from masks
"""

import json
import numpy as np
import helper_functions as hf

def building_blueprints(sort: bool = True, debug: bool = False) -> dict:
    """
    Reads the building blueprints from a JSON file and returns them sorted by priority and size

    Parameters
    ----------
    sort : bool
        If true, the blueprints are sorted by priority and size in descending order

    Returns
    -------
    dict
        A dict of the building blueprints
    """
    with open("buildings.json", "r") as f:
        blueprints = json.load(f)
    
    if debug:
        hf.paste_debugging("(Building blueprints) Received blueprints")
    
    if sort:
        # Sorting by priority (descending) and then by size (descending)
        return sorted(
            blueprints, 
            key=lambda x: (x["priority"], x["size"][0] * x["size"][1]), 
            reverse=True
            )
    else:
        return blueprints

def _place_building(positions: np.ndarray, mask: np.ndarray, buildings_mask: np.ndarray, width: int, height: int, debug: bool = False) -> tuple[int, int]:
    """
    Places a building in the given mask and building mask and returns the coordinates if successful.

    Parameters
    ----------
    positions : np.ndarray
        A 2D array of positions, sorted by distance to the centroid of the mask
    mask : np.ndarray
        The mask in which to place the building
    buildings_mask : np.ndarray
        The mask of all already placed buildings
    width : int
        The width of the building to place
    height : int
        The height of the building to place
    debug : bool, optional
        If true, debugging messages are printed, by default False

    Returns
    -------
    tuple[int, int]
        The coordinates of the building if one is placed, otherwise (-1, -1)
    """
    # Iterate over already sorted positions
    for y, x in positions:

        # Check if rectangle fit within the mask-area
        if (x + width <= mask.shape[1]) and (y + height <= mask.shape[0]):
            if np.all(mask[y:y + height, x:x + width] > 0):
                # Check building collision
                if np.all(buildings_mask[y:y + height, x:x + width] == 0):

                    if debug:
                        hf.paste_debugging("(Building placement) Found position for building")

                    # Return correct coordinates
                    return x, y
    
    if debug:
        hf.paste_debugging("(Building placement) No position found for building")
    
    # No position found
    return -1, -1

def _place_buildings(blueprints: list, masks: dict[str, np.ndarray], min_distance: int = 10, debug: bool = False) -> tuple[list, np.ndarray]:
    """
    Places buildings according to the given blueprints in the given masks

    Parameters
    ----------
    blueprints : list
        A list of building blueprints
    masks : dict[str, np.ndarray]
        A dictionary of masks, where the keys are the names of the masks
    min_distance : int, optional
        The minimum distance between buildings, by default 10
    debug : bool, optional
        If true, debugging messages are printed, by default False

    Returns
    -------
    tuple[list, np.ndarray]
        A tuple of a list of the placed buildings and the building mask
    """
    placed_buildings = []
    building_mask = np.zeros_like(masks["free"])

    # Centroid-sorted positions
    positions = {}
    for mask_name in masks:
        positions[mask_name] = hf.centroid_sorted(masks[mask_name])
    
    if debug:
        hf.paste_debugging("(Building placement) Calculated positions")
    
    for blueprint in blueprints:
        # Get name
        nametag = blueprint["name"]
        # Get dimensions
        width, height = blueprint["size"][0], blueprint["size"][1]
        
        for _ in range(blueprint["amount"]):

            # Place building
            x, y = _place_building(positions[blueprint["location"]], masks[blueprint["location"]], building_mask, width, height, debug=debug)

            # Check if a position was found
            if x != -1 and y != -1:

                # Add building to list
                placed_buildings.append({"nametag": nametag, "rect": (x, y, width, height)})

                # Draw building to building mask, with min_distance padding, but avoid drawing out of bounds
                y_start = np.clip(y - min_distance, 0, building_mask.shape[0])
                y_end   = np.clip(y + blueprint["size"][1] + min_distance, 0, building_mask.shape[0])
                x_start = np.clip(x - min_distance, 0, building_mask.shape[1])
                x_end   = np.clip(x + blueprint["size"][0] + min_distance, 0, building_mask.shape[1])
                building_mask[y_start:y_end, x_start:x_end] = 1

            else:
                # Stop looking for postions for this blueprint if no position was found
                break
        
        if debug:
            hf.paste_debugging(f"(Building placement) Finished placing '{nametag}'")
    
    if debug:
        hf.paste_debugging("(Building placement) Buildings placed")
    
    return placed_buildings, building_mask

def generate_buildings(free_mask: np.ndarray, coast_mask: np.ndarray, inland_mask: np.ndarray, forest_edge_mask: np.ndarray, water_access_mask: np.ndarray, debug: bool = False) -> tuple[list, np.ndarray]:
    blueprints = building_blueprints(debug=debug)

    buildings, building_mask = _place_buildings(blueprints, 
                                    masks={
                                        "free": free_mask,
                                        "coast": coast_mask, 
                                        "inland": inland_mask, 
                                        "forest_edge": forest_edge_mask, 
                                        "water_access": water_access_mask
                                    },
                                    debug=debug
                                    )
    
    if debug:
        hf.paste_debugging("(Building generation) Buildings placed")

    return buildings, building_mask

if __name__ == "__main__":
    import classification as cl

    image_input_path = "./mocking_examples/main2.png"

    tree_mask, water_mask, free_mask, coast_mask, inland_mask, forest_edge_mask, water_access_mask = cl.generate_all_masks(image_input_path, debug=True)
    
    print(generate_buildings(free_mask, coast_mask, inland_mask, forest_edge_mask, water_access_mask, debug=True))
