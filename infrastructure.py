
"""
Infrastructure generation

This module contains functions to generate infrastructure from masks
"""

import json
import numpy as np
import helper_functions as hf
import networkx as nx
import matplotlib.pyplot as plt

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
    padding_mask = np.zeros_like(masks["free"])

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
            x, y = _place_building(positions[blueprint["location"]], masks[blueprint["location"]], padding_mask, width, height, debug=debug)

            # Check if a position was found
            if x != -1 and y != -1:

                # Add building to list
                placed_buildings.append({"nametag": nametag, "rect": (x, y, width, height)})

                # Draw building to building mask
                building_mask[y:y + height, x:x + width] = 1

                # Draw building to padding mask, with min_distance padding, but avoid drawing out of bounds
                y_start = np.clip(y - min_distance, 0, building_mask.shape[0])
                y_end   = np.clip(y + blueprint["size"][1] + min_distance, 0, building_mask.shape[0])
                x_start = np.clip(x - min_distance, 0, building_mask.shape[1])
                x_end   = np.clip(x + blueprint["size"][0] + min_distance, 0, building_mask.shape[1])
                padding_mask[y_start:y_end, x_start:x_end] = 1

            else:
                # Stop looking for postions for this blueprint if no position was found
                break
        
        if debug:
            hf.paste_debugging(f"(Building placement) Finished placing '{nametag}'")
    
    if debug:
        hf.paste_debugging("(Building placement) Buildings placed")
    
    return placed_buildings, building_mask, padding_mask

def generate_buildings(free_mask: np.ndarray, coast_mask: np.ndarray, inland_mask: np.ndarray, forest_edge_mask: np.ndarray, water_access_mask: np.ndarray, debug: bool = False) -> tuple[list, np.ndarray]:
    blueprints = building_blueprints(debug=debug)

    buildings, building_mask, padding_mask = _place_buildings(blueprints, 
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

    return buildings, building_mask, padding_mask

def _building_centers(buildings: list) -> np.ndarray:
    return np.array([(building["rect"][0] + building["rect"][2] // 2, building["rect"][1] + building["rect"][3] // 2) for building in buildings])

def _generate_path_tree(buildings: list, max_length: int|None = None, debug: bool = False) -> list[tuple[tuple]]:
    centers = _building_centers(buildings)

    # Generate paths between all centers using Delauney
    edges = hf.custom_delaunay(centers)

    # Create a graph from the edges
    graph = nx.Graph()
    for edge in edges:
        p1, p2 = centers[edge[0]], centers[edge[1]]
        distance = np.linalg.norm(p1 - p2)
        if max_length is None or distance <= max_length:  # Only add edges that are within the max length
            graph.add_edge(edge[0], edge[1], weight=distance)

    # Create minimum spanning tree of graph
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    
    paths = []
    for edge in mst.edges():
        p1, p2 = centers[edge[0]], centers[edge[1]]
        paths.append((p1, p2))

    if debug:
        hf.paste_debugging("(Path tree generation) Generated path tree")

    return paths

def generate_path_points(buildings: list, masks_and_cost_multipliers: dict[str, tuple[np.ndarray, float]], resolution_factor: float = 1, max_distance: int|None = None, debug: bool = False) -> list[list[tuple]]:
    path_tree = _generate_path_tree(buildings, max_length=max_distance, debug=debug)

    masks = {name: hf.scale_mask(masks_and_cost_multipliers[name][0], resolution_factor) for name in masks_and_cost_multipliers}
    cost_multipliers = {name: masks_and_cost_multipliers[name][1] for name in masks_and_cost_multipliers}

    path_points = []
    bridge_points = []
    for p1, p2 in path_tree:
        p1, p2 = hf.get_connectors_from_centers(p1, p2, masks_and_cost_multipliers['buildings'][0])
        p1 = (int(p1[0] * resolution_factor), int(p1[1] * resolution_factor))
        p2 = (int(p2[0] * resolution_factor), int(p2[1] * resolution_factor))
        points = hf.astar(p1, p2, masks=masks, cost_multipliers=cost_multipliers)
        path_points.append([(x // resolution_factor, y // resolution_factor) for x, y in points])
        bridge_points.extend([(x // resolution_factor, y // resolution_factor) for x, y in points if masks['water'][y][x] > 0])

    if debug:
        hf.paste_debugging("(Path generation) Generated paths")
    
    return path_points, bridge_points

if __name__ == "__main__":
    import classification as cl

    image_input_path = "./mocking_examples/main2.png"

    tree_mask, water_mask, free_mask, coast_mask, inland_mask, forest_edge_mask, water_access_mask = cl.generate_all_masks(image_input_path, debug=True)
    
    buildings, building_mask, padding_mask = generate_buildings(free_mask, coast_mask, inland_mask, forest_edge_mask, water_access_mask, debug=True)

    path_points, bridge_points = generate_path_points(buildings, masks_and_cost_multipliers={
        "free": (free_mask, 1), 
        "trees": (tree_mask, 100), 
        "water": (water_mask, 1000), 
        "buildings": (building_mask, 100000)  # Buildings must be avoided at all costs
        }, debug=True)
