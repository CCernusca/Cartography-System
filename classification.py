
"""
Terrain region classification

Classifies terrain regions into masks
"""

import helper_functions as hf
import detectree as dtr
import numpy as np
import cv2

# Mask caches for optimization
tree_mask_cache = None
water_mask_cache = None
free_mask_cache = None
coast_mask_cache = None
inland_mask_cache = None
forest_edge_mask_cache = None
water_access_mask_cache = None

def generate_tree_mask(img_path: str, expansion_thickness: int = 2, min_area: int = 10, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate a tree mask from an image, containing all areas classified as trees by the tree classifier.

    Parameters
    ----------
    img_path : str
        Path to the image file
    expansion_thickness : int, optional
        Thickness of the contours to draw around the tree mask, by default 2
    min_area : int, optional
        Minimum area of a contour to be considered a tree, by default 10
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Tree mask as a numpy array
    """
    global tree_mask_cache
    if use_cache and tree_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Tree mask generation) Using cached tree mask, this will use potentially outdated data!")

        return tree_mask_cache

    if debug:
        hf.paste_debugging("(Tree mask generation) Start dataset load")

    y_pred = dtr.Classifier().predict_img(img_path)
    tree_mask = y_pred.astype(np.uint8)
    
    if debug:
        hf.paste_debugging("(Tree mask generation) Classification done")

    contours = hf.get_contours(tree_mask)

    # Draw Contours around vegetation areas based on "expansion-thickness"
    expanded_mask = np.zeros_like(tree_mask)  # new mask layer
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.fillPoly(expanded_mask, [cnt], 1)

            if expansion_thickness > 0:
                cv2.drawContours(expanded_mask, [cnt], -1, 1, thickness=expansion_thickness)
    
    # Cache tree mask
    tree_mask_cache = expanded_mask

    if debug:
        hf.paste_debugging("(Tree mask generation) Tree mask generated")

    return expanded_mask

def generate_water_mask(img: cv2.typing.MatLike, lower_water_color: np.ndarray[int] = np.array([90, 50, 50]), upper_water_color: np.ndarray[int] = np.array([140, 255, 255]), min_area_threshold: int = 500, water_kernel_size: int = 12, radius: float = 3, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate a water mask from an image, containing all areas classified as water by color detection and gabor filter.

    Parameters
    ----------
    img : cv2.typing.MatLike
        Input image
    lower_water_color : np.ndarray[int, int, int], optional
        Lower bound of the water color range in HSV, by default np.array([90, 50, 50])
    upper_water_color : np.ndarray[int, int, int], optional
        Upper bound of the water color range in HSV, by default np.array([140, 255, 255])
    min_area_threshold : int, optional
        Minimum area of a contour to be considered water, by default 500
    water_kernel_size : int, optional
        Size of the morphologyEx kernel for closing small gaps in the water layer, by default 12
    radius : float, optional
        Radius of the expansion of the water mask, by default 3
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Water mask as a numpy array
    """
    global water_mask_cache
    if use_cache and water_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Water mask generation) Using cached water mask, this will use potentially outdated data!")

        return water_mask_cache

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_water = cv2.inRange(hsv, lower_water_color, upper_water_color)

    # Morphological operations (close small gaps in layer)
    water_kernel = np.ones((water_kernel_size, water_kernel_size), np.uint8)
    closed_water_mask = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, water_kernel)
    filtered_water_mask = hf.filter_artifacts(closed_water_mask, min_area_threshold=min_area_threshold)

    # Scale mask down for performance
    scale_factor: float = 0.35
    water_mask =  hf.scale_mask((filtered_water_mask > 0).astype(np.uint8), scale_factor)
    
    # Generate gabor filter mask
    gabor_filter_mask = hf.scale_mask(hf.gabor_filter(img), scale_factor)

    # Expands color detected water mask based on gabor filter mask 
    iterations_amount = 8
    for i in range(iterations_amount):
        y_coords, x_coords = np.where(water_mask == 1)
        for y, x in zip(y_coords, x_coords):
            # Get gabor filter mask values from given radius
            radius_values: list = hf.get_values_in_radius(mask=gabor_filter_mask, coords=(x, y), radius=radius)
            # Check if collision with land occurs
            if not sum(radius_values) >= 1:
                # Expand radius if smooth area detected
                water_mask = hf.set_radius(mask=water_mask, coords=(x, y), radius=radius, value=1)

    # Filter artifacts again
    water_mask = hf.filter_artifacts(water_mask, min_area_threshold=min_area_threshold)

    # Resize mask to original size
    water_mask = cv2.resize(water_mask, tuple(reversed(filtered_water_mask.shape)), interpolation=cv2.INTER_NEAREST)

    # Cache water mask
    water_mask_cache = water_mask

    if debug:
        hf.paste_debugging("(Water mask generation) Water mask generated")

    return water_mask

def generate_free_mask(tree_mask: np.ndarray, water_mask: np.ndarray, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate a free mask from tree and water masks, containing all areas that are neither trees nor water.

    Parameters
    ----------
    tree_mask : np.ndarray
        Tree mask as a numpy array
    water_mask : np.ndarray
        Water mask as a numpy array
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Free mask as a numpy array
    """
    if not hf.check_binary(tree_mask):
        raise ValueError("Tree mask must be binary.")
    if not hf.check_binary(water_mask):
        raise ValueError("Water mask must be binary.")
    
    global free_mask_cache
    if use_cache and free_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Free mask generation) Using cached free mask, this will use potentially outdated data!")

        return free_mask_cache

    # Combine tree and water masks to find free areas
    combined_mask = np.logical_or(tree_mask > 0, water_mask > 0).astype(np.uint8)

    # Inverted mask to get free areas
    free_mask = (combined_mask == 0).astype(np.uint8)

    # Cache free mask
    free_mask_cache = free_mask

    if debug:
        hf.paste_debugging("(Free mask generation) Free mask generated")

    return free_mask

def generate_coast_mask(zero_mask: np.ndarray, water_mask: np.ndarray, blob_min_size: int = 1000, coast_range: int = 100, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate a coast mask from a free mask and a water mask, containing free areas in range of water.

    Parameters
    ----------
    zero_mask : np.ndarray
        Free mask as a numpy array
    water_mask : np.ndarray
        Water mask as a numpy array
    blob_min_size : int
        Minimum size of a water source to be considered a water body or surface. Defaults to 1000. A blob is a region of connected pixels.
    coast_range : int
        Range in pixels to consider as coastline. Defaults to 100.
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Coast mask as a numpy array
    """
    if not hf.check_binary(zero_mask):
        raise ValueError("Zero mask must be binary.")
    if not hf.check_binary(water_mask):
        raise ValueError("Water mask must be binary.")
    
    global coast_mask_cache
    if use_cache and coast_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Coast mask generation) Using cached coast mask, this will use potentially outdated data!")

        return coast_mask_cache

    # Find areas in range of large enough water sources (water bodies / water surfaces)
    coast_mask = hf.mask_range(water_mask, blob_min_size=blob_min_size, range_size=coast_range)
    
    # Combine with free areas to get free areas near water
    coast_mask = np.logical_and(zero_mask > 0, coast_mask > 0).astype(np.uint8)

    # Cache coast mask
    coast_mask_cache = coast_mask

    if debug:
        hf.paste_debugging("(Coast mask generation) Coast mask generated")
    
    return coast_mask

def generate_inland_mask(zero_mask: np.ndarray, coast_mask: np.ndarray, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate an inland mask from a free mask and a coast mask, containing free areas not in range of water.

    Parameters
    ----------
    zero_mask : np.ndarray
        Free mask as a numpy array
    coast_mask : np.ndarray
        Coast mask as a numpy array
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Inland mask as a numpy array
    """
    if not hf.check_binary(zero_mask):
        raise ValueError("Zero mask must be binary.")
    if not hf.check_binary(coast_mask):
        raise ValueError("Coast mask must be binary.")
    
    global inland_mask_cache
    if use_cache and inland_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Inland mask generation) Using cached inland mask, this will use potentially outdated data!")

        return inland_mask_cache

    # Take all free areas which are not near water
    inland_mask = cv2.bitwise_and(zero_mask, cv2.bitwise_not(coast_mask))

    # Cache inland mask
    inland_mask_cache = inland_mask

    if debug:
        hf.paste_debugging("(Inland mask generation) Inland mask generated")
    
    return inland_mask

def generate_forest_edge_mask(tree_mask: np.ndarray, zero_mask: np.ndarray, blob_min_size: int = 500, range_size: int = 50, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate a forest edge mask from tree and zero masks, highlighting areas near tree surfaces.

    Parameters
    ----------
    tree_mask : np.ndarray
        Binary mask indicating tree areas.
    zero_mask : np.ndarray
        Binary mask indicating free areas.
    blob_min_size : int, optional
        Minimum size of a tree surface to be considered for edge detection. Defaults to 500.
    range_size : int, optional
        Range in pixels to consider as the forest edge. Defaults to 50.
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Forest edge mask as a numpy array, indicating free areas near tree surfaces.
    """
    if not hf.check_binary(tree_mask):
        raise ValueError("Tree mask must be binary.")
    if not hf.check_binary(zero_mask):
        raise ValueError("Zero mask must be binary.")
    
    global forest_edge_mask_cache
    if use_cache and forest_edge_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Forest edge mask generation) Using cached forest edge mask, this will use potentially outdated data!")

        return forest_edge_mask_cache

    # Find areas in range of large enough tree surfaces
    tree_range_mask = hf.mask_range(tree_mask, blob_min_size=blob_min_size, range_size=range_size)

    # Combine with free areas to get free areas near trees
    forest_edge_mask = np.logical_and(tree_range_mask, zero_mask).astype(np.uint8)

    # Cache forest edge mask
    forest_edge_mask_cache = forest_edge_mask

    if debug:
        hf.paste_debugging("(Forest edge mask generation) Forest edge mask generated")
    
    return forest_edge_mask

def generate_water_access_mask(water_mask: np.ndarray, coast_mask: np.ndarray, debug: bool = False, use_cache: bool = False) -> np.ndarray:
    """
    Generate a water access mask from water and coast masks, highlighting all areas with water access.

    Parameters
    ----------
    water_mask : np.ndarray
        Binary mask indicating water areas.
    coast_mask : np.ndarray
        Binary mask indicating coastal areas.
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached mask if available, by default False

    Returns
    -------
    np.ndarray
        Water access mask as a numpy array, indicating all areas with direct water access.
    """
    if not hf.check_binary(water_mask):
        raise ValueError("Water mask must be binary.")
    if not hf.check_binary(coast_mask):
        raise ValueError("Coast mask must be binary.")
    
    global water_access_mask_cache
    if use_cache and water_access_mask_cache is not None:
        if debug:
            hf.paste_debugging("(Water access mask generation) Using cached water access mask, this will use potentially outdated data!")

        return water_access_mask_cache

    # Combine water and coast masks, for all areas with water access
    water_access_mask = np.logical_or(water_mask == 1, coast_mask == 1).astype(np.uint8)

    # Cache water access mask
    water_access_mask_cache = water_access_mask

    if debug:
        hf.paste_debugging("(Water access mask generation) Water access mask generated")
    
    return water_access_mask

def generate_all_masks(img_path: str, debug: bool = False, use_cache: bool = False) -> tuple[np.ndarray]:
    """
    Generate all masks from a given image path.

    Parameters
    ----------
    img_path : str
        Path to the image file
    debug : bool, optional
        Print debug messages if True, by default False
    use_cache : bool, optional
        Use cached masks if available, by default False

    Returns
    -------
    tuple[np.ndarray]
        Tuple containing all generated masks in the following order:
        tree_mask, water_mask, zero_mask, coast_mask, inland_mask, forest_edge_mask, water_access_mask
    """
    tree_mask = hf.binary_mask(generate_tree_mask(img_path=img_path, debug=debug, use_cache=use_cache))

    img = cv2.imread(img_path)

    water_mask = hf.binary_mask(generate_water_mask(img=img, debug=debug, use_cache=use_cache))

    free_mask = generate_free_mask(tree_mask, water_mask, debug=debug, use_cache=use_cache)

    # Against fully by one mask enclosed zones, specifically artifacts from tree detection
    hf.switch_enclaves(free_mask, tree_mask, water_mask, enclosed_by_one=True, enclave_size_threshold=2500)

    if debug:
        hf.paste_debugging("(All masks generation) Remove enclave artifacts threshold=2500 (True)")

    # Against all artifacts, much smaller threshold as to only get rid of small artifacts and not actually useful areas
    hf.switch_enclaves(free_mask, tree_mask, water_mask, enclosed_by_one=False, enclave_size_threshold=500)

    if debug:
        hf.paste_debugging("(All masks generation) Remove enclave artifacts threshold=500 (False)")

    coast_mask = generate_coast_mask(free_mask, water_mask, debug=debug, use_cache=use_cache)

    inland_mask = generate_inland_mask(free_mask, coast_mask, debug=debug, use_cache=use_cache)

    forest_edge_mask = generate_forest_edge_mask(tree_mask, free_mask, debug=debug, use_cache=use_cache)

    water_access_mask = generate_water_access_mask(water_mask, coast_mask, debug=debug, use_cache=use_cache)

    return (tree_mask, water_mask, free_mask, coast_mask, inland_mask, forest_edge_mask, water_access_mask)

if __name__ == "__main__":
    image_input_path = "./mocking_examples/main2.png"

    result_tuple = generate_all_masks(image_input_path, debug=True)

    generate_tree_mask(image_input_path, debug=True, use_cache=True)
    generate_tree_mask(image_input_path, debug=True, use_cache=False)
