import numpy as np
import json
import cv2
import networkx as nx
from queue import PriorityQueue
from time import gmtime, strftime
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation

def __plot_overlap__(rect1, rect2, min_distance):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return not (
        x1 + w1 + min_distance <= x2 or
        x2 + w2 + min_distance <= x1 or
        y1 + h1 + min_distance <= y2 or
        y2 + h2 + min_distance <= y1
    )

def get_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mask_range(mask: np.ndarray, contour_min_size: int = 1000, range_size: int = 200) -> np.ndarray:
    near_mask = np.zeros_like(mask) # new empty mask
    contours = get_contours(mask)

    # Add radius around contours to mask
    for cnt in contours:
        if cv2.contourArea(cnt) >= contour_min_size:
            cv2.drawContours(near_mask, [cnt], -1, 1, thickness=range_size)  # Debug: Changed to 1 instead of 255

    return near_mask

def get_mask_regions(mask: np.ndarray, min_size: int = 0) -> list:
    # Apply connected components to label each separate area
    num_labels, labels = cv2.connectedComponents(mask)

    # Create a list to store individual masks
    mask_regions = []

    # Generate each mask for the labeled regions (ignore label 0, which is the background)
    for label in range(1, num_labels):
        # Create a new mask for each component
        component_mask = (labels == label).astype(np.uint8)
        
        # Check if the region's size is greater than the minimum size
        if np.sum(component_mask) >= min_size:
            mask_regions.append(component_mask)
    
    return mask_regions

def get_mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    # Calculate moments of the mask
    moments = cv2.moments(mask)

    # Calculate the centroid from the moments (cx, cy)
    if moments["m00"] != 0:  # To avoid division by zero
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # If the area is zero, set centroid to None
        return None
    
    return cx, cy

def get_buildings(sort_priority: bool = True) -> list:
    with open("buildings.json", "r") as f:
        buildings = json.load(f)
    
    if sort_priority:
        # Sorting by priority (descending) and then by size (descending)
        return sorted(
            buildings, 
            key=lambda x: (x["priority"], x["size"][0] * x["size"][1]), 
            reverse=True
            )
    else:
        return buildings

def place_buildings(blueprints: list, masks: dict[str, np.ndarray], scaling_factor: float = 1) -> tuple[list, np.ndarray]:
    # Place buildings
    placed_buildings = []
    building_mask = np.zeros_like(masks["zero"])
    for blueprint in blueprints:
        for _ in range(blueprint["amount"]):
            nametag = blueprint["name"]
            dimensions = blueprint["size"]
            location = blueprint["location"]

            # Get possible location mask
            mask = masks[location]
            centroid = get_mask_centroid(mask)

            # Iterate over the mask
            for y, x in sorted(np.argwhere(mask > 0), key=lambda x: ((x[0] - centroid[0]) ** 2 + (x[1] - centroid[1]) ** 2) ** 0.5):  # Sort by distance to centroid
                rect_width, rect_height = int(dimensions[0] * scaling_factor), int(dimensions[1] * scaling_factor)

                #Check if rectangles fit within the mask-area
                if (x + rect_width <= mask.shape[1]) and (y + rect_height <= mask.shape[0]):
                    if np.all(mask[y:y + rect_height, x:x + rect_width] > 0):
                        new_rect = (x, y, rect_width, rect_height)

                        # Check building collision
                        min_distance = 10
                        if all(not __plot_overlap__(new_rect, placed_building["rect"], min_distance) for placed_building in placed_buildings):
                            placed_buildings.append({"nametag": nametag, "rect": new_rect})
                            building_mask[y:y + rect_height, x:x + rect_width] = 1
                            break
    
    return placed_buildings, building_mask

def overlay_from_masks(img, *masks: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = img_rgb.copy()

    for mask, color, alpha in masks:
        mask_color = np.zeros_like(img_rgb, dtype=np.float32)
        mask_color[mask > 0] = color

        alpha_mask = (mask > 0).astype(np.float32) * alpha
        overlay = overlay * (1 - alpha_mask[..., None]) + mask_color * alpha_mask[..., None]

    return np.clip(overlay, 0, 255).astype(np.uint8)

def filter_artifacts(mask: np.ndarray, min_area_threshold: int = 2500) -> np.ndarray:
    contours = get_contours(mask)
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_threshold:
            cv2.drawContours(filtered_mask, [cnt], -1, 1, thickness=cv2.FILLED)  # Debug: Changed to 1 instead of 255

    return filtered_mask

# Custom Delaunay triangulation function
def custom_delaunay(points) -> set[tuple[int, int]]:
    edges = set()
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                pi, pj, pk = points[i], points[j], points[k]
                circumcenter, radius = get_circumcircle(pi, pj, pk)
                if circumcenter is None:
                    continue
                is_valid = True
                for m in range(num_points):
                    if m != i and m != j and m != k:
                        if np.linalg.norm(points[m] - circumcenter) < radius:
                            is_valid = False
                            break
                if is_valid:
                    edges.add((i, j))
                    edges.add((j, k))
                    edges.add((k, i))
    return edges

def get_circumcircle(p1, p2, p3) ->  tuple[np.ndarray, float]:
    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1]
    ])
    D = np.linalg.det(A)
    if D == 0:
        return None, np.inf
    A1 = np.array([
        [p1[0]**2 + p1[1]**2, p1[1], 1],
        [p2[0]**2 + p2[1]**2, p2[1], 1],
        [p3[0]**2 + p3[1]**2, p3[1], 1]
    ])
    A2 = np.array([
        [p1[0]**2 + p1[1]**2, p1[0], 1],
        [p2[0]**2 + p2[1]**2, p2[0], 1],
        [p3[0]**2 + p3[1]**2, p3[0], 1]
    ])
    A3 = np.array([
        [p1[0]**2 + p1[1]**2, p1[0], p1[1]],
        [p2[0]**2 + p2[1]**2, p2[0], p2[1]],
        [p3[0]**2 + p3[1]**2, p3[0], p3[1]]
    ])
    x = np.linalg.det(A1) / (2 * D)
    y = -np.linalg.det(A2) / (2 * D)
    radius = np.sqrt(np.linalg.det(A3) / D + x**2 + y**2)
    return np.array([x, y]), radius

def building_centers(buildings: list) -> np.ndarray:
    return np.array([(building["rect"][0] + building["rect"][2] // 2, building["rect"][1] + building["rect"][3] // 2) for building in buildings])

def generate_path_tree(buildings: list, max_length: int|None = None) -> list[tuple[tuple]]:
    centers = building_centers(buildings)

    # Generate paths between all centers using Delauney
    edges = custom_delaunay(centers)

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

    return paths

def astar(start: tuple, goal: tuple, masks: dict[str, np.ndarray], cost_multipliers: dict[str, float]) -> list[tuple]|None:
    start, goal = tuple(start), tuple(goal)
    rows, cols = masks["zero"].shape

    # Precompute cost map
    cost_map = np.zeros((rows, cols))
    for name, mask in masks.items():
        cost_map += mask * cost_multipliers[name]

    # Heuristic function: Octile distance
    def heuristic(a, b):
        D, D2 = 1, 1.414  # D for cardinal moves, D2 for diagonal moves
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    # Directions: including diagonals
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # cardinal directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal directions
    ]

    # Initialize PriorityQueue
    queue = PriorityQueue()
    queue.put((0, start))  # (cost, position)
    costs = {start: 0}  # Cost from start to each position
    came_from = {start: None}  # For reconstructing path

    while not queue.empty():
        current_cost, current = queue.get()

        # Check if reached the goal
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path from start to goal

        # Explore neighbors
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])

            # Check if the neighbor is within bounds
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                # Cost to move to neighbor (1 for cardinal, ~1.414 for diagonal)
                move_cost = 1 if d in [(1, 0), (-1, 0), (0, 1), (0, -1)] else 1.414
                # Apply mask multipliers
                move_cost *= cost_map[neighbor[1]][neighbor[0]]
                new_cost = costs[current] + move_cost

                # If the neighbor hasn't been visited or we found a cheaper path
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    queue.put((priority, neighbor))
                    came_from[neighbor] = current

    return None  # No path found if loop completes without returning

def get_connectors_from_centers(p1: np.ndarray, p2: np.ndarray, building_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dir_vector = (p1 - p2) / np.linalg.norm(p2 - p1)

    dir_vector = np.round(dir_vector).astype(int)

    # Move points out of building
    while building_mask[p1[1]][p1[0]] > 0:
        p1 -= dir_vector
    while building_mask[p2[1]][p2[0]] > 0:
        p2 += dir_vector

    return p1, p2

def generate_path_points(buildings: list, masks_and_cost_multipliers: dict[str, tuple[np.ndarray, float]], resolution_factor: float = 1, max_distance: int|None = None) -> list[list[tuple]]:
    path_tree = generate_path_tree(buildings, max_length=max_distance)

    masks = {name: scale_mask(masks_and_cost_multipliers[name][0], resolution_factor) for name in masks_and_cost_multipliers}
    cost_multipliers = {name: masks_and_cost_multipliers[name][1] for name in masks_and_cost_multipliers}

    path_points = []
    bridge_points = []
    for p1, p2 in path_tree:
        p1, p2 = get_connectors_from_centers(p1, p2, masks_and_cost_multipliers['buildings'][0])
        p1 = (int(p1[0] * resolution_factor), int(p1[1] * resolution_factor))
        p2 = (int(p2[0] * resolution_factor), int(p2[1] * resolution_factor))
        points = astar(p1, p2, masks=masks, cost_multipliers=cost_multipliers)
        path_points.append([(x // resolution_factor, y // resolution_factor) for x, y in points])
        bridge_points.extend([(x // resolution_factor, y // resolution_factor) for x, y in points if masks['water'][y][x] > 0])
    
    return path_points, bridge_points

def scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    new_mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

    return new_mask

def get_mask_exit_point(mask: np.ndarray, direction_vector: np.ndarray, step_size: float = 1.0) -> np.ndarray:
    current_point = np.array(get_mask_centroid(mask))
    direction = np.array(direction_vector) / np.linalg.norm(direction_vector)  # Normalize direction
    
    while current_point.astype(int) in np.argwhere(mask > 0):
        current_point += step_size * direction

    return current_point  # This point is outside the mask
  
def get_mask_boundry(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask) & ~mask

def is_mask_enclosed(mask: np.ndarray, enclosing_mask: np.ndarray) -> bool:
    mask_boundry = get_mask_boundry(mask)

    return np.all(enclosing_mask[mask_boundry > 0] > 0)

def switch_enclaves(*masks: np.ndarray, enclosed_by_one: bool = True, enclave_size_threshold: int = 2500) -> None:
    for mask in masks:
        other_masks = [m for m in masks if m is not mask]
        regions = get_mask_regions(mask)
        for region in regions:
            # Check if region is an enclave (size and enclosement check)
            if np.sum(region) < enclave_size_threshold:
                # If enclosed_by_one, check if region is enclosed by only one mask
                if enclosed_by_one:
                    for other_mask in other_masks:
                        if is_mask_enclosed(region, other_mask):
                            # Switch region to other mask
                            mask[region > 0] = 0
                            other_mask[region > 0] = 1  # Debug: Changed to 1 instead of 255
                            break
                # Else give region to other mask with longest border to region
                else:
                    mask[region > 0] = 0
                    mask_with_longest_border = max(other_masks, key=lambda m: border_length(region, m))
                    mask_with_longest_border[region > 0] = 1  # Debug: Changed to 1 instead of 255

def border_length(mask1: np.ndarray, mask2: np.ndarray) -> int:
    mask1_boundry = get_mask_boundry(mask1)

    return np.sum(mask1_boundry & mask2)

def get_nearst_point_in_mask(mask: np.ndarray, point: tuple) -> tuple[tuple, float]:
    # Convert mask to a list of coordinates
    mask_points = np.array([(y, x) for y in range(mask.shape[0]) for x in range(mask.shape[1]) if mask[y, x] > 0])

    # Build KD-Tree
    tree = KDTree(mask_points)

    # Find closest point
    distance, index = tree.query(point)
    closest_point = mask_points[index]

    return closest_point, distance

def get_mask_edge_points(mask: np.ndarray) -> list[tuple]:
    # Use Canny edge detection to find the edges
    edges = cv2.Canny(mask.astype(np.uint8) * 1, 100, 200)  # Debug: Changed to 1 instead of 255

    # Get the coordinates of the edge pixels
    edge_points = np.column_stack(np.where(edges > 0))

    return edge_points

def get_values_in_radius(mask: np.ndarray, coords: tuple[int, int], radius: int) -> list:
    """Returns a list of values from a mask based on given radius"""
    h, w = mask.shape
    values = []

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if np.sqrt(i**2 + j**2) <= radius:
                nx, ny = coords[0] + i, coords[1] + j
                if 0 <= nx < w and 0 <= ny < h:
                    values.append(mask[ny, nx])

    return values

def set_radius(mask: np.ndarray, coords: tuple[int, int], radius: int, value: int) -> np.ndarray:
    """Returns mask with drawn radius (set to 1 or 0)"""
    new_mask = mask.copy()
    h, w = new_mask.shape

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            # Calculate center
            if np.sqrt(i**2 + j**2) <= radius:
                nx, ny = coords[0] + i, coords[1] + j
                # Check if coordinates in mask-barrier
                if 0 <= nx < w and 0 <= ny < h:
                    new_mask[ny, nx] = value  # set value

    return new_mask.astype(np.uint8)

def refactor_rescale(mask: np.ndarray, scaling_factor: float) -> np.ndarray:
    """Scales mask with factor and rescales it back to original shape"""
    scaled_mask = scale_mask(mask=mask, scale=scaling_factor)
    return cv2.resize(scaled_mask, tuple(reversed(mask.shape)), interpolation=cv2.INTER_NEAREST)

def subtract_masks(mask_1: np.ndarray, mask_2: np.ndarray) -> np.ndarray:
    result_mask = np.copy(mask_1)
    result_mask[mask_2 == 1] = 0
    return result_mask

def paste_debugging(message: str) -> None:
    print(f"{strftime('%H:%M:%S', gmtime())} - {message}")
