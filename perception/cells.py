"""
Perception cell helpers module (Track D / Integration).

Provides utility functions for cell aggregation, radius filtering, baseline cell counting,
and dynamic local raw-point refinement around hazard zones.
"""

import numpy as np


def clip_points_by_radius(points: np.ndarray, labels: np.ndarray, max_radius: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Clips raw point cloud and labels by Euclidean radius from origin (0, 0) in 2D ego plane.
    """
    if len(points) == 0:
        return points, labels

    x = points[:, 0]
    y = points[:, 1]
    dist = np.hypot(x, y)
    mask = dist <= max_radius
    return points[mask], labels[mask]


def get_cell_center(cell: dict) -> tuple[float, float]:
    """Calculates physical (x, y) center coordinate of a cell in ego frame."""
    res = cell["resolution"]
    center_x = (cell["x_index"] + 0.5) * res
    center_y = (cell["y_index"] + 0.5) * res
    return float(center_x), float(center_y)


def get_cell_bounds(cell: dict) -> tuple[float, float, float, float]:
    """Calculates physical bounding box (x_min, x_max, y_min, y_max) of a cell in ego frame."""
    res = cell["resolution"]
    x_min = cell["x_index"] * res
    x_max = (cell["x_index"] + 1) * res
    y_min = cell["y_index"] * res
    y_max = (cell["y_index"] + 1) * res
    return float(x_min), float(x_max), float(y_min), float(y_max)


def get_cell_dominant_label(cell: dict) -> int:
    """Extracts dominant semantic label from cell layers."""
    layers = cell.get("layers", [])
    if not layers:
        return 0

    best_layer = max(layers, key=lambda l: l.get("point_count", 0))
    return int(best_layer.get("label", 0))


def filter_cells_by_radius(cells: list, max_radius: float = 60.0) -> list:
    """
    Filters cell representation by center radius from ego vehicle origin (0, 0).
    Cell centers must be explicitly checked against max_radius.
    """
    filtered_cells = []
    for cell in cells:
        cx, cy = get_cell_center(cell)
        if np.hypot(cx, cy) <= max_radius:
            filtered_cells.append(cell)
    return filtered_cells


def compute_fixed_fine_baseline_cells(points: np.ndarray, fine_res: float = 0.25, max_radius: float = 60.0) -> int:
    """
    Computes the baseline cell count if the entire frame (clipped to max_radius)
    were represented using fixed fine-resolution grid cells.
    Measured directly from raw source points.
    """
    pts_clipped, _ = clip_points_by_radius(points, np.zeros(len(points), dtype=int), max_radius=max_radius)
    if len(pts_clipped) == 0:
        return 0

    x = pts_clipped[:, 0]
    y = pts_clipped[:, 1]

    grid_x = np.floor(x / fine_res).astype(np.int32)
    grid_y = np.floor(y / fine_res).astype(np.int32)

    grid_coords = grid_x + 1j * grid_y
    unique_coords = np.unique(grid_coords)

    # Filter cell centers by max_radius
    valid_count = 0
    for coord in unique_coords:
        gx = np.real(coord)
        gy = np.imag(coord)
        cx = (gx + 0.5) * fine_res
        cy = (gy + 0.5) * fine_res
        if np.hypot(cx, cy) <= max_radius:
            valid_count += 1

    return int(valid_count)


def refine_local_region(
    points: np.ndarray,
    labels: np.ndarray,
    center: tuple,
    radius: float = 5.0,
    res: float = 0.25,
) -> list:
    """
    Triggers local resolution refinement around hazard coordinates from RAW SOURCE POINTS.
    Re-bins raw points inside the target radius at the fine tier resolution (default 0.25m).
    """
    if len(points) == 0:
        return []

    cx_ref, cy_ref = center
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Distance to hazard center in 2D
    dist_to_center = np.hypot(x - cx_ref, y - cy_ref)
    mask = dist_to_center <= radius
    if not np.any(mask):
        return []

    ref_x = x[mask]
    ref_y = y[mask]
    ref_z = z[mask]
    ref_labels = labels[mask]

    grid_x = np.floor(ref_x / res).astype(np.int32)
    grid_y = np.floor(ref_y / res).astype(np.int32)

    grid_coords = grid_x + 1j * grid_y
    sort_idx = np.argsort(grid_coords)

    grid_coords = grid_coords[sort_idx]
    sorted_z = ref_z[sort_idx]
    sorted_labels = ref_labels[sort_idx]

    unique_coords, start_indices, counts = np.unique(grid_coords, return_index=True, return_counts=True)
    z_chunks = np.split(sorted_z, start_indices[1:])
    labels_chunks = np.split(sorted_labels, start_indices[1:])

    refined_cells = []
    for i in range(len(unique_coords)):
        cell_z = z_chunks[i]
        cell_labels = labels_chunks[i]
        coord = unique_coords[i]
        gx = int(np.real(coord))
        gy = int(np.imag(coord))
        point_count = int(counts[i])

        vals, freqs = np.unique(cell_labels, return_counts=True)
        dom_label = int(vals[np.argmax(freqs)])

        z_min = float(cell_z.min())
        z_max = float(cell_z.max())

        refined_cells.append({
            "zone": "refined",
            "resolution": float(res),
            "x_index": gx,
            "y_index": gy,
            "point_count": point_count,
            "height_min": z_min,
            "height_max": z_max,
            "height_variation": float(z_max - z_min),
            "layers": [
                {
                    "z_min": z_min,
                    "z_max": z_max,
                    "representative_height": z_max,
                    "label": dom_label,
                    "point_count": point_count,
                }
            ],
        })

    return refined_cells


def merge_cells_with_refinement(
    baseline_cells: list,
    refined_cells: list,
    hazard_center: tuple,
    radius: float = 5.0,
    refinement_active: bool = False,
) -> tuple[list, int, int, int]:
    """
    Merges baseline converter cells with dynamic raw-point local refinement cells.

    Returns:
    - merged_cells (list)
    - cells_before (int): baseline cell count inside local hazard radius
    - cells_after (int): fine cell count inside local hazard radius after re-binning
    - total_cells (int): len(merged_cells)
    """
    hx, hy = hazard_center

    # Compute baseline cells inside hazard radius
    inside_baseline = []
    outside_baseline = []

    for cell in baseline_cells:
        cx, cy = get_cell_center(cell)
        if np.hypot(cx - hx, cy - hy) <= radius:
            inside_baseline.append(cell)
        else:
            outside_baseline.append(cell)

    cells_before = len(inside_baseline)

    if refinement_active and len(refined_cells) > 0:
        cells_after = len(refined_cells)
        merged_cells = outside_baseline + refined_cells
    else:
        cells_after = cells_before
        merged_cells = baseline_cells

    total_cells = len(merged_cells)
    return merged_cells, cells_before, cells_after, total_cells
