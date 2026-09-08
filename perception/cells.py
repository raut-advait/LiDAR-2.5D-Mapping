import numpy as np


def get_cell_bounds(cell: dict) -> tuple:
    """Return cell bounding box (x_min, x_max, y_min, y_max)."""
    res = float(cell["resolution"])
    gx = int(cell["x_index"])
    gy = int(cell["y_index"])
    x_min = gx * res
    x_max = x_min + res
    y_min = gy * res
    y_max = y_min + res
    return (x_min, x_max, y_min, y_max)


def get_cell_dominant_label(cell: dict) -> int:
    """Extract dominant semantic label from cell layers."""
    layers = cell.get("layers", [])
    if not layers:
        return 0
    if len(layers) == 1:
        return int(layers[0].get("label", 0))
    counts = {}
    for layer in layers:
        lbl = int(layer.get("label", 0))
        counts[lbl] = counts.get(lbl, 0) + int(layer.get("point_count", 1))
    return max(counts, key=counts.get)


def filter_cells_by_radius(cells: list, center_x: float, center_y: float, radius: float) -> list:
    """Filter list of cell dicts to those whose centers lie within radius of (center_x, center_y)."""
    filtered = []
    for c in cells:
        res = c["resolution"]
        cx = (c["x_index"] + 0.5) * res
        cy = (c["y_index"] + 0.5) * res
        if np.hypot(cx - center_x, cy - center_y) <= radius:
            filtered.append(c)
    return filtered
