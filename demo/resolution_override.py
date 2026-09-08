import numpy as np


def apply_local_refinement(
    cells: list,
    points: np.ndarray,
    labels: np.ndarray,
    hazard_state: dict,
    risk_metrics: dict,
    converter,
    refinement_radius: float = 5.0,
) -> dict:
    """Apply local resolution refinement within refinement_radius around a hazard vehicle.

    Rules:
        1. Trigger: risk_level == "HIGH" or "CRITICAL" (i.e. final_risk >= 0.60).
        2. Refinement radius: 5.0m around hazard's (x, y).
        3. Raw point re-binning:
           a. Filter raw points (points_clipped, labels_clipped) to those within 5.0m of (hx, hy).
           b. Re-bin those points into 0.25m x 0.25m fine tier cells (reusing converter.near_resolution).
           c. Replace any existing converter cells whose center falls within 5.0m of (hx, hy).
           d. Untouched cells outside 5.0m are preserved.
        4. Source-Point Conservation:
           Confirm 100% of local source points are preserved in fine cells (difference = 0).
    """
    total_cells_before = len(cells)

    if hazard_state is None:
        return {
            "refined_cells": cells,
            "refinement_active": False,
            "refinement_radius": refinement_radius,
            "hazard_x": 0.0,
            "hazard_y": 0.0,
            "pts_in_local_region": 0,
            "pts_represented_by_fine_cells": 0,
            "source_point_difference": 0,
            "min_pt_distance": 0.0,
            "cells_before_local": 0,
            "cells_removed": 0,
            "fine_cells_added": 0,
            "remaining_coarse_cells_inside_refine_radius": 0,
            "total_cells_before": total_cells_before,
            "total_cells_rendered": total_cells_before,
        }

    hx = float(hazard_state["x"])
    hy = float(hazard_state["y"])
    final_risk = risk_metrics.get("final_risk", 0.0)
    risk_level = risk_metrics.get("risk_level", "LOW")

    # 1. Source points distance to hazard
    dist_pts = np.sqrt((points[:, 0] - hx) ** 2 + (points[:, 1] - hy) ** 2)
    mask_local_pts = dist_pts <= refinement_radius
    pts_in_local_region = int(np.sum(mask_local_pts))
    min_pt_distance = float(np.min(dist_pts)) if len(dist_pts) > 0 else 0.0

    pts_local = points[mask_local_pts]
    lbls_local = labels[mask_local_pts]

    # 2. Existing adaptive cells distance to hazard
    local_cells_before = []
    untouched_cells = []

    for cell in cells:
        res = cell["resolution"]
        cx = (cell["x_index"] + 0.5) * res
        cy = (cell["y_index"] + 0.5) * res
        dist_to_hazard = float(np.sqrt((cx - hx) ** 2 + (cy - hy) ** 2))

        if dist_to_hazard <= refinement_radius:
            local_cells_before.append(cell)
        else:
            untouched_cells.append(cell)

    cells_before_local_count = len(local_cells_before)
    refinement_active = bool(final_risk >= 0.60 or risk_level in ["HIGH", "CRITICAL"])

    if not refinement_active:
        return {
            "refined_cells": cells,
            "refinement_active": False,
            "refinement_radius": refinement_radius,
            "hazard_x": hx,
            "hazard_y": hy,
            "pts_in_local_region": pts_in_local_region,
            "pts_represented_by_fine_cells": 0,
            "source_point_difference": 0,
            "min_pt_distance": min_pt_distance,
            "cells_before_local": cells_before_local_count,
            "cells_removed": 0,
            "fine_cells_added": 0,
            "remaining_coarse_cells_inside_refine_radius": sum(
                1 for c in local_cells_before if c["resolution"] > converter.near_resolution
            ),
            "total_cells_before": total_cells_before,
            "total_cells_rendered": total_cells_before,
        }

    # Refinement Active: Re-bin all raw source points within refinement_radius into fine tier cells
    new_fine_cells = []
    if len(pts_local) > 0:
        x = pts_local[:, 0]
        y = pts_local[:, 1]
        z = pts_local[:, 2]

        res = converter.near_resolution
        max_layers = converter.near_max_layers

        grid_x = np.floor(x / res).astype(np.int32)
        grid_y = np.floor(y / res).astype(np.int32)

        grid_coords = grid_x + 1j * grid_y
        sort_idx = np.argsort(grid_coords)
        grid_coords = grid_coords[sort_idx]
        sorted_z = z[sort_idx]
        sorted_labels = lbls_local[sort_idx]

        unique_coords, start_indices, counts = np.unique(
            grid_coords, return_index=True, return_counts=True
        )

        z_chunks = np.split(sorted_z, start_indices[1:])
        labels_chunks = np.split(sorted_labels, start_indices[1:])

        for i in range(len(unique_coords)):
            cell_z = z_chunks[i]
            cell_labels = labels_chunks[i]

            z_min = float(cell_z.min())
            z_max = float(cell_z.max())
            height_variation = z_max - z_min

            coord = unique_coords[i]
            gx = int(np.real(coord))
            gy = int(np.imag(coord))
            point_count = int(counts[i])

            layers = 1
            if height_variation >= converter.height_variation_threshold:
                layers += 1
            num_unique_labels = len(np.unique(cell_labels))
            if (num_unique_labels / point_count) > 0.05:
                layers += 1
            if point_count > 50:
                layers += 1
            layers = min(layers, max_layers)

            cell_layers = []
            if z_min == z_max or layers == 1:
                vals, freqs = np.unique(cell_labels, return_counts=True)
                dom_label = int(vals[np.argmax(freqs)])
                cell_layers.append({
                    "z_min": z_min,
                    "z_max": z_max,
                    "representative_height": z_max,
                    "label": dom_label,
                    "point_count": point_count,
                })
            else:
                layer_edges = np.linspace(z_min, z_max, layers + 1)
                for l_idx in range(layers):
                    lb = layer_edges[l_idx]
                    ub = layer_edges[l_idx + 1]
                    if l_idx == layers - 1:
                        l_mask = (cell_z >= lb) & (cell_z <= ub)
                    else:
                        l_mask = (cell_z >= lb) & (cell_z < ub)

                    if not np.any(l_mask):
                        continue

                    l_z = cell_z[l_mask]
                    l_labels = cell_labels[l_mask]
                    l_z_min = float(l_z.min())
                    l_z_max = float(l_z.max())
                    vals, freqs = np.unique(l_labels, return_counts=True)
                    l_dom_label = int(vals[np.argmax(freqs)])

                    cell_layers.append({
                        "z_min": l_z_min,
                        "z_max": l_z_max,
                        "representative_height": l_z_max,
                        "label": l_dom_label,
                        "point_count": int(np.sum(l_mask)),
                    })

            new_fine_cells.append({
                "zone": "near",
                "resolution": float(res),
                "x_index": gx,
                "y_index": gy,
                "point_count": point_count,
                "height_min": z_min,
                "height_max": z_max,
                "height_variation": float(height_variation),
                "layers": cell_layers,
            })

    cells_removed_count = cells_before_local_count
    fine_cells_added_count = len(new_fine_cells)
    refined_cells = untouched_cells + new_fine_cells
    total_cells_rendered = len(refined_cells)

    # Diagnostic check: source points represented by new fine cells
    pts_represented_by_fine_cells = sum(c["point_count"] for c in new_fine_cells)
    source_point_difference = pts_in_local_region - pts_represented_by_fine_cells

    # Diagnostic check: remaining coarse/medium cells inside refinement radius
    remaining_coarse_cells = 0
    for cell in refined_cells:
        res = cell["resolution"]
        cx = (cell["x_index"] + 0.5) * res
        cy = (cell["y_index"] + 0.5) * res
        dist_to_hazard = float(np.sqrt((cx - hx) ** 2 + (cy - hy) ** 2))
        if dist_to_hazard <= refinement_radius and res > converter.near_resolution:
            remaining_coarse_cells += 1

    return {
        "refined_cells": refined_cells,
        "refinement_active": True,
        "refinement_radius": refinement_radius,
        "hazard_x": hx,
        "hazard_y": hy,
        "pts_in_local_region": pts_in_local_region,
        "pts_represented_by_fine_cells": pts_represented_by_fine_cells,
        "source_point_difference": source_point_difference,
        "min_pt_distance": min_pt_distance,
        "cells_before_local": cells_before_local_count,
        "cells_removed": cells_removed_count,
        "fine_cells_added": fine_cells_added_count,
        "remaining_coarse_cells_inside_refine_radius": remaining_coarse_cells,
        "total_cells_before": total_cells_before,
        "total_cells_rendered": total_cells_rendered,
    }
