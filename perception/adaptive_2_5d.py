import numpy as np


class Adaptive2_5DConverter:
    """
    Distance-aware and detail-aware adaptive 2.5D converter.
    Optimized for extremely fast CPU vectorization.
    """

    def __init__(
        self,
        near_distance=15.0,
        medium_distance=35.0,
        far_distance=60.0,
        near_resolution=0.25,
        medium_resolution=0.50,
        far_resolution=1.00,
        near_max_layers=5,
        medium_max_layers=3,
        far_max_layers=2,
        height_variation_threshold=0.5,
    ):
        self.near_distance = near_distance
        self.medium_distance = medium_distance
        self.far_distance = far_distance
        self.near_resolution = near_resolution
        self.medium_resolution = medium_resolution
        self.far_resolution = far_resolution
        self.near_max_layers = near_max_layers
        self.medium_max_layers = medium_max_layers
        self.far_max_layers = far_max_layers
        self.height_variation_threshold = height_variation_threshold

    def convert(self, points, labels):
        if hasattr(points, "cpu"):
            points = points.cpu().numpy()
        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()

        if len(points) == 0:
            raise ValueError("Point cloud is empty.")

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        distances = np.sqrt(x**2 + y**2)
        final_cells = []

        zones = [
            ("near", 0.0, self.near_distance, self.near_resolution, self.near_max_layers),
            ("medium", self.near_distance, self.medium_distance, self.medium_resolution, self.medium_max_layers),
            ("far", self.medium_distance, self.far_distance, self.far_resolution, self.far_max_layers),
        ]

        for zone_name, d_min, d_max, res, max_layers in zones:
            mask = (distances > d_min) & (distances <= d_max)
            if not np.any(mask):
                continue

            zone_x = x[mask]
            zone_y = y[mask]
            zone_z = z[mask]
            zone_labels = labels[mask]

            grid_x = np.floor(zone_x / res).astype(np.int32)
            grid_y = np.floor(zone_y / res).astype(np.int32)

            # Combine to unique identifiers using complex numbers
            grid_coords = grid_x + 1j * grid_y

            # Sort to group efficiently
            sort_idx = np.argsort(grid_coords)
            grid_coords = grid_coords[sort_idx]
            sorted_z = zone_z[sort_idx]
            sorted_labels = zone_labels[sort_idx]

            unique_coords, start_indices, counts = np.unique(grid_coords, return_index=True, return_counts=True)

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

                # Calculate layers
                layers = 1
                if height_variation >= self.height_variation_threshold:
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

                final_cells.append({
                    "zone": zone_name,
                    "resolution": float(res),
                    "x_index": gx,
                    "y_index": gy,
                    "point_count": point_count,
                    "height_min": z_min,
                    "height_max": z_max,
                    "height_variation": float(height_variation),
                    "layers": cell_layers,
                })

        return {
            "cells": final_cells,
            "near_distance": self.near_distance,
            "medium_distance": self.medium_distance,
            "far_distance": self.far_distance,
            "total_cells": len(final_cells),
            "input_points": len(points),
        }
