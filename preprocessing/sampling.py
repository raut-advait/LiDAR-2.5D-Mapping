"""Deterministic spatial sampling for LiDAR model input."""

import numpy as np


def voxel_sample_points(
    points: np.ndarray,
    num_points: int = 32768,
    voxel_size: float = 0.25,
) -> np.ndarray:
    """Select a deterministic, spatially distributed fixed-size point sample."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, features) with at least XYZ.")
    if len(points) == 0:
        raise ValueError("Cannot sample an empty point cloud.")
    if num_points <= 0:
        raise ValueError("num_points must be greater than zero.")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero.")

    xyz = points[:, :3]
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)
    _, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
    representative_indices = np.sort(first_indices)

    if len(representative_indices) > num_points:
        selection_positions = np.linspace(
            0,
            len(representative_indices) - 1,
            num_points,
            dtype=np.int64,
        )
        representative_indices = representative_indices[selection_positions]
    elif len(representative_indices) < num_points:
        padding_positions = np.arange(
            num_points - len(representative_indices),
            dtype=np.int64,
        ) % len(representative_indices)
        representative_indices = np.concatenate(
            [representative_indices, representative_indices[padding_positions]]
        )

    return points[representative_indices]