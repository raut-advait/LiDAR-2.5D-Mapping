"""
Preprocessing module (Track B).

Handles raw frame loading, coordinate convention enforcement (+X forward, +Y left, +Z height),
ego-centered clipping (60m radius), and self-return removal.
"""

import numpy as np


class FrameLoader:
    """Loads and preprocesses raw LiDAR point cloud frames."""

    def __init__(
        self,
        clip_radius: float = 60.0,
        ego_bounds: tuple[float, float, float, float, float, float] = (
            -2.5,
            2.5,
            -1.5,
            1.5,
            -2.5,
            1.0,
        ),
    ):
        self.clip_radius = clip_radius
        self.ego_bounds = ego_bounds

    def load_frame(self, frame_path: str) -> np.ndarray:
        """
        Loads a LiDAR frame and returns clipped points meeting coordinate conventions:
        +X = forward, +Y = left, +Z = height, Ego at (0, 0, 0).
        """
        if self.clip_radius <= 0:
            raise ValueError("clip_radius must be greater than zero.")

        raw_data = np.fromfile(frame_path, dtype=np.float32)

        if raw_data.size == 0:
            raise ValueError(f"LiDAR frame is empty: {frame_path}")

        if raw_data.size % 5 != 0:
            raise ValueError(
                "Invalid LiDAR frame format: expected five float32 values per point."
            )

        points = raw_data.reshape(-1, 5)
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) == 0:
            raise ValueError(f"LiDAR frame has no finite points: {frame_path}")

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        within_radius = (x * x + y * y) <= self.clip_radius * self.clip_radius

        min_x, max_x, min_y, max_y, min_z, max_z = self.ego_bounds
        ego_returns = (
            (x >= min_x)
            & (x <= max_x)
            & (y >= min_y)
            & (y <= max_y)
            & (z >= min_z)
            & (z <= max_z)
        )

        points = points[within_radius & ~ego_returns]

        if len(points) == 0:
            raise ValueError(
                f"LiDAR frame has no points after clipping and ego filtering: {frame_path}"
            )

        return points
