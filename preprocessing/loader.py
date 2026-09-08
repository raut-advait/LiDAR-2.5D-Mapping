"""
Preprocessing module (Track B).

Handles raw frame loading, coordinate convention enforcement (+X forward, +Y left, +Z height),
ego-centered clipping (60m radius), and self-return removal.
"""

import numpy as np


class FrameLoader:
    """Loads and preprocesses raw LiDAR point cloud frames."""

    def __init__(self, clip_radius: float = 60.0):
        self.clip_radius = clip_radius

    def load_frame(self, frame_path: str) -> np.ndarray:
        """
        Loads a LiDAR frame and returns clipped points meeting coordinate conventions:
        +X = forward, +Y = left, +Z = height, Ego at (0, 0, 0).
        """
        raise NotImplementedError("Track B implementation pending.")
