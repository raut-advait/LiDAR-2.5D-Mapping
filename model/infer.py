"""
Model module (Track A).

Runs PointNet++ inference on LiDAR point clouds and exports cached .npz predictions
matching the required interface contract:
  - points: (N, 5) float32 (x, y, z, intensity, ring)
  - model_points: (N, 5) float32
  - labels: (N,) int64 (0=Drivable, 1=Static, 2=Dynamic)
  - source_file: str
"""

import numpy as np


class PointNetInference:
    """PointNet++ inference and export wrapper."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path

    def predict(self, points: np.ndarray) -> dict:
        """Runs model inference on input point cloud."""
        raise NotImplementedError("Track A implementation pending.")

    def export_prediction(self, output_path: str, prediction_data: dict) -> None:
        """Exports prediction dictionary to .npz file."""
        raise NotImplementedError("Track A implementation pending.")
