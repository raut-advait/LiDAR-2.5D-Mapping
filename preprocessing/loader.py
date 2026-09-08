import os
import glob
import numpy as np
from pathlib import Path


def verify_coordinate_geometry(points: np.ndarray) -> bool:
    """Validate point cloud coordinate array format."""
    if not isinstance(points, np.ndarray):
        return False
    if points.ndim != 2 or points.shape[1] < 3:
        return False
    return True


class FrameLoader:
    """
    Frame sequence loader for recorded prediction NPZ files.
    Reads prediction files from lidar_inference_pipeline/data/processed/predictions.
    Preserves ego-local sensor coordinates (+X = forward, +Y = left, +Z = up).
    """

    def __init__(
        self,
        data_dir: str = "lidar_inference_pipeline/data/processed/predictions",
        max_cache_size: int = 50,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            repo_root = Path(__file__).resolve().parent.parent
            alt = repo_root / data_dir
            if alt.is_dir():
                self.data_dir = alt

        self.files = sorted(list(self.data_dir.glob("*.npz"))) if self.data_dir.is_dir() else []
        self.max_cache_size = max_cache_size
        self.cache = {}

    def get_total_frames(self) -> int:
        return len(self.files)

    def load_frame(self, index: int) -> dict:
        if not self.files:
            raise FileNotFoundError(f"No prediction NPZ files found in directory: {self.data_dir}")
        idx = index % len(self.files)
        if idx in self.cache:
            return self.cache[idx]

        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        points = data["points"]
        labels = data["labels"]

        if "source_file" in data:
            src = data["source_file"]
            source_file = str(src.item() if hasattr(src, "item") else src)
        else:
            source_file = file_path.name

        import math
        ego_heading = math.pi / 2
        if "ego_heading_rad" in data:
            ego_heading = float(data["ego_heading_rad"])
        elif "heading" in data:
            ego_heading = float(data["heading"])

        frame_data = {
            "points": points,
            "labels": labels,
            "source_file": source_file,
            "file_name": file_path.name,
            "ego_heading_rad": ego_heading,
        }

        if len(self.cache) < self.max_cache_size:
            self.cache[idx] = frame_data

        return frame_data
