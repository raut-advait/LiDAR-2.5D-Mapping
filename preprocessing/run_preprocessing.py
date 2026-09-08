"""Batch preprocessing for raw LiDAR frames."""

from pathlib import Path

import numpy as np

from preprocessing.loader import FrameLoader
from preprocessing.preprocess import normalize_xyz
from preprocessing.sampling import voxel_sample_points


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_2" / "cleaned_frames"


def preprocess_all_frames(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
    clip_radius: float = 60.0,
    num_points: int = 32768,
    voxel_size: float = 0.25,
) -> int:
    """Save cleaned real-world points and normalized model points for every frame."""
    raw_files = sorted(raw_dir.glob("*.pcd.bin"))

    if not raw_files:
        raise FileNotFoundError(f"No raw LiDAR files found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    loader = FrameLoader(clip_radius=clip_radius)
    processed_count = 0

    for raw_file in raw_files:
        points = loader.load_frame(str(raw_file))
        sampled_points = voxel_sample_points(
            points,
            num_points=num_points,
            voxel_size=voxel_size,
        )
        model_points = normalize_xyz(sampled_points)
        output_file = output_dir / f"{raw_file.stem}_cleaned.npz"

        np.savez_compressed(
            output_file,
            points=points.astype(np.float32, copy=False),
            sampled_points=sampled_points.astype(np.float32, copy=False),
            model_points=model_points.astype(np.float32, copy=False),
            source_file=np.array(raw_file.name),
        )

        processed_count += 1
        print(
            f"[{processed_count}/{len(raw_files)}] "
            f"{raw_file.name}: "
            f"raw={points.shape}, sampled={sampled_points.shape}, "
            f"model={model_points.shape} "
            f"-> {output_file.name}"
        )

    return processed_count


if __name__ == "__main__":
    count = preprocess_all_frames()
    print(f"Completed preprocessing for {count} LiDAR frames.")