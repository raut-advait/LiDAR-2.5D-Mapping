import argparse
from pathlib import Path
import numpy as np


def check_ego_returns(npz_path: str) -> None:
    """Analyze dynamic object points (label == 2) to check for potential ego-vehicle self-returns."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    # Load npz prediction file
    data = np.load(path, allow_pickle=True)
    if "points" not in data or "labels" not in data:
        raise KeyError(f"File {path} must contain 'points' and 'labels' keys.")

    points = data["points"]
    labels = data["labels"]

    # 1. Filter points where labels == 2 (Dynamic Object)
    dynamic_mask = labels == 2
    dynamic_points = points[dynamic_mask]
    total_dynamic = len(dynamic_points)

    print("=" * 60)
    print(f"EGO-VEHICLE SELF-RETURNS ANALYSIS")
    print(f"File: {path.name}")
    print("=" * 60)
    print(f"Total Dynamic Object points (label == 2): {total_dynamic}")

    if total_dynamic == 0:
        print("No dynamic object points found in file.")
        return

    # 2. Compute distance from origin (0, 0)
    x = dynamic_points[:, 0]
    y = dynamic_points[:, 1]
    dist = np.sqrt(x**2 + y**2)

    # 3. Classify into buckets (< 2.5 vs >= 2.5)
    ego_returns_mask = dist < 2.5
    valid_dynamic_mask = dist >= 2.5

    count_ego = np.sum(ego_returns_mask)
    count_valid = np.sum(valid_dynamic_mask)

    # 4. Compute percentages
    pct_ego = (count_ego / total_dynamic) * 100.0
    pct_valid = (count_valid / total_dynamic) * 100.0

    print(f"\nDist < 2.5m  (Likely Ego-Vehicle Self-Returns): {count_ego:6d} ({pct_ego:6.2f}%)")
    print(f"Dist >= 2.5m (Valid Dynamic Objects)          : {count_valid:6d} ({pct_valid:6.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check dynamic object points for ego-vehicle self-returns (dist < 2.5m)."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the prediction .npz file.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to the prediction .npz file (alternative flag).",
    )

    args = parser.parse_args()
    target_path = args.input_path or args.input

    if not target_path:
        parser.error("Please specify a prediction .npz file path.")

    check_ego_returns(target_path)


if __name__ == "__main__":
    main()
