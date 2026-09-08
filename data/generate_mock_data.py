"""
Generator script for synthetic mock .npz dataset for Balerion Track D.

Matches agreed prediction schema:
points      : (N, 5) float32 (x, y, z, intensity, ring)
model_points: (N, 5) float32
labels      : (N,) int64 (0=Drivable, 1=Static, 2=Dynamic)
source_file : str
"""

import os
import numpy as np


def generate_synthetic_mock_npz(output_path: str = "data/predictions/synthetic_mock.npz"):
    np.random.seed(42)

    # 1. Drivable points (Label 0) - road surface ahead
    n_drivable = 350
    x_drivable = np.random.uniform(0.0, 55.0, n_drivable)
    y_drivable = np.random.uniform(-7.0, 7.0, n_drivable)
    z_drivable = np.random.uniform(-1.8, -1.5, n_drivable)
    labels_drivable = np.zeros(n_drivable, dtype=np.int64)

    # 2. Static points (Label 1) - side structures, barriers, buildings, and far wall (>60m)
    n_static = 350
    x_static_near = np.random.uniform(-10.0, 55.0, 250)
    y_static_left = np.random.uniform(10.0, 25.0, 125)
    y_static_right = np.random.uniform(-25.0, -10.0, 125)
    y_static_near = np.concatenate([y_static_left, y_static_right])
    z_static_near = np.random.uniform(-1.5, 3.0, 250)

    # Far static points beyond 60m radius to test 60m radius filter
    x_static_far = np.random.uniform(62.0, 70.0, 100)
    y_static_far = np.random.uniform(-20.0, 20.0, 100)
    z_static_far = np.random.uniform(-1.5, 2.0, 100)

    x_static = np.concatenate([x_static_near, x_static_far])
    y_static = np.concatenate([y_static_near, y_static_far])
    z_static = np.concatenate([z_static_near, z_static_far])
    labels_static = np.ones(n_static, dtype=np.int64)

    # 3. Dynamic points (Label 2) - moving vehicle at x=22, y=1.5 and pedestrian at x=10, y=-4
    n_vehicle = 80
    x_veh = np.random.uniform(20.0, 24.0, n_vehicle)
    y_veh = np.random.uniform(0.5, 2.5, n_vehicle)
    z_veh = np.random.uniform(-1.2, 0.5, n_vehicle)

    n_ped = 20
    x_ped = np.random.uniform(9.5, 10.5, n_ped)
    y_ped = np.random.uniform(-4.5, -3.5, n_ped)
    z_ped = np.random.uniform(-1.2, 0.6, n_ped)

    x_dynamic = np.concatenate([x_veh, x_ped])
    y_dynamic = np.concatenate([y_veh, y_ped])
    z_dynamic = np.concatenate([z_veh, z_ped])
    labels_dynamic = np.full(len(x_dynamic), 2, dtype=np.int64)

    # Combine all
    x_all = np.concatenate([x_drivable, x_static, x_dynamic]).astype(np.float32)
    y_all = np.concatenate([y_drivable, y_static, y_dynamic]).astype(np.float32)
    z_all = np.concatenate([z_drivable, z_static, z_dynamic]).astype(np.float32)
    labels_all = np.concatenate([labels_drivable, labels_static, labels_dynamic]).astype(np.int64)

    n_total = len(labels_all)
    intensity = np.random.uniform(0.1, 1.0, n_total).astype(np.float32)
    ring = np.random.randint(0, 32, n_total).astype(np.float32)

    points = np.column_stack([x_all, y_all, z_all, intensity, ring]).astype(np.float32)
    model_points = np.copy(points)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        points=points,
        model_points=model_points,
        labels=labels_all,
        source_file="synthetic_mock_frame_001.bin",
    )
    print(f"Successfully generated synthetic mock dataset at: {output_path}")
    print(f"Total points: {n_total}")
    print(f"  - Drivable (0): {np.sum(labels_all == 0)}")
    print(f"  - Static   (1): {np.sum(labels_all == 1)}")
    print(f"  - Dynamic  (2): {np.sum(labels_all == 2)}")


if __name__ == "__main__":
    generate_synthetic_mock_npz()
