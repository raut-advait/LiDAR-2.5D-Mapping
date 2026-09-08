import numpy as np


# ============================================================
# REMOVE INVALID POINTS
# ============================================================

def remove_invalid_points(points):

    valid_mask = np.isfinite(
        points
    ).all(
        axis=1
    )

    return points[valid_mask]


# ============================================================
# REMOVE EGO VEHICLE POINTS
# ============================================================

def remove_ego_vehicle_points(points):

    """
    Remove points belonging to the ego vehicle itself.

    This is a GEOMETRY-BASED filter.

    We are NOT using:
        - Ground truth labels
        - Dataset semantic labels
        - Predefined object labels

    The filter removes points inside the approximate
    physical region occupied by the sensor vehicle.
    """


    # ========================================================
    # EXTRACT REAL XYZ COORDINATES
    # ========================================================

    x = points[:, 0]

    y = points[:, 1]

    z = points[:, 2]


    # ========================================================
    # EGO VEHICLE EXCLUSION REGION
    #
    # These values can be adjusted later.
    #
    # X = forward / backward
    # Y = left / right
    # Z = height
    # ========================================================

    ego_mask = (

        (x >= -2.5)
        &
        (x <= 2.5)

        &
        (y >= -1.5)
        &
        (y <= 1.5)

        &
        (z >= -2.5)
        &
        (z <= 1.0)
    )


    # ========================================================
    # KEEP ONLY NON-EGO POINTS
    # ========================================================

    filtered_points = points[
        ~ego_mask
    ]


    removed_points = int(
        np.sum(
            ego_mask
        )
    )


    print(
        f"Ego vehicle points removed: "
        f"{removed_points}"
    )


    print(
        f"Remaining points: "
        f"{len(filtered_points)}"
    )


    return filtered_points


# ============================================================
# SAMPLE POINTS
# ============================================================

def sample_points(
    points,
    num_points=32768
):

    total_points = len(
        points
    )


    if total_points == 0:

        raise ValueError(
            "No valid LiDAR points available."
        )


    # ========================================================
    # ENOUGH POINTS
    # ========================================================

    if total_points >= num_points:

        indices = np.random.choice(

            total_points,

            num_points,

            replace=False
        )


    # ========================================================
    # FEWER POINTS
    # ========================================================

    else:

        indices = np.random.choice(

            total_points,

            num_points,

            replace=True
        )


    sampled_points = points[
        indices
    ]


    return sampled_points


# ============================================================
# NORMALIZE XYZ
# ============================================================

def normalize_xyz(points):

    """
    Normalize XYZ coordinates for model inference.

    IMPORTANT:

    This normalized version is ONLY for the model.

    The original real-world coordinates are preserved
    separately for adaptive mapping.
    """


    points = points.copy()


    # ========================================================
    # EXTRACT XYZ
    # ========================================================

    xyz = points[:, :3]


    # ========================================================
    # CALCULATE CENTER
    # ========================================================

    xyz_center = np.mean(

        xyz,

        axis=0
    )


    # ========================================================
    # CENTER POINT CLOUD
    # ========================================================

    xyz = xyz - xyz_center


    # ========================================================
    # CALCULATE DISTANCES
    # ========================================================

    distances = np.sqrt(

        np.sum(

            xyz ** 2,

            axis=1
        )
    )


    max_distance = np.max(
        distances
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    if max_distance > 0:

        xyz = (

            xyz
            /
            max_distance
        )


    # ========================================================
    # PUT XYZ BACK
    # ========================================================

    points[:, :3] = xyz


    return points


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_lidar(
    points,
    num_points=32768
):

    print(
        "\nStarting preprocessing..."
    )


    # ========================================================
    # STEP 1
    # REMOVE INVALID POINTS
    # ========================================================

    points = remove_invalid_points(
        points
    )


    print(
        f"Valid points: "
        f"{len(points)}"
    )


    # ========================================================
    # STEP 2
    # REMOVE EGO VEHICLE POINTS
    # ========================================================

    points = remove_ego_vehicle_points(
        points
    )


    # ========================================================
    # STEP 3
    # SAMPLE REAL POINTS
    # ========================================================

    real_mapping_points = sample_points(

        points,

        num_points
    )


    print(
        f"Sampled real points shape: "
        f"{real_mapping_points.shape}"
    )


    # ========================================================
    # STEP 4
    # CREATE MODEL INPUT
    # ========================================================

    model_input_points = normalize_xyz(
        real_mapping_points
    )


    print(
        "XYZ coordinates normalized "
        "for model inference!"
    )


    print(
        f"Real mapping points shape: "
        f"{real_mapping_points.shape}"
    )


    print(
        f"Model input points shape: "
        f"{model_input_points.shape}"
    )


    # ========================================================
    # RETURN BOTH
    # ========================================================

    return (

        real_mapping_points,

        model_input_points
    )