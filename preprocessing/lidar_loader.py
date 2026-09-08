import numpy as np
from pathlib import Path


# ============================================================
# LOAD RAW LIDAR FILE
# ============================================================

def load_lidar_file(file_path):

    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():

        raise FileNotFoundError(
            f"LiDAR file not found:\n{file_path}"
        )


    # Load raw binary data
    raw_data = np.fromfile(
        file_path,
        dtype=np.float32
    )


    # Check data format
    if raw_data.size % 5 != 0:

        raise ValueError(
            "Invalid LiDAR file format. "
            "Data cannot be reshaped into 5 features."
        )


    # Reshape to points
    points = raw_data.reshape(
        -1,
        5
    )


    return points


# ============================================================
# TEST LOADER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("RAW LIDAR LOADER TEST")

    print("=" * 60)


    # Project root
    PROJECT_ROOT = Path(
        __file__
    ).resolve().parent.parent


    # Raw data folder
    RAW_DATA_DIR = (
        PROJECT_ROOT /
        "data" /
        "raw"
    )


    # Find raw LiDAR files
    lidar_files = list(
        RAW_DATA_DIR.glob(
            "*.pcd.bin"
        )
    )


    if len(lidar_files) == 0:

        print(
            "\n❌ No .pcd.bin files found!"
        )


    else:

        print(
            f"\nFound {len(lidar_files)} "
            f"LiDAR files."
        )


        # Test first file
        file_path = lidar_files[0]


        print(
            f"\nTesting file:\n"
            f"{file_path.name}"
        )


        # Load points
        points = load_lidar_file(
            file_path
        )


        print(
            f"\nPoint cloud shape: "
            f"{points.shape}"
        )


        print(
            f"\nNumber of points: "
            f"{len(points)}"
        )


        print(
            f"\nFeatures per point: "
            f"{points.shape[1]}"
        )


        print(
            "\nFirst 5 points:"
        )


        print(
            points[:5]
        )


        print("\n" + "=" * 60)

        print("LOADER TEST COMPLETED! ✅")

        print("=" * 60)