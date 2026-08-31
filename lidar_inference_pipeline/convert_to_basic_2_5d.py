from pathlib import Path

import numpy as np

from conversion.basic_2_5d_converter import (
    Basic2_5DConverter
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


# ============================================================
# INPUT DIRECTORY
# ============================================================

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "maps_2_5d"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

GRID_RESOLUTION = 0.5


# ============================================================
# CREATE CONVERTER
# ============================================================

converter = Basic2_5DConverter(
    resolution=GRID_RESOLUTION
)


# ============================================================
# GET PREDICTION FILES
# ============================================================

prediction_files = sorted(
    PREDICTIONS_DIR.glob(
        "*.npz"
    )
)


if len(prediction_files) == 0:

    raise FileNotFoundError(
        "No prediction files found."
    )


# ============================================================
# START CONVERSION
# ============================================================

print("\n" + "=" * 60)

print(
    "BASIC 3D TO 2.5D CONVERSION"
)

print("=" * 60)


print(
    f"\nPrediction files found: "
    f"{len(prediction_files)}"
)


# ============================================================
# PROCESS EACH FILE
# ============================================================

successful_files = 0

failed_files = 0


for file_path in prediction_files:

    print("\n" + "-" * 60)

    print(
        f"Processing: "
        f"{file_path.name}"
    )


    try:

        # ====================================================
        # LOAD PREDICTION DATA
        # ====================================================

        data = np.load(
            file_path,
            allow_pickle=True
        )


        points = data[
            "points"
        ]


        labels = data[
            "labels"
        ]


        print(
            f"Points shape: "
            f"{points.shape}"
        )


        print(
            f"Labels shape: "
            f"{labels.shape}"
        )


        # ====================================================
        # CONVERT TO 2.5D
        # ====================================================

        grid = converter.convert(
            points=points,
            labels=labels
        )


        # ====================================================
        # CREATE OUTPUT FILE NAME
        # ====================================================

        output_name = (
            file_path.stem
            .replace(
                "_prediction",
                "_basic_2_5d"
            )
            + ".npz"
        )


        output_path = (
            OUTPUT_DIR
            / output_name
        )


        # ====================================================
        # SAVE 2.5D MAP
        # ====================================================

        np.savez_compressed(
            output_path,

            height=grid[
                "height"
            ],

            labels=grid[
                "labels"
            ],

            point_count=grid[
                "point_count"
            ],

            resolution=grid[
                "resolution"
            ],

            min_x=grid[
                "min_x"
            ],

            max_x=grid[
                "max_x"
            ],

            min_y=grid[
                "min_y"
            ],

            max_y=grid[
                "max_y"
            ]
        )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        print(
            "✅ Conversion successful!"
        )


        print(
            f"Grid shape: "
            f"{grid['height'].shape}"
        )


        print(
            f"Saved: "
            f"{output_path.name}"
        )


        successful_files += 1


    except Exception as error:

        print(
            f"❌ Failed: "
            f"{error}"
        )


        failed_files += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "CONVERSION COMPLETED"
)

print("=" * 60)


print(
    f"\nSuccessful: "
    f"{successful_files}"
)


print(
    f"Failed: "
    f"{failed_files}"
)


print(
    f"\nOutput directory:"
)


print(
    OUTPUT_DIR
)


print("\n" + "=" * 60)