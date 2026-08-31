from pathlib import Path

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# ADAPTIVE MAP DIRECTORY
# ============================================================

ADAPTIVE_MAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "adaptive_maps_2_5d"
)


# ============================================================
# FIND ADAPTIVE MAP FILES
# ============================================================

adaptive_files = sorted(
    ADAPTIVE_MAP_DIR.glob(
        "*.npz"
    )
)


if len(adaptive_files) == 0:

    raise FileNotFoundError(
        f"No adaptive 2.5D files found in:\n"
        f"{ADAPTIVE_MAP_DIR}"
    )


# ============================================================
# SELECT FIRST FILE
# ============================================================

file_path = adaptive_files[0]


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)

print(
    "ADAPTIVE 2.5D MAP INSPECTION"
)

print("=" * 70)


print(
    f"\nInspecting file:\n"
    f"{file_path.name}"
)


# ============================================================
# LOAD FILE
# ============================================================

data = np.load(
    file_path,
    allow_pickle=True
)


# ============================================================
# BASIC INFORMATION
# ============================================================

cells = data["cells"]


input_points = int(
    data["input_points"]
)


total_cells = int(
    data["total_cells"]
)


print("\n" + "-" * 70)

print(
    "BASIC INFORMATION"
)

print("-" * 70)


print(
    f"\nInput points: "
    f"{input_points}"
)


print(
    f"Adaptive cells: "
    f"{total_cells}"
)


# ============================================================
# LOAD DISTANCE CONFIGURATION
# ============================================================

near_distance = float(
    data["near_distance"]
)


medium_distance = float(
    data["medium_distance"]
)


far_distance = float(
    data["far_distance"]
)


print("\n" + "-" * 70)

print(
    "DISTANCE CONFIGURATION"
)

print("-" * 70)


print(
    f"\nNear distance: "
    f"{near_distance:.2f} m"
)


print(
    f"Medium distance: "
    f"{medium_distance:.2f} m"
)


print(
    f"Far distance: "
    f"{far_distance:.2f} m"
)


# ============================================================
# CONVERT OBJECT ARRAY TO PYTHON OBJECTS
# ============================================================

cells_list = []


for cell in cells:

    # NumPy object arrays may contain
    # Python dictionaries directly.

    if isinstance(
        cell,
        dict
    ):

        cells_list.append(
            cell
        )


    # Sometimes NumPy stores dictionary
    # as a zero-dimensional object.

    elif isinstance(
        cell,
        np.ndarray
    ):

        try:

            extracted_cell = cell.item()

            if isinstance(
                extracted_cell,
                dict
            ):

                cells_list.append(
                    extracted_cell
                )

        except ValueError:

            pass


print(
    f"\nValid cell objects loaded: "
    f"{len(cells_list)}"
)


# ============================================================
# DISTANCE ZONE STATISTICS
# ============================================================

print("\n" + "-" * 70)

print(
    "DISTANCE ZONE STATISTICS"
)

print("-" * 70)


zones = {

    "near": {
        "cells": 0,
        "points": 0,
        "layers": []
    },

    "medium": {
        "cells": 0,
        "points": 0,
        "layers": []
    },

    "far": {
        "cells": 0,
        "points": 0,
        "layers": []
    }

}


# ============================================================
# CALCULATE ZONE STATISTICS
# ============================================================

for cell in cells_list:

    zone = cell.get(
        "zone",
        "unknown"
    )


    if zone not in zones:

        continue


    zones[zone]["cells"] += 1


    # --------------------------------------------------------
    # POINT COUNT
    # --------------------------------------------------------

    if "point_count" in cell:

        point_count = int(
            cell["point_count"]
        )


    elif "points" in cell:

        points_data = cell["points"]

        try:

            point_count = len(
                points_data
            )

        except TypeError:

            point_count = 0


    else:

        point_count = 0


    zones[zone]["points"] += (
        point_count
    )


    # --------------------------------------------------------
    # VERTICAL LAYER COUNT
    # --------------------------------------------------------

    if "layers" in cell:

        layer_count = len(
            cell["layers"]
        )


    elif "vertical_layers" in cell:

        vertical_layers = cell[
            "vertical_layers"
        ]

        try:

            layer_count = len(
                vertical_layers
            )

        except TypeError:

            layer_count = int(
                vertical_layers
            )


    elif "layer_count" in cell:

        layer_count = int(
            cell[
                "layer_count"
            ]
        )


    else:

        layer_count = 0


    zones[zone]["layers"].append(
        layer_count
    )


# ============================================================
# PRINT ZONE STATISTICS
# ============================================================

for zone_name in [

    "near",
    "medium",
    "far"

]:

    zone_data = zones[
        zone_name
    ]


    print(
        f"\nZone: "
        f"{zone_name.upper()}"
    )


    print(
        f"Cells: "
        f"{zone_data['cells']}"
    )


    print(
        f"Points: "
        f"{zone_data['points']}"
    )


    if len(
        zone_data["layers"]
    ) > 0:

        average_layers = np.mean(
            zone_data["layers"]
        )


    else:

        average_layers = 0.0


    print(
        f"Average layers per cell: "
        f"{average_layers:.2f}"
    )


# ============================================================
# SAMPLE ADAPTIVE CELLS
# ============================================================

print("\n" + "-" * 70)

print(
    "SAMPLE ADAPTIVE CELLS"
)

print("-" * 70)


# ============================================================
# NUMBER OF SAMPLE CELLS
# ============================================================

NUM_SAMPLE_CELLS = min(
    5,
    len(cells_list)
)


# ============================================================
# PRINT CELL DETAILS
# ============================================================

for index, cell in enumerate(

    cells_list[
        :NUM_SAMPLE_CELLS
    ],

    start=1
):

    print("\n")

    print(
        f"CELL {index}"
    )


    # --------------------------------------------------------
    # ZONE
    # --------------------------------------------------------

    zone = cell.get(
        "zone",
        "Unknown"
    )


    print(
        f"Zone: "
        f"{zone}"
    )


    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    resolution = cell.get(
        "resolution",
        "N/A"
    )


    if isinstance(
        resolution,
        (float, int, np.floating, np.integer)
    ):

        print(
            f"Resolution: "
            f"{float(resolution):.2f} m"
        )


    else:

        print(
            f"Resolution: "
            f"{resolution}"
        )


    # --------------------------------------------------------
    # GRID INDEX
    # --------------------------------------------------------
    # Handle different possible key names.
    # This prevents KeyError.

    grid_index = cell.get(

        "grid_index",

        cell.get(

            "cell_index",

            cell.get(

                "index",

                "Not stored"

            )

        )

    )


    print(
        f"Grid Index: "
        f"{grid_index}"
    )


    # --------------------------------------------------------
    # POINT COUNT
    # --------------------------------------------------------

    if "point_count" in cell:

        point_count = int(
            cell[
                "point_count"
            ]
        )


    elif "points" in cell:

        try:

            point_count = len(
                cell[
                    "points"
                ]
            )

        except TypeError:

            point_count = 0


    else:

        point_count = 0


    print(
        f"Points: "
        f"{point_count}"
    )


    # --------------------------------------------------------
    # HEIGHT INFORMATION
    # --------------------------------------------------------

    min_height = cell.get(
        "min_height",
        None
    )


    max_height = cell.get(
        "max_height",
        None
    )


    if (

        min_height is not None

        and

        max_height is not None

    ):

        print(
            f"Height Range: "
            f"{float(min_height):.3f} m "
            f"to "
            f"{float(max_height):.3f} m"
        )


        height_variation = (

            float(max_height)

            -

            float(min_height)

        )


        print(
            f"Height Variation: "
            f"{height_variation:.3f} m"
        )


    else:

        print(
            "Height Range: Not directly stored"
        )


    # --------------------------------------------------------
    # VERTICAL LAYERS
    # --------------------------------------------------------

    if "layers" in cell:

        layers = cell[
            "layers"
        ]


        layer_count = len(
            layers
        )


    elif "vertical_layers" in cell:

        layers = cell[
            "vertical_layers"
        ]


        try:

            layer_count = len(
                layers
            )

        except TypeError:

            layer_count = int(
                layers
            )


    elif "layer_count" in cell:

        layer_count = int(
            cell[
                "layer_count"
            ]
        )


    else:

        layer_count = 0


    print(
        f"Vertical Layers: "
        f"{layer_count}"
    )


    # --------------------------------------------------------
    # SHOW AVAILABLE KEYS
    # --------------------------------------------------------

    print(
        "\nAvailable Keys:"
    )


    print(
        list(
            cell.keys()
        )
    )


# ============================================================
# OVERALL EFFICIENCY SUMMARY
# ============================================================

print("\n" + "-" * 70)

print(
    "EFFICIENCY SUMMARY"
)

print("-" * 70)


print(
    f"\nOriginal points: "
    f"{input_points}"
)


print(
    f"Stored adaptive cells: "
    f"{len(cells_list)}"
)


if len(cells_list) > 0:

    average_points_per_cell = (

        input_points

        /

        len(cells_list)

    )


else:

    average_points_per_cell = 0.0


print(
    f"Points per adaptive cell (average): "
    f"{average_points_per_cell:.2f}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "INSPECTION COMPLETED!"
)

print("=" * 70)