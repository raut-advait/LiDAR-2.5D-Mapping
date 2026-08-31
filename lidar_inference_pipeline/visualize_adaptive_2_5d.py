from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import (
    Rectangle,
    Circle,
    Patch
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


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

        f"No adaptive map files found in:\n"
        f"{ADAPTIVE_MAP_DIR}"
    )


# ============================================================
# SELECT FILE
# ============================================================
#
# Change [0] to another index if you want another frame.
#
# ============================================================

file_path = adaptive_files[0]


# ============================================================
# LOAD ADAPTIVE MAP
# ============================================================

print(
    "\nLoading adaptive map..."
)


print(

    f"File:\n"
    f"{file_path.name}"
)


data = np.load(

    file_path,

    allow_pickle=True
)


cells = data["cells"]


# ============================================================
# DISTANCE CONFIGURATION
# ============================================================

near_distance = float(

    data[
        "near_distance"
    ]
)


medium_distance = float(

    data[
        "medium_distance"
    ]
)


far_distance = float(

    data[
        "far_distance"
    ]
)


# ============================================================
# CREATE PLOT
# ============================================================

fig, ax = plt.subplots(

    figsize=(12, 12)
)


# ============================================================
# SEMANTIC COLORS
# ============================================================
#
# MODEL GENERATED LABELS
#
# Class 0 -> Drivable Surface
#
# Class 1 -> Non-Drivable / Static Object
#
# Class 2 -> Dynamic Object
#
# ============================================================

SEMANTIC_COLORS = {

    0: "gray",

    1: "black",

    2: "red"
}


# ============================================================
# EGO VEHICLE FILTER
# ============================================================
#
# LiDAR sensor is mounted on the ego vehicle.
#
# Points/cells belonging to the ego vehicle itself should not
# be visualized as Dynamic Objects.
#
# These boundaries define the approximate ego vehicle area.
#
# ============================================================


# Left / Right boundary

EGO_X_MIN = -2.5

EGO_X_MAX = 2.5


# Front / Back boundary

EGO_Y_MIN = -4.5

EGO_Y_MAX = 2.5


# ============================================================
# COUNTERS
# ============================================================

near_count = 0

medium_count = 0

far_count = 0


drivable_count = 0

static_count = 0

dynamic_count = 0


ego_filtered_count = 0


# ============================================================
# DETERMINE CELL SEMANTIC LABEL
# ============================================================

def get_cell_label(layers):

    """
    Determine final semantic label for an adaptive cell.

    Labels come from model predictions.

    Priority:

        Dynamic Object (2)

                ↓

        Static / Non-Drivable (1)

                ↓

        Drivable Surface (0)

    Dynamic has highest priority so that it remains visible.
    """


    # ========================================================
    # EMPTY CELL
    # ========================================================

    if layers is None:

        return 1


    if len(layers) == 0:

        return 1


    # ========================================================
    # COLLECT MODEL-PREDICTED LABELS
    # ========================================================

    layer_labels = []


    for layer in layers:


        if not isinstance(
            layer,
            dict
        ):

            continue


        if "label" in layer:

            try:

                label = int(

                    layer[
                        "label"
                    ]
                )


                layer_labels.append(
                    label
                )


            except (
                ValueError,
                TypeError
            ):

                continue


    # ========================================================
    # NO VALID LABEL
    # ========================================================

    if len(layer_labels) == 0:

        return 1


    # ========================================================
    # PRIORITY 1
    #
    # DYNAMIC OBJECT
    # ========================================================

    if 2 in layer_labels:

        return 2


    # ========================================================
    # PRIORITY 2
    #
    # STATIC / NON-DRIVABLE
    # ========================================================

    if 1 in layer_labels:

        return 1


    # ========================================================
    # PRIORITY 3
    #
    # DRIVABLE SURFACE
    # ========================================================

    return 0


# ============================================================
# PROCESS ADAPTIVE CELLS
# ============================================================

for raw_cell in cells:


    # ========================================================
    # EXTRACT CELL DICTIONARY
    # ========================================================

    if isinstance(

        raw_cell,

        dict
    ):

        cell = raw_cell


    elif isinstance(

        raw_cell,

        np.ndarray
    ):


        try:

            cell = raw_cell.item()


        except ValueError:

            continue


    else:

        continue


    # ========================================================
    # VALIDATE CELL
    # ========================================================

    if not isinstance(
        cell,
        dict
    ):

        continue


    # ========================================================
    # GET CELL INFORMATION
    # ========================================================

    zone = cell.get(

        "zone",

        "unknown"
    )


    try:

        resolution = float(

            cell[
                "resolution"
            ]
        )


        x_index = int(

            cell[
                "x_index"
            ]
        )


        y_index = int(

            cell[
                "y_index"
            ]
        )


    except (
        KeyError,
        ValueError,
        TypeError
    ):

        continue


    layers = cell.get(

        "layers",

        []
    )


    # ========================================================
    # DETERMINE SEMANTIC LABEL
    #
    # LABEL IS GENERATED BY THE MODEL
    # ========================================================

    semantic_label = get_cell_label(

        layers
    )


    # ========================================================
    # GET SEMANTIC COLOR
    # ========================================================

    cell_color = SEMANTIC_COLORS.get(

        semantic_label,

        "black"
    )


    # ========================================================
    # CALCULATE REAL CELL POSITION
    # ========================================================

    x = (

        x_index

        *

        resolution
    )


    y = (

        y_index

        *

        resolution
    )


    # ========================================================
    # CALCULATE CELL CENTER
    # ========================================================

    cell_center_x = (

        x

        +

        resolution / 2
    )


    cell_center_y = (

        y

        +

        resolution / 2
    )


    # ========================================================
    # EGO VEHICLE FILTER
    # ========================================================
    #
    # Hide cells belonging to the vehicle carrying the sensor.
    #
    # This prevents the ego vehicle from appearing as a
    # Dynamic Object in the visualization.
    #
    # ========================================================

    inside_ego_vehicle = (

        EGO_X_MIN

        <=

        cell_center_x

        <=

        EGO_X_MAX

        and

        EGO_Y_MIN

        <=

        cell_center_y

        <=

        EGO_Y_MAX
    )


    if inside_ego_vehicle:

        ego_filtered_count += 1

        continue


    # ========================================================
    # CREATE SEMANTIC CELL
    # ========================================================
    #
    # Only semantic color.
    #
    # No cell border.
    #
    # ========================================================

    rectangle = Rectangle(

        (

            x,

            y
        ),

        resolution,

        resolution,


        facecolor=cell_color,


        edgecolor="none",


        linewidth=0,


        alpha=0.85
    )


    ax.add_patch(

        rectangle
    )


    # ========================================================
    # COUNT DISTANCE ZONES
    # ========================================================

    if zone == "near":

        near_count += 1


    elif zone == "medium":

        medium_count += 1


    elif zone == "far":

        far_count += 1


    # ========================================================
    # COUNT SEMANTIC CLASSES
    # ========================================================

    if semantic_label == 0:

        drivable_count += 1


    elif semantic_label == 1:

        static_count += 1


    elif semantic_label == 2:

        dynamic_count += 1


# ============================================================
# DRAW DISTANCE RANGE CIRCLES
# ============================================================
#
# ONLY BORDERS
#
# NO FILL
#
# ============================================================


# ============================================================
# NEAR RANGE
# ============================================================

near_circle = Circle(

    (

        0,

        0
    ),


    radius=near_distance,


    fill=False,


    edgecolor="blue",


    linewidth=2
)


ax.add_patch(

    near_circle
)


# ============================================================
# MEDIUM RANGE
# ============================================================

medium_circle = Circle(

    (

        0,

        0
    ),


    radius=medium_distance,


    fill=False,


    edgecolor="green",


    linewidth=2
)


ax.add_patch(

    medium_circle
)


# ============================================================
# FAR RANGE
# ============================================================

far_circle = Circle(

    (

        0,

        0
    ),


    radius=far_distance,


    fill=False,


    edgecolor="orange",


    linewidth=2
)


ax.add_patch(

    far_circle
)


# ============================================================
# MARK LIDAR SENSOR POSITION
# ============================================================
#
# Optional:
# Shows the center position of the LiDAR sensor.
#
# ============================================================

ax.scatter(

    0,

    0,


    marker="x",


    color="blue",


    s=50,


    linewidths=2,


    zorder=10
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

ax.set_title(

    "Semantic Adaptive 2.5D LiDAR Map",

    fontsize=18
)


ax.set_xlabel(

    "X Distance (meters)"
)


ax.set_ylabel(

    "Y Distance (meters)"
)


ax.set_aspect(

    "equal"
)


ax.grid(

    True,


    linestyle="--",


    alpha=0.3
)


# ============================================================
# DISPLAY RANGE
# ============================================================

display_limit = (

    far_distance

    +

    5
)


ax.set_xlim(

    -display_limit,

    display_limit
)


ax.set_ylim(

    -display_limit,

    display_limit
)


# ============================================================
# CREATE LEGEND
# ============================================================

legend_elements = [


    # ========================================================
    # SEMANTIC CLASSES
    # ========================================================

    Patch(

        facecolor="gray",

        edgecolor="none",

        label="Drivable Surface (Class 0)"
    ),


    Patch(

        facecolor="black",

        edgecolor="none",

        label="Non-Drivable / Static (Class 1)"
    ),


    Patch(

        facecolor="red",

        edgecolor="none",

        label="Dynamic Object (Class 2)"
    ),


    # ========================================================
    # DISTANCE RANGES
    # ========================================================

    plt.Line2D(

        [0],

        [0],


        color="blue",


        linewidth=2,


        label=(
            f"Near Range "
            f"({near_distance:.0f} m)"
        )
    ),


    plt.Line2D(

        [0],

        [0],


        color="green",


        linewidth=2,


        label=(
            f"Medium Range "
            f"({medium_distance:.0f} m)"
        )
    ),


    plt.Line2D(

        [0],

        [0],


        color="orange",


        linewidth=2,


        label=(
            f"Far Range "
            f"({far_distance:.0f} m)"
        )
    )

]


ax.legend(

    handles=legend_elements,

    loc="upper right"
)


# ============================================================
# PRINT STATISTICS
# ============================================================

print(

    "\n"

    +

    "=" * 60
)


print(

    "VISUALIZATION STATISTICS"
)


print(

    "=" * 60
)


# ============================================================
# DISTANCE ZONES
# ============================================================

print(

    "\nDISTANCE ZONES"
)


print(

    f"Near cells: "
    f"{near_count}"
)


print(

    f"Medium cells: "
    f"{medium_count}"
)


print(

    f"Far cells: "
    f"{far_count}"
)


# ============================================================
# SEMANTIC INFORMATION
# ============================================================

print(

    "\nSEMANTIC INFORMATION"
)


print(

    f"Drivable cells: "
    f"{drivable_count}"
)


print(

    f"Non-drivable / Static cells: "
    f"{static_count}"
)


print(

    f"Dynamic object cells: "
    f"{dynamic_count}"
)


# ============================================================
# EGO VEHICLE INFORMATION
# ============================================================

print(

    "\nEGO VEHICLE FILTER"
)


print(

    f"Filtered ego vehicle cells: "
    f"{ego_filtered_count}"
)


print(

    f"Ego X range: "
    f"{EGO_X_MIN} m "
    f"to "
    f"{EGO_X_MAX} m"
)


print(

    f"Ego Y range: "
    f"{EGO_Y_MIN} m "
    f"to "
    f"{EGO_Y_MAX} m"
)


# ============================================================
# TOTAL
# ============================================================

total_visible_cells = (

    near_count

    +

    medium_count

    +

    far_count
)


print(

    f"\nVisible adaptive cells: "
    f"{total_visible_cells}"
)


print(

    f"Original adaptive cells: "
    f"{len(cells)}"
)


print(

    "\n"

    +

    "=" * 60
)


# ============================================================
# DISPLAY
# ============================================================

plt.tight_layout()


plt.show()