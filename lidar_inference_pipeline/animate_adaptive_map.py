from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

FPS = 10

INTERVAL = 100   # milliseconds

MAX_FRAMES = None

SAVE_VIDEO = True

VIDEO_NAME = "adaptive_lidar_animation.gif"


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
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (

    PROJECT_ROOT
    / "data"
    / "processed"
    / "visualizations"

)


OUTPUT_DIR.mkdir(

    parents=True,
    exist_ok=True

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
# LIMIT FRAMES
# ============================================================

if MAX_FRAMES is not None:

    adaptive_files = adaptive_files[
        :MAX_FRAMES
    ]


print("\n" + "=" * 65)

print(
    "ADAPTIVE 2.5D LIDAR ANIMATION"
)

print("=" * 65)


print(
    f"\nFrames found: "
    f"{len(adaptive_files)}"
)


# ============================================================
# LOAD ALL FRAME DATA
# ============================================================

frames = []


for index, file_path in enumerate(

    adaptive_files,
    start=1

):

    print(

        f"Loading frame "
        f"[{index}/{len(adaptive_files)}]: "
        f"{file_path.name}"

    )


    data = np.load(

        file_path,
        allow_pickle=True

    )


    cells = data["cells"]


    frame_cells = []


    for raw_cell in cells:


        # ====================================================
        # CONVERT TO DICTIONARY
        # ====================================================

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


        # ====================================================
        # VALIDATE REQUIRED DATA
        # ====================================================

        required_keys = [

            "zone",
            "resolution",
            "x_index",
            "y_index",
            "layers"

        ]


        if not all(

            key in cell
            for key in required_keys

        ):

            continue


        # ====================================================
        # CELL POSITION
        # ====================================================

        resolution = float(

            cell["resolution"]

        )


        x_index = int(

            cell["x_index"]

        )


        y_index = int(

            cell["y_index"]

        )


        x = (

            x_index
            * resolution

        )


        y = (

            y_index
            * resolution

        )


        # ====================================================
        # SEMANTIC LABEL
        # ====================================================

        semantic_label = None


        layers = cell.get(

            "layers",
            []

        )


        for layer in layers:


            if not isinstance(

                layer,
                dict

            ):

                continue


            # Try common label names

            if "label" in layer:

                semantic_label = int(
                    layer["label"]
                )

                break


            if "semantic_label" in layer:

                semantic_label = int(
                    layer["semantic_label"]
                )

                break


            if "class_id" in layer:

                semantic_label = int(
                    layer["class_id"]
                )

                break


        # ====================================================
        # DEFAULT LABEL
        # ====================================================

        if semantic_label is None:

            semantic_label = 1


        # ====================================================
        # STORE CELL
        # ====================================================

        frame_cells.append({

            "x": x,

            "y": y,

            "resolution": resolution,

            "label": semantic_label

        })


    frames.append({

        "cells": frame_cells,

        "name": file_path.name

    })


print(

    f"\nSuccessfully loaded "
    f"{len(frames)} frames."

)


# ============================================================
# COLOR CONFIGURATION
# ============================================================

SEMANTIC_COLORS = {

    # Class 0
    0: "gray",

    # Class 1
    1: "black",

    # Class 2
    2: "red"

}


# ============================================================
# RANGE CONFIGURATION
# ============================================================

NEAR_DISTANCE = 15

MEDIUM_DISTANCE = 35

FAR_DISTANCE = 60


# ============================================================
# DETERMINE FIXED AXIS LIMITS
# ============================================================

all_x = []

all_y = []


for frame in frames:

    for cell in frame["cells"]:

        all_x.append(
            cell["x"]
        )

        all_y.append(
            cell["y"]
        )


if len(all_x) == 0:

    raise ValueError(

        "No valid visualization cells found."

    )


PADDING = 5


x_min = min(

    all_x

) - PADDING


x_max = max(

    all_x

) + PADDING


y_min = min(

    all_y

) - PADDING


y_max = max(

    all_y

) + PADDING


# Ensure ranges are visible

x_min = min(
    x_min,
    -65
)


x_max = max(
    x_max,
    65
)


y_min = min(
    y_min,
    -65
)


y_max = max(
    y_max,
    65
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(

    figsize=(12, 12)

)


# ============================================================
# DRAW STATIC RANGE CIRCLES
# ============================================================

near_circle = plt.Circle(

    (0, 0),

    NEAR_DISTANCE,

    fill=False,

    color="blue",

    linewidth=2.5

)


medium_circle = plt.Circle(

    (0, 0),

    MEDIUM_DISTANCE,

    fill=False,

    color="green",

    linewidth=2.5

)


far_circle = plt.Circle(

    (0, 0),

    FAR_DISTANCE,

    fill=False,

    color="orange",

    linewidth=2.5

)


ax.add_patch(
    near_circle
)


ax.add_patch(
    medium_circle
)


ax.add_patch(
    far_circle
)


# ============================================================
# DRAW EGO VEHICLE POSITION
# ============================================================

ego_marker = ax.plot(

    0,

    0,

    marker="x",

    markersize=12,

    markeredgewidth=3,

    color="blue"

)[0]


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Patch(

        facecolor="gray",

        label="Drivable Surface (Class 0)"

    ),

    Patch(

        facecolor="black",

        label="Non-Drivable / Static (Class 1)"

    ),

    Patch(

        facecolor="red",

        label="Dynamic Object (Class 2)"

    ),

    Line2D(

        [0],

        [0],

        color="blue",

        linewidth=2,

        label="Near Range (15 m)"

    ),

    Line2D(

        [0],

        [0],

        color="green",

        linewidth=2,

        label="Medium Range (35 m)"

    ),

    Line2D(

        [0],

        [0],

        color="orange",

        linewidth=2,

        label="Far Range (60 m)"

    )

]


ax.legend(

    handles=legend_elements,

    loc="upper right"

)


# ============================================================
# GRAPH SETTINGS
# ============================================================

ax.set_xlim(

    x_min,
    x_max

)


ax.set_ylim(

    y_min,
    y_max

)


ax.set_aspect(

    "equal"

)


ax.set_xlabel(

    "X Distance (meters)",

    fontsize=12

)


ax.set_ylabel(

    "Y Distance (meters)",

    fontsize=12

)


ax.grid(

    True,

    linestyle="--",

    alpha=0.3

)


# ============================================================
# CURRENT CELL ARTISTS
# ============================================================

cell_artists = []


# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame_number):


    global cell_artists


    # ========================================================
    # REMOVE OLD FRAME
    # ========================================================

    for artist in cell_artists:

        artist.remove()


    cell_artists = []


    # ========================================================
    # GET CURRENT FRAME
    # ========================================================

    frame = frames[
        frame_number
    ]


    cells = frame[
        "cells"
    ]


    # ========================================================
    # DRAW CURRENT FRAME
    # ========================================================

    for cell in cells:


        x = cell[
            "x"
        ]


        y = cell[
            "y"
        ]


        resolution = cell[
            "resolution"
        ]


        label = cell[
            "label"
        ]


        color = SEMANTIC_COLORS.get(

            label,

            "black"

        )


        rectangle = plt.Rectangle(

            (

                x,

                y

            ),

            resolution,

            resolution,

            facecolor=color,

            edgecolor="none",

            alpha=0.85

        )


        ax.add_patch(

            rectangle

        )


        cell_artists.append(

            rectangle

        )


    # ========================================================
    # UPDATE TITLE
    # ========================================================

    ax.set_title(

        f"Semantic Adaptive 2.5D LiDAR Map\n"
        f"Frame {frame_number + 1} / {len(frames)}",

        fontsize=18

    )


    return cell_artists


# ============================================================
# CREATE ANIMATION
# ============================================================

animation = FuncAnimation(

    fig,

    update,

    frames=len(frames),

    interval=INTERVAL,

    repeat=True,

    blit=False

)


# ============================================================
# SHOW FIRST FRAME
# ============================================================

update(
    0
)


plt.tight_layout()


# ============================================================
# SAVE VIDEO
# ============================================================

if SAVE_VIDEO:


    output_video_path = (

        OUTPUT_DIR
        / VIDEO_NAME

    )


    print(

        "\nSaving video..."

    )


    try:


        animation.save(

            output_video_path,

            fps=FPS,

            dpi=120,

            writer="pillow"

        )


        print(

            "\n✅ VIDEO SAVED SUCCESSFULLY!"

        )


        print(

            f"\nVideo location:\n"
            f"{output_video_path}"

        )


    except Exception as error:


        print(

            "\n⚠️ VIDEO COULD NOT BE SAVED."

        )


        print(

            f"Reason: {error}"

        )


        print(

            "\nThe animation will still open."

        )


# ============================================================
# DISPLAY ANIMATION
# ============================================================

# plt.show()


# ============================================================
# COMPLETED
# ============================================================

print(

    "\n🎉 ANIMATION COMPLETED!"

)