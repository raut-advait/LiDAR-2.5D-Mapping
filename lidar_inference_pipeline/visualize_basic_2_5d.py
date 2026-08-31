from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# INPUT DIRECTORY
# ============================================================

MAPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "maps_2_5d"
)


# ============================================================
# GET AVAILABLE MAP FILES
# ============================================================

map_files = sorted(
    MAPS_DIR.glob("*.npz")
)


if len(map_files) == 0:

    raise FileNotFoundError(
        "No 2.5D map files found."
    )


# ============================================================
# SELECT FIRST MAP
# ============================================================

file_path = map_files[0]


print("\n" + "=" * 60)

print("BASIC 2.5D MAP VISUALIZATION")

print("=" * 60)


print(
    f"\nVisualizing: "
    f"{file_path.name}"
)


# ============================================================
# LOAD MAP
# ============================================================

data = np.load(
    file_path
)


height_grid = data[
    "height"
]


label_grid = data[
    "labels"
]


point_count_grid = data[
    "point_count"
]


# ============================================================
# CREATE HEIGHT MAP
# ============================================================

plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    height_grid.T,
    origin="lower",
    aspect="auto"
)


plt.colorbar(
    label="Height (meters)"
)


plt.title(
    "Basic 2.5D Height Map"
)


plt.xlabel(
    "Grid X"
)


plt.ylabel(
    "Grid Y"
)


plt.tight_layout()


plt.show()


# ============================================================
# CREATE SEMANTIC LABEL MAP
# ============================================================

semantic_display = np.where(
    label_grid.T == -1,
    np.nan,
    label_grid.T
)


plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    semantic_display,
    origin="lower",
    aspect="auto"
)


plt.colorbar(
    label="Semantic Class"
)


plt.title(
    "Basic 2.5D Semantic Map"
)


plt.xlabel(
    "Grid X"
)


plt.ylabel(
    "Grid Y"
)


plt.tight_layout()


plt.show()


# ============================================================
# CREATE POINT DENSITY MAP
# ============================================================

density_display = np.where(
    point_count_grid.T == 0,
    np.nan,
    point_count_grid.T
)


plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    density_display,
    origin="lower",
    aspect="auto"
)


plt.colorbar(
    label="Points per Cell"
)


plt.title(
    "Basic 2.5D Point Density Map"
)


plt.xlabel(
    "Grid X"
)


plt.ylabel(
    "Grid Y"
)


plt.tight_layout()


plt.show()


# ============================================================
# COMPLETED
# ============================================================

print("\nVisualization completed!")