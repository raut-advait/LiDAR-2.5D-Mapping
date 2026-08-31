from pathlib import Path
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 2.5D MAP DIRECTORY
# ============================================================

MAPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "maps_2_5d"
)


# ============================================================
# GET MAP FILES
# ============================================================

map_files = sorted(
    MAPS_DIR.glob("*.npz")
)


if len(map_files) == 0:
    raise FileNotFoundError(
        "No 2.5D map files found."
    )


# ============================================================
# SELECT FIRST FILE
# ============================================================

file_path = map_files[0]


print("\n" + "=" * 65)
print("BASIC 2.5D MAP INSPECTION")
print("=" * 65)

print(f"\nFile: {file_path.name}")


# ============================================================
# LOAD 2.5D MAP
# ============================================================

data = np.load(
    file_path,
    allow_pickle=True
)


# ============================================================
# DISPLAY AVAILABLE KEYS
# ============================================================

print("\nAvailable keys:")

for key in data.files:
    print(f" - {key}")


print("\n" + "=" * 65)
print("ARRAY DETAILS")
print("=" * 65)


# ============================================================
# INSPECT EACH KEY
# ============================================================

for key in data.files:

    value = data[key]

    print(f"\nKey: {key}")

    print(
        f"Shape: {value.shape}"
    )

    print(
        f"Dimensions: {value.ndim}"
    )

    print(
        f"Data type: {value.dtype}"
    )


    # ========================================================
    # SPECIAL INFORMATION FOR GRID ARRAYS
    # ========================================================

    if key == "height":

        valid_cells = np.sum(
            ~np.isnan(value)
        )

        total_cells = value.size

        empty_cells = total_cells - valid_cells

        print(
            f"\nValid cells: "
            f"{valid_cells}"
        )

        print(
            f"Empty cells: "
            f"{empty_cells}"
        )

        print(
            f"Total cells: "
            f"{total_cells}"
        )


        if valid_cells > 0:

            print(
                f"Min height: "
                f"{np.nanmin(value):.4f}"
            )

            print(
                f"Max height: "
                f"{np.nanmax(value):.4f}"
            )


    # ========================================================
    # SPECIAL INFORMATION FOR LABEL GRID
    # ========================================================

    elif key == "labels":

        valid_labels = value[
            value != -1
        ]


        if len(valid_labels) > 0:

            unique_labels, counts = np.unique(
                valid_labels,
                return_counts=True
            )


            print(
                "\nSemantic Label Distribution:"
            )


            for label, count in zip(
                unique_labels,
                counts
            ):

                percentage = (
                    count
                    / len(valid_labels)
                ) * 100


                print(
                    f"Class {label}: "
                    f"{count} cells "
                    f"({percentage:.2f}%)"
                )


    # ========================================================
    # SPECIAL INFORMATION FOR POINT COUNT GRID
    # ========================================================

    elif key == "point_count":

        occupied_cells = np.sum(
            value > 0
        )

        total_points = np.sum(
            value
        )


        print(
            f"\nOccupied cells: "
            f"{occupied_cells}"
        )


        print(
            f"Points represented: "
            f"{total_points}"
        )


    # ========================================================
    # DISPLAY SAMPLE VALUES
    # ========================================================

    print(
        "\nSample value:"
    )


    if value.ndim == 0:

        print(value)

    elif value.ndim == 1:

        print(
            value[:10]
        )

    elif value.ndim == 2:

        print(
            value[:5, :5]
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("INSPECTION COMPLETED!")
print("=" * 65)