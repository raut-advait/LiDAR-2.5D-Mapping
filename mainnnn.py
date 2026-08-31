from pathlib import Path

import numpy as np
import torch


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from preprocessing.lidar_loader import load_lidar_file

from preprocessing.preprocess import preprocess_lidar

from models.pointnet_model import (
    PointNetPlusPlusSegmentation
)

from inference.predict import (
    load_model,
    predict,
    print_prediction_summary
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_POINTS = 8192

NUM_CLASSES = 3

INPUT_FEATURES = 2

MAX_FRAMES = 5
# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


RAW_DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "raw"
)


CHECKPOINT_PATH = (
    PROJECT_ROOT /
    "checkpoints" /
    "best_weighted_pointnet_model.pth"
)


# Output folder
PREDICTION_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "predictions"
)


# Create output folder automatically
PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# START
# ============================================================

print("\n" + "=" * 60)

print("LIDAR BATCH INFERENCE PIPELINE")

print("=" * 60)

print(f"\nDevice: {device}")


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        f"\nModel checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# FIND ALL RAW LIDAR FILES
# ============================================================

lidar_files = sorted(
    RAW_DATA_DIR.glob(
        "*.pcd.bin"
    )
)

# ============================================================
# LIMIT TO FIRST 5 FRAMES
# ============================================================

lidar_files = lidar_files[:MAX_FRAMES]


if len(lidar_files) == 0:

    raise FileNotFoundError(
        f"\nNo .pcd.bin files found in:\n"
        f"{RAW_DATA_DIR}"
    )


print(
    f"\nFound {len(lidar_files)} "
    f"raw LiDAR file(s)."
)


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "\nCreating PointNet++ model..."
)


model = PointNetPlusPlusSegmentation(
    num_classes=NUM_CLASSES,
    input_features=INPUT_FEATURES
)


# ============================================================
# LOAD TRAINED MODEL ONCE
# ============================================================

model = load_model(
    model=model,
    checkpoint_path=CHECKPOINT_PATH,
    device=device
)


# ============================================================
# PROCESS ALL LIDAR FILES
# ============================================================

successful_files = 0

failed_files = 0


for index, file_path in enumerate(
    lidar_files,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"PROCESSING FRAME "
        f"[{index}/{len(lidar_files)}]"
    )

    print("-" * 60)


    print(
        f"\nFile:\n"
        f"{file_path.name}"
    )


    try:

        # ====================================================
        # LOAD RAW LIDAR
        # ====================================================

        raw_points = load_lidar_file(
            file_path
        )


        print(
            f"\nRaw point cloud shape: "
            f"{raw_points.shape}"
        )


        # ====================================================
        # PREPROCESS
        # ====================================================

        processed_points = preprocess_lidar(
            raw_points,
            num_points=NUM_POINTS
        )


        # ====================================================
        # PREDICT
        # ====================================================

        predictions = predict(
            model=model,
            points=processed_points,
            device=device
        )


        # ====================================================
        # CREATE OUTPUT FILE NAME
        # ====================================================

        # Remove only ".bin"
        base_name = file_path.name

        if base_name.endswith(".pcd.bin"):

            base_name = base_name[:-8]


        output_path = (
            PREDICTION_DIR /
            f"{base_name}_prediction.npz"
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        np.savez_compressed(
            output_path,
            points=processed_points,
            labels=predictions,
            source_file=file_path.name
        )


        print(
            f"\n✅ Prediction saved:"
        )

        print(
            output_path.name
        )


        successful_files += 1


    except Exception as error:

        print(
            f"\n❌ ERROR processing:"
        )

        print(
            file_path.name
        )

        print(
            f"Error: {error}"
        )


        failed_files += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("BATCH INFERENCE COMPLETED")

print("=" * 60)


print(
    f"\nTotal files: "
    f"{len(lidar_files)}"
)


print(
    f"Successfully processed: "
    f"{successful_files}"
)


print(
    f"Failed: "
    f"{failed_files}"
)


print(
    f"\nPredictions saved in:"
)


print(
    PREDICTION_DIR
)


print("\n🎉 PIPELINE COMPLETED!")

print("=" * 60)