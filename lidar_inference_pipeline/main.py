from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# IMPORT PROJECT MODULES
# ============================================================
from preprocessing.lidar_loader import load_lidar_file
from preprocessing.preprocess import preprocess_lidar
from models.pointnet_model import PointNetPlusPlusSegmentation
from inference.predict import load_model, predict_batch, print_prediction_summary

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 8
NUM_POINTS = 32768
NUM_CLASSES = 3
INPUT_FEATURES = 2
MAX_FRAMES = None

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_weighted_pointnet_model.pth"
PREDICTION_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET CLASS FOR MULTIPROCESSING
# ============================================================
class LidarDataset(Dataset):
    def __init__(self, files, num_points):
        self.files = files
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            raw_points = load_lidar_file(file_path)
            real_points, model_points = preprocess_lidar(raw_points, num_points=self.num_points)
            return real_points, model_points, file_path.name, True
        except Exception as e:
            # Return dummy data on failure to not crash the dataloader worker
            dummy = np.zeros((self.num_points, 5), dtype=np.float32)
            return dummy, dummy, file_path.name, False

def main():
    # ============================================================
    # DEVICE
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("LIDAR BATCH INFERENCE PIPELINE (OPTIMIZED)")
    print("=" * 65)
    print(f"\nDevice: {device}")
    print(f"Number of points per frame: {NUM_POINTS}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    if MAX_FRAMES is None:
        print("Frame limit: ALL FRAMES")
    else:
        print(f"Maximum frames to process: {MAX_FRAMES}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"\nModel checkpoint not found:\n{CHECKPOINT_PATH}")
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"\nRaw data directory not found:\n{RAW_DATA_DIR}")

    all_lidar_files = sorted(RAW_DATA_DIR.glob("*.pcd.bin"))
    if len(all_lidar_files) == 0:
        raise FileNotFoundError(f"\nNo .pcd.bin files found in:\n{RAW_DATA_DIR}")

    if MAX_FRAMES is None:
        lidar_files = all_lidar_files
    else:
        lidar_files = all_lidar_files[:MAX_FRAMES]

    print(f"\nTotal raw LiDAR files found: {len(all_lidar_files)}")
    print(f"Frames selected for processing: {len(lidar_files)}")

    print("\nCreating PointNet++ model...")
    model = PointNetPlusPlusSegmentation(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)

    print("\nLoading trained model...")
    model = load_model(model=model, checkpoint_path=CHECKPOINT_PATH, device=device)

    # Dataset and DataLoader
    dataset = LidarDataset(lidar_files, num_points=NUM_POINTS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, drop_last=False)

    successful_files = 0
    failed_files = 0

    total_batches = len(loader)
    
    for batch_idx, (real_points_batch, model_points_batch, file_names, success_flags) in enumerate(loader, start=1):
        print("\n" + "-" * 65)
        print(f"PROCESSING BATCH [{batch_idx}/{total_batches}]")
        print("-" * 65)

        # Filter out failed files from the batch
        valid_indices = [i for i, success in enumerate(success_flags) if success]
        
        for i, success in enumerate(success_flags):
            if not success:
                print(f"? ERROR PROCESSING FILE: {file_names[i]}")
                failed_files += 1

        if not valid_indices:
            continue

        # Get only the successful frames for model inference
        valid_model_points = model_points_batch[valid_indices]
        
        print(f"\nRunning PointNet++ prediction on {len(valid_indices)} frames...")
        
        try:
            # Predict
            predictions_batch = predict_batch(model=model, points_tensor=valid_model_points, device=device)
            
            # Save results
            for local_idx, batch_idx_orig in enumerate(valid_indices):
                real_p = real_points_batch[batch_idx_orig].numpy()
                model_p = model_points_batch[batch_idx_orig].numpy()
                pred = predictions_batch[local_idx]
                fname = file_names[batch_idx_orig]
                
                base_name = fname[:-8] if fname.endswith(".pcd.bin") else fname
                output_path = PREDICTION_DIR / f"{base_name}_prediction.npz"
                
                np.savez_compressed(
                    output_path,
                    points=real_p,
                    model_points=model_p,
                    labels=pred,
                    source_file=fname
                )
                
                print(f"? Prediction saved: {output_path.name}")
                successful_files += 1

        except Exception as e:
            print(f"? BATCH PREDICTION ERROR: {e}")
            failed_files += len(valid_indices)

    print("\n" + "=" * 65)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 65)
    print(f"\nTotal raw files found: {len(all_lidar_files)}")
    print(f"Frames selected: {len(lidar_files)}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"\nPredictions saved at:\n{PREDICTION_DIR}")
    print("\n?? ALL FRAME PROCESSING COMPLETED!")
    print("=" * 65)

if __name__ == "__main__":
    # Needed for multiprocessing in Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
