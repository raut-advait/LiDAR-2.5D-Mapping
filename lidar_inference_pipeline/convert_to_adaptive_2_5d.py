from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from conversion.adaptive_2_5d_converter import Adaptive2_5DConverter

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "adaptive_maps_2_5d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FRAMES = None  # Change this to limit the number of frames to convert (e.g. MAX_FRAMES = 10)

converter = Adaptive2_5DConverter(
    near_distance=15.0,
    medium_distance=35.0,
    far_distance=60.0,
    near_resolution=0.25,
    medium_resolution=0.50,
    far_resolution=1.00,
    near_max_layers=5,
    medium_max_layers=3,
    far_max_layers=2,
    height_variation_threshold=0.5
)

def process_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        if "points" not in data or "labels" not in data:
            raise KeyError("Missing required keys in prediction file.")
            
        points = data["points"]
        labels = data["labels"]
        
        adaptive_map = converter.convert(points=points, labels=labels)
            
        output_name = file_path.stem.replace("_prediction", "_adaptive_2_5d") + ".npz"
        output_path = OUTPUT_DIR / output_name
        
        cells_array = np.array(adaptive_map["cells"], dtype=object)
        
        np.savez_compressed(
            output_path,
            cells=cells_array,
            near_distance=np.array(adaptive_map["near_distance"], dtype=np.float32),
            medium_distance=np.array(adaptive_map["medium_distance"], dtype=np.float32),
            far_distance=np.array(adaptive_map["far_distance"], dtype=np.float32),
            near_resolution=np.array(0.25, dtype=np.float32),
            medium_resolution=np.array(0.50, dtype=np.float32),
            far_resolution=np.array(1.00, dtype=np.float32),
            total_cells=np.array(adaptive_map["total_cells"], dtype=np.int32),
            input_points=np.array(adaptive_map["input_points"], dtype=np.int32)
        )
        
        return (True, file_path.name, adaptive_map["total_cells"])
    except Exception as e:
        return (False, file_path.name, str(e))

def main():
    prediction_files = sorted(PREDICTIONS_DIR.glob("*.npz"))
    if len(prediction_files) == 0:
        raise FileNotFoundError(f"\nNo prediction files found in:\n{PREDICTIONS_DIR}")
        
    if MAX_FRAMES is not None:
        prediction_files = prediction_files[:MAX_FRAMES]
        
    print("\n" + "=" * 65)
    print("ADAPTIVE 3D TO 2.5D CONVERSION (VECTORIZED CPU ACCELERATED)")
    print("=" * 65)
    print(f"\nPrediction files found: {len(prediction_files)}")
    print("\nProcessing using ultra-fast Numpy vectorization with ThreadPoolExecutor...")
    
    successful_files = 0
    failed_files = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in prediction_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            success, fname, info = future.result()
            if success:
                print(f"[{i}/{len(prediction_files)}] ✅ Saved {fname} ({info} cells)")
                successful_files += 1
            else:
                print(f"[{i}/{len(prediction_files)}] ❌ ERROR on {fname}: {info}")
                failed_files += 1
                
    print("\n" + "=" * 65)
    print("ADAPTIVE CONVERSION COMPLETED")
    print("=" * 65)
    print(f"Total files: {len(prediction_files)}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"\nOutput directory:\n{OUTPUT_DIR}")
    print("\n🎉 ADAPTIVE 2.5D MAPPING PIPELINE COMPLETED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
