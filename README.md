# Adaptive 2.5D LiDAR Mapping & Real-time Semantic Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

An advanced pipeline for converting raw 3D LiDAR point clouds into highly optimized, adaptive 2.5D maps. This system is designed to provide foveated, adaptive-resolution mapping that mimics how human vision allocates detail—dense near the vehicle where collision risk is highest, and sparse far away where full precision isn't needed.

---

## 🌟 Proposed Solution & Innovation
- **The Problem:** Full 3D voxel maps are extremely precise but memory and compute-heavy `O(N³)`. Flat 2D grids are efficient but lose height data needed for curbs, potholes, and overhangs.
- **Our Solution:** Our adaptive 2.5D grid keeps semantic and elevation data while varying cell resolution by distance.
- **Innovation:** Features a from-scratch **PointNet++** implementation (set abstraction + feature propagation layers) trained end-to-end on real nuScenes LiDAR data, avoiding reliance on pretrained off-the-shelf models.

---

## 🛠️ Architecture & Pipeline Flow

**Technologies:** Python 3.8+, PyTorch (CUDA-enabled), NumPy (vectorized CPU ops), Matplotlib.
**Datasets:** KITTI (rule-based prototype track), nuScenes (deep learning training track).

### Pipeline Flow:
1. **Raw LiDAR Ingestion:** `.pcd.bin` files.
2. **Preprocessing:** Ego-vehicle point removal, downsampling, and normalization.
3. **Semantic Inference (PointNet++):** 3-Class Segmentation:
   - ⬜ **Gray (Class 0):** Drivable Surface / Terrain (Roads)
   - ⬛ **Black (Class 1):** Non-Drivable / Static Objects (Sidewalks, Buildings)
   - 🟥 **Red (Class 2):** Dynamic Objects (Vehicles, Pedestrians)
4. **Adaptive Radial Grid Projection:** Multi-ring resolution scheduling (Near 15m, Medium 35m, Far 60m).
5. **Real-Time Visualization Dashboard:** Outputs animated `.gif` renders of both the 3D semantic cloud and the top-down 2.5D map.

---

## 📊 3D vs Adaptive 2.5D: Benchmark & Feasibility

The full pipeline has been validated end-to-end on **404 real LiDAR frames with 0 failures**.

| Metric / Feature | Results & Validation |
| :--- | :--- |
| **Memory Efficiency** | **73.2% measured reduction** vs a sparse 3D voxel baseline, and **99.8% reduction** vs a theoretical dense 3D array baseline. |
| **Grid Engine Metrics** | Confirmed concentration of resolution near the vehicle: Near cells (2,532) · Medium cells (797) · Far cells (305). |
| **Model Accuracy** | PointNet++ training converges successfully, with best validation accuracy reaching **80.14% over 50 epochs**. |
| **Latency Challenges** | Measured pipeline latency is ~1409ms/frame (~0.7 FPS) on GPU, with PointNet++'s forward pass taking ~1341ms. (CPU-only: ~16.6s/frame). |

### Current Challenges & Strategies
- **Inference Latency:** Currently the biggest gap to real-time. We plan to optimize inference via batching, mixed precision, or falling back to a pretrained segmentation backbone (SalsaNext/Cylinder3D) if needed.
- **Zone Boundaries:** Alignment between adjacent resolution rings is an ongoing challenge; a true Patchwork-style zone model is on the roadmap.
- **Testing:** Fully held-out test-set evaluation is being finalized.

---

## 🌍 Impact and Benefits
Targeted towards Autonomous/semi-autonomous ground vehicle systems (e.g., DRDO/IDEX Smart Vehicles, agriculture, logistics, disaster response).

- **Safety:** Full-resolution perception exactly where collision risk is highest (near-field).
- **Computational / Economic:** 73.2%+ memory reduction enables perception on lighter, power-constrained onboard compute hardware.
- **Operational:** Lower memory footprint supports longer continuous operation without RAM saturation.
- **Extensibility:** The adaptive-resolution approach generalizes to any real-time 3D perception system.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vrajsavliya/SIH_final_project.git
cd SIH_final_project
```

### 2. Set Up Virtual Environment (CUDA recommended)
```bash
python -m venv .venv_cuda
.\.venv_cuda\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 Running the Pipeline

1. **Full Inference Pipeline:** `python lidar_inference_pipeline/main.py`
2. **Convert to Adaptive 2.5D Map:** `python lidar_inference_pipeline/convert_to_adaptive_2_5d.py`
3. **Visualization Dashboard:** `python lidar_inference_pipeline/animate_adaptive_map.py`
4. **3D Semantic Point Cloud Visualization:** `python lidar_inference_pipeline/animate_raw_3d.py`

---
## 📚 Research & References
- Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*
- *SalsaNext* and *Cylinder3D* (benchmark literature)
- *KITTI* and *nuScenes* datasets
