# 2.5D LiDAR Mapping Project - Commands

## 1. Open PowerShell / Terminal

Go to your new copied project folder:

```powershell
cd "D:\PATH\TO\lidar_2_5d_mapping_project"
```

Replace the path above with your actual project path.

---

## 2. Check Python

```powershell
python --version
```

---

## 3. Check PyTorch

```powershell
python -c "import torch; print(torch.__version__)"
```

---

## 4. Check CUDA / GPU

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5. Run the existing inference pipeline

```powershell
python main.py
```

This will:
- Load raw LiDAR data
- Preprocess the point cloud
- Load the trained PointNet++ model
- Predict semantic classes
- Save prediction output

---

## 6. Expected project structure

```text
lidar_2_5d_mapping_project/
│
├── checkpoints/
├── data/
│   ├── raw/
│   └── processed/
│       └── predictions/
├── inference/
├── models/
├── preprocessing/
└── main.py
```

---

# NEXT STEPS (TO BE ADDED)

## 7. 3D to 2.5D Conversion

Future command:

```powershell
python conversion/run_conversion.py
```

---

## 8. Visualization

Future command:

```powershell
python visualization/visualize_map.py
```

---

## 9. Adaptive Resolution Mapping

Future command:

```powershell
python mapping/adaptive_mapping.py
```


////


python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install numpy matplotlib

pip install torch torchvision torchaudio

pip install open3d tqdm pandas

pip freeze > requirements.txt

python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"