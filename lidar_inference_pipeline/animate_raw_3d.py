from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

# ============================================================
# CONFIGURATION
# ============================================================

FPS = 10
INTERVAL = 100   # milliseconds
MAX_FRAMES = 100 # Adjust to limit frame count for faster rendering
MAX_POINTS_PER_FRAME = 15000  # Downsample for faster matplotlib rendering
SAVE_VIDEO = True
VIDEO_NAME = "adaptive_raw_3d.gif" # Keeping the same name as requested

# Set limits for the 3D visualization
X_LIM = (-50, 50)
Y_LIM = (-50, 50)
Z_LIM = (-5, 10)

SEMANTIC_COLORS = {
    0: "gray",
    1: "black",
    2: "red"
}

# ============================================================
# SETUP DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
# Load PREDICTIONS to get the labels corresponding to the 2.5D colors
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND FILES
# ============================================================

pred_files = sorted(PREDICTIONS_DIR.glob("*.npz"))
if len(pred_files) == 0:
    raise FileNotFoundError(f"No prediction files found in {PREDICTIONS_DIR}")

if MAX_FRAMES is not None:
    pred_files = pred_files[:MAX_FRAMES]

print("\n" + "=" * 65)
print("3D SEMANTIC LIDAR ANIMATION (MATPLOTLIB)")
print("=" * 65)
print(f"Frames to render: {len(pred_files)}")
print(f"Downsampling to: {MAX_POINTS_PER_FRAME} points/frame")

# ============================================================
# SETUP PLOT
# ============================================================

# Use white background theme
plt.style.use('default')

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

ax.set_title("3D Semantic Point Cloud (Matched to 2.5D)", fontsize=16, color='black', pad=20)
ax.set_xlim(X_LIM)
ax.set_ylim(Y_LIM)
ax.set_zlim(Z_LIM)
ax.set_xlabel('X (m)', color='black')
ax.set_ylabel('Y (m)', color='black')
ax.set_zlabel('Z Elevation (m)', color='black')
ax.tick_params(colors='black')

# Configure viewing angle
ax.view_init(elev=35, azim=-45)

# Initialize an empty scatter plot
scatter = ax.scatter([], [], [], c=[], s=0.5, alpha=0.8)

# Add legend matching 2.5D
legend_elements = [
    Patch(facecolor="gray", label="Drivable Surface (Class 0)"),
    Patch(facecolor="black", label="Non-Drivable / Static (Class 1)"),
    Patch(facecolor="red", label="Dynamic Object (Class 2)")
]
ax.legend(handles=legend_elements, loc="upper right", facecolor='white', framealpha=0.9)

# ============================================================
# UPDATE FUNCTION FOR ANIMATION
# ============================================================

def update(frame_idx):
    file_path = pred_files[frame_idx]
    
    # Load predictions data
    try:
        data = np.load(file_path, allow_pickle=True)
        points = data["points"]
        labels = data["labels"]
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return scatter,
        
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Filter points within bounds
    mask = (x >= X_LIM[0]) & (x <= X_LIM[1]) & \
           (y >= Y_LIM[0]) & (y <= Y_LIM[1]) & \
           (z >= Z_LIM[0]) & (z <= Z_LIM[1])
           
    x = x[mask]
    y = y[mask]
    z = z[mask]
    labels = labels[mask]
    
    # Downsample for matplotlib performance
    if len(x) > MAX_POINTS_PER_FRAME:
        # Randomly select points
        indices = np.random.choice(len(x), size=MAX_POINTS_PER_FRAME, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        labels = labels[indices]
        
    # Map labels to color strings, then to rgba or just pass the color array
    # Matplotlib scatter accepts an array of color strings
    colors = [SEMANTIC_COLORS.get(int(l), "black") for l in labels]
        
    # Update scatter offsets
    scatter._offsets3d = (x, y, z)
    # Update colors
    scatter.set_color(colors)
    
    print(f"Rendering frame [{frame_idx + 1}/{len(pred_files)}]: {file_path.name} ({len(x)} points)")
    
    return scatter,

# ============================================================
# GENERATE ANIMATION
# ============================================================

print("\nStarting rendering process...")
anim = FuncAnimation(fig, update, frames=len(pred_files), interval=INTERVAL, blit=False)

if SAVE_VIDEO:
    output_path = OUTPUT_DIR / VIDEO_NAME
    print(f"\nSaving GIF to: {output_path}")
    print("This may take several minutes depending on the frame count...")
    
    try:
        anim.save(output_path, writer='pillow', fps=FPS, dpi=100)
        print("\n✅ 3D ANIMATION SAVED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ FAILED TO SAVE ANIMATION: {e}")

print("\n🎉 ALL DONE!")
