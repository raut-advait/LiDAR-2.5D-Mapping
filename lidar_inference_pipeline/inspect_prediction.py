import numpy as np


# ============================================================
# PREDICTION FILE PATH
# ============================================================

FILE_PATH = (
    r"D:\final project\lidar_inference_pipeline\data\processed"
    r"\predictions"
    r"\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590_prediction.npz"
)


# ============================================================
# LOAD FILE
# ============================================================

data = np.load(
    FILE_PATH,
    allow_pickle=True
)


print("\n" + "=" * 60)
print("PREDICTION FILE INSPECTION")
print("=" * 60)


# ============================================================
# SHOW KEYS
# ============================================================

print("\nAvailable arrays:")

for key in data.files:
    print(f"  - {key}")


# ============================================================
# SHOW DATA DETAILS
# ============================================================

print("\n" + "=" * 60)
print("ARRAY DETAILS")
print("=" * 60)


for key in data.files:

    array = data[key]

    print(f"\nKey: {key}")
    print(f"Shape: {array.shape}")
    print(f"Dimensions: {array.ndim}")
    print(f"Data type: {array.dtype}")

    print("\nValue / First values:")

    # --------------------------------------------------------
    # HANDLE SCALAR VALUES
    # --------------------------------------------------------

    if array.ndim == 0:

        print(array.item())

    else:

        print(array[:5])


print("\n" + "=" * 60)
print("INSPECTION COMPLETED!")
print("=" * 60)

# iska kam hai bus data samjna aur verify krr na joo data pipline se output me aaya hai vo
