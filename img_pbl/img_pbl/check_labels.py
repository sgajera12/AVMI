import numpy as np
from PIL import Image
from pathlib import Path

#Check label values in Rellis-3D
root = Path('/home/pinaka/dataset/rellis3d/Rellis-3D')
label_dir = root / 'image_id/00000/pylon_camera_node_label_id'

label_files = sorted(list(label_dir.glob('*.png')))[:10]  # Check first 10

print(f"Checking {len(label_files)} label files...\n")

all_values = set()
for label_file in label_files:
    mask = np.array(Image.open(label_file))
    unique_vals = np.unique(mask)
    all_values.update(unique_vals)
    print(f"{label_file.name}: min={mask.min()}, max={mask.max()}, unique={len(unique_vals)}")

print(f"\nAll unique values across files: {sorted(all_values)}")
print(f"Range: [{min(all_values)}, {max(all_values)}]")

if max(all_values) >= 20:
    print(f"\n WARNING: Labels exceed NUM_CLASSES=20!")
    print(f"Values >= 20: {sorted([v for v in all_values if v >= 20])}")
