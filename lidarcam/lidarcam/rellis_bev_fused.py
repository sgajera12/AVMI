#!/usr/bin/env python3
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------
BASE = Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ  = "00000"
FRAME_ID = 200   # choose a frame index

calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"

OUT_DIR = Path("./bev_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Load calibration
# -----------------------------
with open(calib_file, "r") as f:
    tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]

q, t = tf["q"], tf["t"]
R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)

# Invert to get LiDAR -> Camera
R_l2c = R_c2l.T
t_l2c = -R_c2l.T @ t_c2l

fx, fy, cx, cy = np.loadtxt(intr_file)
K = np.array([[fx, 0,  cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float32)

print("Loaded intrinsics:", fx, fy, cx, cy)

# -----------------------------
# 2) Load one LiDAR frame + image
# -----------------------------
bins  = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
imgs  = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])

assert bins and imgs, "No LiDAR or image files found."

lidar_file = bins[FRAME_ID]
image_file = imgs[FRAME_ID]
print("Using:", lidar_file.name, image_file.name)

pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)  # (N,4)
xyz = pts[:, :3]  # ignore intensity for now
img = cv2.imread(str(image_file))
if img is None:
    raise FileNotFoundError(f"Could not read {image_file}")
h, w = img.shape[:2]

# -----------------------------
# 3) Project LiDAR -> Camera, sample colors
# -----------------------------
Pc = (R_l2c @ xyz.T + t_l2c.reshape(3, 1)).T   # (N,3) in camera frame
Z = Pc[:, 2]
front = Z > 0.0
Pc = Pc[front]
Z = Z[front]
xyz_front = xyz[front]

uv = (K @ (Pc.T / Z)).T   # (N,3) homogeneous
u, v = uv[:, 0], uv[:, 1]

inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
u = u[inside].astype(np.int32)
v = v[inside].astype(np.int32)
xyz_vis = xyz_front[inside]
# Convert RELLIS LiDAR axes (x=right, y=forward, z=up) → KITTI style (x=forward, y=left, z=up)
xyz_vis = np.stack([xyz_vis[:, 1], -xyz_vis[:, 0], xyz_vis[:, 2]], axis=1)


colors_bgr = img[v, u, :]          # (N,3)
colors_rgb = colors_bgr[:, ::-1]   # BGR -> RGB

print(f"Visible LiDAR points in image: {xyz_vis.shape[0]}")

# -----------------------------
# 4) Define BEV grid (LiDAR XY plane)
# -----------------------------
# RELLIS OS1 is roughly KITTI-like: x forward, y left, z up
# Choose a region in front of the vehicle
# X_MIN, X_MAX = 0.0, 50.0     # meters forward
# Y_MIN, Y_MAX = -25.0, 25.0   # meters left/right
X_MIN, X_MAX = -10.0, 20.0     # forward/back (meters)
Y_MIN, Y_MAX = 0.0, 50.0       # sideways (right side)

RES = 0.1                    # meters per pixel (0.1m = 10cm)

bev_H = int((X_MAX - X_MIN) / RES)
bev_W = int((Y_MAX - Y_MIN) / RES)

print(f"BEV grid size: {bev_H} x {bev_W}")

plt.scatter(xyz_vis[:, 0], xyz_vis[:, 1], s=1, c=xyz_vis[:, 2], cmap='jet')
plt.xlabel("X (forward)")
plt.ylabel("Y (left)")
plt.title("LiDAR points in XY after conversion")
plt.axis("equal")
plt.show()
# We'll build:
# - RGB BEV map (fused color)
# - height map (max Z)
bev_rgb   = np.zeros((bev_H, bev_W, 3), dtype=np.float32)
bev_count = np.zeros((bev_H, bev_W), dtype=np.int32)
bev_height = np.full((bev_H, bev_W), -np.inf, dtype=np.float32)

# -----------------------------
# 5) Fill BEV grid with colored points
# -----------------------------
# Use original LiDAR coordinates xyz_vis for XY/Z; colors_rgb for appearance.
print("XYZ range after conversion:",
      np.min(xyz_vis, axis=0),
      np.max(xyz_vis, axis=0))

X = xyz_vis[:, 0]
Y = xyz_vis[:, 1]
Z = xyz_vis[:, 2]

# Filter to BEV bounds
mask = (X >= X_MIN) & (X < X_MAX) & (Y >= Y_MIN) & (Y < Y_MAX)
X = X[mask]
Y = Y[mask]
Z = Z[mask]
C = colors_rgb[mask]

# Map each point to BEV indices
ix = ((X - X_MIN) / RES).astype(np.int32)  # row index
iy = ((Y - Y_MIN) / RES).astype(np.int32)  # col index

# Safety clip
ix = np.clip(ix, 0, bev_H - 1)
iy = np.clip(iy, 0, bev_W - 1)

for x_idx, y_idx, z_val, col in zip(ix, iy, Z, C):
    bev_rgb[x_idx, y_idx] += col
    bev_count[x_idx, y_idx] += 1
    if z_val > bev_height[x_idx, y_idx]:
        bev_height[x_idx, y_idx] = z_val

# Avoid division by zero
mask_nonzero = bev_count > 0
bev_rgb[mask_nonzero] /= bev_count[mask_nonzero][..., None]

# Fill empty cells with dark gray or black
bev_rgb[~mask_nonzero] = np.array([0, 0, 0], dtype=np.float32)

# Convert to uint8 image
bev_rgb_img = bev_rgb.astype(np.uint8)

# Optionally flip vertically so that "forward X" is at the top
bev_rgb_img_flipped = np.flipud(bev_rgb_img)

out_bev_path = OUT_DIR / f"bev_fused_frame{FRAME_ID:06d}.png"
cv2.imwrite(str(out_bev_path), cv2.cvtColor(bev_rgb_img_flipped, cv2.COLOR_RGB2BGR))
print(f"Saved fused BEV RGB map to {out_bev_path}")
