#!/usr/bin/env python3
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os

#Paths
BASE = Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ  = "00000"

calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"
pose_file  = BASE / f"lidar_pos/{SEQ}/poses.txt"

# Load Camera to lidar extrinsic
with open(calib_file, "r") as f:tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]

q, t = tf["q"], tf["t"]
R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32) 
t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)
# Load camera intrinsics
fx, fy, cx, cy = np.loadtxt(intr_file)
K = np.array([[fx, 0,  cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float32)

print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
#Pick one frame (matching by index)
bins  = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
imgs  = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])
assert bins, "No .bin files found"
assert imgs, "No images found"

frame_id  = 200#file/frame num to process

lidar_file = bins[frame_id]
image_file = imgs[frame_id]
#load data
pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[:, :3]  # Nx3 in LiDAR frame
img = cv2.imread(str(image_file))
if img is None:
    raise FileNotFoundError(f"Could not read {image_file}")
h, w = img.shape[:2]\

#camera to lidar frame
R_l2c = R_c2l.T
t_l2c = -R_c2l.T @ t_c2l
Pc = (R_l2c @ pts.T + t_l2c.reshape(3, 1)).T

#keep only points in front of camera
Z = Pc[:, 2]
front = Z > 0
Pc = Pc[front]
Z  = Z[front]

#project to image plane
uv = (K @ (Pc.T / Z)).T
u, v = uv[:, 0], uv[:, 1]
inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
u = u[inside].astype(np.int32)
v = v[inside].astype(np.int32)
Z = Z[inside]

print(f"Projected {pts.shape[0]} LiDAR points, "f"{front.sum()} in front, {inside.sum()} on image.")
#normalize depth to [0, 255]
Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
Z_color = (Z_norm * 255).astype(np.uint8)

#Convert depth to colormap
colors = cv2.applyColorMap(Z_color, cv2.COLORMAP_JET)

overlay = img.copy()
for (x, y, c) in zip(u, v, colors):
    color_tuple = tuple(int(v) for v in c[0])  #to convert from array to tuple (B,G,R)
    cv2.circle(overlay, (x, y), 1, color_tuple, -1)
cv2.imwrite(f"image/overlay_{frame_id:06d}.png", overlay)

#LiDAR-only depth view
depth = np.zeros((h, w), np.float32)
depth[v, u] = Z
if depth.max() > 0:
    depth /= depth.max()
depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(f"image/lidar_only_depth_{frame_id:06d}.png", depth_colored)

print(f"Saved overlay_{frame_id:06d}.png and lidar_only_depth_{frame_id:06d}.png")
