#!/usr/bin/env python3
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import glob,os

# --- camera intrinsics ---
fx, fy, cx, cy = 2813.643275, 2808.326079, 969.285772, 624.049972
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# --- lidar->camera extrinsics (from quaternion + translation) ---
q = [0.49757205900281865, -0.5191354403519034,
     -0.49750025670087406, 0.48519473951486275]  # [w, x, y, z]
t = np.array([-0.362996576109623, 0.060237225768389635, -0.18926884540339461])

# Convert quaternion to rotation matrix
rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # (x, y, z, w) order

# --- paths ---
lidar_dir = "/home/pinaka/dataset/rellis3d/Rellis-3D/lidar/00000/os1_cloud_node_kitti_bin"
img_dir   = "/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node"

img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
# pick first frame
img_path = img_files[10]
lidar_path = lidar_files[10]

img = cv2.imread(img_path)
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

# --- transform LiDAR → camera frame ---
pts_cam = (rot @ points[:, :3].T + t.reshape(3, 1)).T

# Keep only points in front of camera
mask = pts_cam[:, 2] > 0
pts_cam = pts_cam[mask]

# --- project to image plane ---
uv = (K @ pts_cam.T).T
u = uv[:, 0] / uv[:, 2]
v = uv[:, 1] / uv[:, 2]

# --- overlay points ---
h, w, _ = img.shape
for x, y in zip(u, v):
    if 0 <= int(x) < w and 0 <= int(y) < h:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)

cv2.imshow("LiDAR-Camera Fusion", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
