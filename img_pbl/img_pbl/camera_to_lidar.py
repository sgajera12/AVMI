#!/usr/bin/env python3
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import open3d as o3d


# 1. Paths
BASE = Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ  = "00000"

calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"

# 2. Load Camera->LiDAR transform and invert
with open(calib_file, "r") as f:
    tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]

q, t = tf["q"], tf["t"]
R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)

# Invert to get LiDAR -> Camera
R_l2c = R_c2l.T
t_l2c = -R_c2l.T @ t_c2l

# 3. Load camera intrinsics
fx, fy, cx, cy = np.loadtxt(intr_file)
K = np.array([[fx, 0,  cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float32)

# 4. Load LiDAR + Image
bins = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
imgs = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])
assert bins and imgs, "Missing .bin or .jpg/.png files"

frame_id = 254   # select the frame you want
lidar_file = bins[frame_id]
image_file = imgs[frame_id]

pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[:, :3]  # Nx3 LiDAR points
img = cv2.imread(str(image_file))
if img is None:
    raise FileNotFoundError(f"Could not read {image_file}")
h, w = img.shape[:2]

# 5. Transform LiDAR -> Camera
Pc = (R_l2c @ pts.T + t_l2c.reshape(3, 1)).T
Z = Pc[:, 2]
front = Z > 0
Pc = Pc[front]
pts_front = pts[front]
Z = Z[front]

# 6. Project to image plane
uv = (K @ (Pc.T / Z)).T
u, v = uv[:, 0], uv[:, 1]
inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)

u = u[inside].astype(np.int32)
v = v[inside].astype(np.int32)
pts_visible = pts_front[inside]

# 7. Sample image colors
colors_bgr = img[v, u]           # (N,3)
colors_rgb = colors_bgr[:, ::-1] # convert BGR -> RGB

# 8. Save colored point cloud as PLY
colored_pc = np.hstack((pts_visible, colors_rgb.astype(np.float32)))

ply_header = f'''ply
format ascii 1.0
element vertex {colored_pc.shape[0]}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

ply_file = f"ply/colored_lidar_{frame_id:06d}.ply"
with open(ply_file, "w") as f:
    f.write(ply_header)
    for p, c in zip(pts_visible, colors_rgb):
        f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


print(f"Saved colored point cloud to {ply_file}")
pcd = o3d.io.read_point_cloud(ply_file)
o3d.visualization.draw_geometries([pcd])
