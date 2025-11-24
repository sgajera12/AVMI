#!/usr/bin/env python3
"""
Full 360° BEV for Rellis-3D  (2D + 2.5D version)
-------------------------------------------------
• Uses Rellis-3D calibration (transforms.yaml + camera_info.txt)
• Generates a fused color BEV for visualization
• Also outputs a 2.5D numeric grid with channels:
      height (Zmax), intensity, density, RGB
-------------------------------------------------
"""

import numpy as np
import cv2, yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

#configurations
BASE= Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ= "00000"
FRAME_ID=250# change this to switch frame
OUT_DIR=Path("./bev_out_2_5D")
OUT_DIR.mkdir(parents=True, exist_ok=True)
#as BEV range (in vehicle frame, meters)
#X = forward, Y = left
X_MIN,X_MAX=-30.0,30.0
Y_MIN,Y_MAX=-30.0,30.0
RES=0.1# meters per pixel

#now we load the calibration and all rewuired files
calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"

with open(calib_file, "r") as f:
    tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]

q, t = tf["q"], tf["t"]
R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)
R_l2c = R_c2l.T
t_l2c = -R_c2l.T @ t_c2l

fx, fy, cx, cy = np.loadtxt(intr_file)
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], np.float32)

#loading LiDAR & image
bins = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
imgs = sorted([p for p in image_dir.iterdir()
               if p.suffix.lower() in [".jpg", ".png"]])
assert bins and imgs, "No LiDAR or image files found."

lidar_file = bins[FRAME_ID]
image_file = imgs[FRAME_ID]
print(f"Using: {lidar_file.name} & {image_file.name}")

pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
xyz_lidar = pts[:, :3]
intensity = pts[:, 3]
img = cv2.imread(str(image_file))
h, w = img.shape[:2]

#LiDAR to Camera projection for colorization
Pc = (R_l2c @ xyz_lidar.T + t_l2c.reshape(3, 1)).T
Z = Pc[:, 2]
front = Z > 0.0
uv = (K @ (Pc.T / np.maximum(Z, 1e-6))).T
u, v = uv[:, 0], uv[:, 1]
inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
cam_visible = front & inside

colors_cam = np.zeros((xyz_lidar.shape[0], 3), np.uint8)
if np.any(cam_visible):
    u_vis, v_vis = u[cam_visible].astype(int), v[cam_visible].astype(int)
    colors_cam[cam_visible] = img[v_vis, u_vis, :]

#intensity to grayscale color
inten = intensity.copy()
inten -= inten.min()
inten /= (inten.max() + 1e-6)
inten_u8 = (inten * 255).astype(np.uint8)
lidar_colors = cv2.applyColorMap(inten_u8, cv2.COLORMAP_BONE)[:, 0, :]
final_colors = lidar_colors.copy()
final_colors[cam_visible] = colors_cam[cam_visible]

#convert to vehicle frame
xyz_vehicle = np.stack([xyz_lidar[:, 1], -xyz_lidar[:, 0],xyz_lidar[:, 2]], 1)
X = xyz_vehicle[:, 0]
Y = xyz_vehicle[:, 1]
Z = xyz_vehicle[:, 2]

#create BEV grid (visual + numeric)
bev_H= int((X_MAX - X_MIN) / RES)
bev_W= int((Y_MAX - Y_MIN) / RES)
bev_rgb= np.zeros((bev_H, bev_W, 3), np.float32)
bev_height= np.full((bev_H, bev_W), -np.inf, np.float32)
bev_inten= np.zeros((bev_H, bev_W), np.float32)
bev_density= np.zeros((bev_H, bev_W), np.int32)

mask=(X >= X_MIN)&(X < X_MAX)&(Y >= Y_MIN)&(Y < Y_MAX)
Xb,Yb,Zb,Cb,Ib = X[mask],Y[mask],Z[mask],final_colors[mask],intensity[mask]

ix = ((Xb - X_MIN) / RES).astype(int)
iy = ((Yb - Y_MIN) / RES).astype(int)
ix, iy = np.clip(ix, 0, bev_H - 1), np.clip(iy, 0, bev_W - 1)

for x_i, y_i, z, col, inten in zip(ix, iy, Zb, Cb, Ib):
    bev_rgb[x_i, y_i] += col.astype(np.float32)
    bev_height[x_i, y_i] = max(bev_height[x_i, y_i], z)
    bev_inten[x_i, y_i] += inten
    bev_density[x_i, y_i] += 1

nz = bev_density > 0
bev_rgb[nz] /= bev_density[nz][..., None]
bev_inten[nz] /= bev_density[nz]
bev_height[~nz] = np.nan

#visual BEV image(flip+rotate)
bev_rgb[~nz] = 0
# bev_img =np.flipud(bev_rgb.astype(np.uint8))
bev_img = bev_rgb.astype(np.uint8)
bev_img =cv2.rotate(bev_img, cv2.ROTATE_90_CLOCKWISE)

out_img=OUT_DIR / f"bev_fused_full360_frame{FRAME_ID:06d}.png"
cv2.imwrite(str(out_img), bev_img)
print(f"Saved visual BEV = {out_img}")

#save numeric 2.5D grid(.npz)
bev_tensor = np.stack([np.nan_to_num(bev_height, nan=0.0),bev_inten,np.log1p(bev_density.astype(np.float32))], axis=-1)

out_npz = OUT_DIR / f"bev_tensor_frame{FRAME_ID:06d}.npz"
np.savez_compressed(out_npz, height=bev_height, intensity=bev_inten,density=bev_density, rgb=bev_rgb, tensor=bev_tensor)
print(f"Saved 2.5D BEV tensor = {out_npz}")

#quick display
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
plt.title("Full 360° BEV (Camera+LiDAR colorized)")
plt.axis("off")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(bev_height, cmap='terrain')
plt.title("BEV Height Channel (m)")
plt.colorbar()
plt.show()

print("\nChannels stored in .npz = ['height', 'intensity', 'density', 'rgb', 'tensor']")
