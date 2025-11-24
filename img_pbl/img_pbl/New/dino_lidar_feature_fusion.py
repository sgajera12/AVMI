#!/usr/bin/env python3
import numpy as np
import cv2, yaml, os
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R

# ===================== USER SETTINGS =====================
BASE     = Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ      = "00000"
FRAME_ID = 250  # <-- change frame here

# DINO feature map saved earlier: expected shape (H, W, C)
# (use the extractor we wrote; file name is up to you)
FEAT_PATH = Path("./out_dino_features/dino_feat_frame000250-1581624677_749.npy")

OUT_DIR  = Path("./fusion_out"); OUT_DIR.mkdir(exist_ok=True, parents=True)
NUM_CLUSTERS = 6  # try 4..8
# =========================================================

# ---------- Paths (follow your existing layout) ----------
calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"

# ---------- Load extrinsics: camera -> lidar, then invert ----------
with open(calib_file, "r") as f:
    tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]  # from your scripts
q, t = tf["q"], tf["t"]
R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)

# Invert (LiDAR <- Camera)^T to get LiDAR -> Camera (same as your code)
R_l2c = R_c2l.T
t_l2c = -R_c2l.T @ t_c2l
# (ref: you compute exactly this in both scripts). :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

# ---------- Load intrinsics ----------
fx, fy, cx, cy = np.loadtxt(intr_file)
K = np.array([[fx, 0,  cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float32)  # :contentReference[oaicite:5]{index=5}

# ---------- Pick matching LiDAR and RGB frame ----------
bins = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
imgs = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])
assert bins and imgs, "Missing .bin or images"
lidar_file = bins[FRAME_ID]
image_file = imgs[FRAME_ID]

pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[:, :3]  # (N,3) lidar
img = cv2.imread(str(image_file))
if img is None:
    raise FileNotFoundError(image_file)
H, W = img.shape[:2]

# ---------- Load DINO feature map ----------
feat = np.load(FEAT_PATH)  # expect (H, W, C)
if feat.ndim != 3 or feat.shape[0] != H or feat.shape[1] != W:
    raise ValueError(f"Feature map shape {feat.shape} must match image (H,W,·)=({H},{W},·)")

C = feat.shape[2]
# Optional: L2-normalize features (helps KMeans/DBSCAN)
feat = feat.astype(np.float32)
norm = np.linalg.norm(feat, axis=2, keepdims=True) + 1e-6
feat = feat / norm

# ---------- LiDAR -> Camera, keep front points ----------
Pc = (R_l2c @ pts.T + t_l2c.reshape(3,1)).T
Z = Pc[:, 2]
front = Z > 0
Pc = Pc[front]
pts_front = pts[front]
Z = Z[front]

# ---------- Project to image plane ----------
uv = (K @ (Pc.T / Z)).T
u, v = uv[:, 0], uv[:, 1]
inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

u = u[inside].astype(np.int32)
v = v[inside].astype(np.int32)
pts_vis = pts_front[inside]
Pc_vis  = Pc[inside]

print(f"LiDAR points: {pts.shape[0]} | front: {front.sum()} | on image: {u.shape[0]}")

# ---------- Sample DINO feature for each projected point ----------
pix_feat = feat[v, u]   # (M, C) — fast fancy indexing

# ---------- Build joint vector [x, y, z, feature…] ----------
X = np.hstack([Pc_vis, pix_feat])   # (M, 3 + C)

# (Optional) standardize only the 3D coords to balance with features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = np.hstack([scaler.fit_transform(Pc_vis), pix_feat])  # leave features normalized

# ---------- Cluster ----------
print(f"Clustering {X_scaled.shape[0]} points in (x,y,z,feat{C}) space with K={NUM_CLUSTERS}…")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=5)
labels = kmeans.fit_predict(X_scaled)  # (M,)

# ---------- Overlay on image ----------
colors = np.random.randint(0, 255, (NUM_CLUSTERS, 3), dtype=np.uint8)
overlay = img.copy()
for (x, y, lab) in zip(u, v, labels):
    c = tuple(int(v) for v in colors[lab])
    cv2.circle(overlay, (x, y), 2, c, -1)

alpha = 0.5
blended = cv2.addWeighted(img, 1.0, overlay, alpha, 0)
out_img = OUT_DIR / f"overlay_{FRAME_ID:06d}.png"
cv2.imwrite(str(out_img), blended)
print(f"Saved overlay → {out_img}")

# ---------- Save colored point cloud ----------
rgb = colors[labels]  # (M,3) BGR
rgb = rgb[:, ::-1]    # to RGB for PLY
ply_file = OUT_DIR / f"clustered_points_{FRAME_ID:06d}.ply"
with open(ply_file, "w") as f:
    f.write(f"ply\nformat ascii 1.0\n")
    f.write(f"element vertex {pts_vis.shape[0]}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(pts_vis, rgb):
        f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
print(f"Saved colored point cloud → {ply_file}")
