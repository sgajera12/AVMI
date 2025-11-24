#!/usr/bin/env python3
"""
make_visual_bev.py
Create a basic Visual-BEV map by:
- extracting DINOv2 features from the camera image (dense HxW x C map)
- projecting LiDAR to the image, sampling per-point features
- splatting per-point features into a BEV grid (x-y plane), averaging per cell
Saves:
- bev_out/visual_bev_feat.npy  (H_bev, W_bev, C)
- bev_out/visual_bev_rgb.png   (PCA->RGB visualization)
- bev_out/points_used.png      (LiDAR->image overlay sanity check)
"""

import os
import cv2
import yaml
import math
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA

# ----------------- User paths (match your earlier scripts) -----------------
BASE = Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ  = "00000"

calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"

# Choose a frame index you know exists in both folders
FRAME_ID = 200  # change as needed

# Output directory
OUTDIR = Path("./bev_out"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------- BEV grid config (adjust later as needed) ----------------
# Coordinate convention (LiDAR/vehicle frame):
#   x: forward (meters), y: left (meters).  (If your dataset uses y right, flip sign below.)
X_MIN, X_MAX = 0.0, 50.0      # forward 0..50 m
Y_MIN, Y_MAX = -20.0, 20.0    # left -20..20 m
RES = 0.25                    # meters per cell
# Derived sizes
H_bev = int(math.ceil((X_MAX - X_MIN) / RES))  # rows over x
W_bev = int(math.ceil((Y_MAX - Y_MIN) / RES))  # cols over y

# ----------------- Model config -----------------
# DINOv2 (base) dense features
VFM_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Utils -----------------
def load_calib():
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
    return K, R_l2c, t_l2c

def load_frame(idx):
    bins = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
    imgs = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])
    assert bins and imgs, "Missing .bin or .jpg/.png files"
    lidar_file = bins[idx]
    image_file = imgs[idx]
    pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)[:, :3]  # Nx3
    img = cv2.imread(str(image_file))
    if img is None:
        raise FileNotFoundError(f"Could not read image {image_file}")
    return pts, img, image_file.name

def get_dinov2_featmap(img_bgr):
    """
    Return dense feature map aligned to image size: (H, W, C=768)
    Implementation detail:
      - run DINOv2 on resized square input
      - discard CLS token, reshape tokens -> (h_patches, w_patches, C)
      - upsample each channel to (H, W)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    processor = AutoImageProcessor.from_pretrained(VFM_NAME)
    model = AutoModel.from_pretrained(VFM_NAME).to(DEVICE).eval()

    # Use a square multiple-of-14 size (e.g., 518), like we stabilized earlier
    target = 518
    img_sq = cv2.resize(img_rgb, (target, target), interpolation=cv2.INTER_AREA)

    with torch.no_grad():
        inputs = processor(images=img_sq, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)  # last_hidden_state: (1, 1+N, C)
        tokens = outputs.last_hidden_state[0, 1:, :]  # remove CLS, shape (N, C)

    tokens = tokens.detach().cpu().numpy()
    C = tokens.shape[-1]
    side = int(np.sqrt(tokens.shape[0]))  # e.g., 37x37 for 518/14
    feat_small = tokens.reshape(side, side, C)  # (h_p, w_p, C)

    # Upsample channel-wise to (H, W)
    feat_full = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        feat_full[:, :, c] = cv2.resize(feat_small[:, :, c], (W, H), interpolation=cv2.INTER_LINEAR)

    return feat_full  # (H, W, C)

def project_lidar_to_image(pts_lidar, K, R_l2c, t_l2c, img_shape):
    """ Return pixel coords and mask of in-FOV points; also camera-space Z for depth. """
    H, W = img_shape[:2]
    Pc = (R_l2c @ pts_lidar.T + t_l2c.reshape(3, 1)).T  # Nx3 in camera coords
    Z = Pc[:, 2]
    front = Z > 0
    Pc = Pc[front]
    Z = Z[front]

    uv = (K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u = u[inside].astype(np.int32)
    v = v[inside].astype(np.int32)
    Z = Z[inside]
    pts_vis = pts_lidar[front][inside]

    return u, v, Z, pts_vis

def splat_to_bev(points_lidar, feats_per_point, x_min, x_max, y_min, y_max, res, agg="mean"):
    """
    points_lidar: (N,3) in LiDAR/vehicle frame
    feats_per_point: (N, C) sampled from image feature map
    Returns:
      bev_feat: (H_bev, W_bev, C)
      bev_count: (H_bev, W_bev) number of points per cell
    """
    C = feats_per_point.shape[1]
    H_bev = int(math.ceil((x_max - x_min) / res))
    W_bev = int(math.ceil((y_max - y_min) / res))
    bev_feat = np.zeros((H_bev, W_bev, C), dtype=np.float32)
    bev_count = np.zeros((H_bev, W_bev), dtype=np.int32)

    # Extract x (forward), y (left)
    x = points_lidar[:, 0]
    y = points_lidar[:, 1]   # NOTE: if your y is right-handed, negate here: y = -points_lidar[:,1]

    # Cell indices
    ix = ((x - x_min) / res).astype(np.int32)
    iy = ((y - y_min) / res).astype(np.int32)

    valid = (ix >= 0) & (ix < H_bev) & (iy >= 0) & (iy < W_bev)
    ix = ix[valid]
    iy = iy[valid]
    F = feats_per_point[valid]

    # Accumulate sums and counts
    for i, j, f in zip(ix, iy, F):
        bev_feat[i, j] += f
        bev_count[i, j] += 1

    # Average where count > 0
    nonzero = bev_count > 0
    bev_feat[nonzero] /= bev_count[nonzero][..., None]

    return bev_feat, bev_count

def pca_visualize_bev(bev_feat, out_png):
    """
    bev_feat: (H_bev, W_bev, C) -> PCA to 3 channels -> save PNG
    """
    H, W, C = bev_feat.shape
    flat = bev_feat.reshape(-1, C)
    # Only use cells with non-zero features for PCA fit
    mask = np.any(flat != 0.0, axis=1)
    if mask.sum() < 10:
        print("[WARN] Too few populated BEV cells for PCA visualization.")
        img = np.zeros((H, W, 3), np.uint8)
        cv2.imwrite(out_png, img)
        return

    pca = PCA(n_components=3)
    comps = np.zeros((flat.shape[0], 3), dtype=np.float32)
    comps[mask] = pca.fit_transform(flat[mask])
    # Normalize to [0,255]
    comps -= comps[mask].min(axis=0, keepdims=True)
    denom = (comps[mask].max(axis=0, keepdims=True) - comps[mask].min(axis=0, keepdims=True) + 1e-6)
    comps[mask] = comps[mask] / denom
    rgb = (comps.reshape(H, W, 3) * 255).astype(np.uint8)
    cv2.imwrite(out_png, rgb)

def main():
    print("=== Visual-BEV basic generator ===")
    # 1) Load data + calibration
    K, R_l2c, t_l2c = load_calib()
    pts_lidar, img_bgr, img_name = load_frame(FRAME_ID)
    H, W = img_bgr.shape[:2]
    print(f"[I/O] Image {img_name} size = {W}x{H}, LiDAR pts = {pts_lidar.shape[0]}")

    # 2) Dense DINOv2 feature map (H, W, C=768)
    print("[VFM] Extracting DINOv2 dense feature map...")
    feat_map = get_dinov2_featmap(img_bgr)  # (H, W, C)

    # 3) Project LiDAR -> image; sample per-point features
    print("[Proj] Project LiDAR -> image & sample features...")
    u, v, Z, pts_vis = project_lidar_to_image(pts_lidar, K, R_l2c, t_l2c, img_bgr.shape)
    # sample features at those pixels
    per_point_feat = feat_map[v, u]  # (N_vis, C)

    # Debug overlay: depth-colored points
    overlay = img_bgr.copy()
    Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
    Zc = (Zn * 255).astype(np.uint8)
    colors = cv2.applyColorMap(Zc, cv2.COLORMAP_JET)
    for (x, y, c) in zip(u, v, colors):
        overlay = cv2.circle(overlay, (int(x), int(y)), 1, tuple(int(x) for x in c[0]), -1)
    cv2.imwrite(str(OUTDIR / f"points_used_{FRAME_ID:06d}.png"), overlay)

    # 4) Splat per-point features into a BEV grid (x forward, y left)
    print("[BEV] Splatting into BEV grid...")
    bev_feat, bev_count = splat_to_bev(
        points_lidar=pts_vis,
        feats_per_point=per_point_feat,
        x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, res=RES,
        agg="mean"
    )
    print(f"[BEV] BEV feat shape = {bev_feat.shape}, non-empty cells = {int((bev_count>0).sum())}")

    # 5) Save outputs
    np.save(OUTDIR / f"visual_bev_feat_{FRAME_ID:06d}.npy", bev_feat)
    np.save(OUTDIR / f"visual_bev_count_{FRAME_ID:06d}.npy", bev_count)

    # 6) PCA visualization of BEV features
    print("[Vis] Making PCA RGB visualization of BEV...")
    pca_visualize_bev(bev_feat, str(OUTDIR / f"visual_bev_rgb_{FRAME_ID:06d}.png"))
    print(f"[DONE] Saved:\n  - {OUTDIR}/visual_bev_feat_{FRAME_ID:06d}.npy\n  - {OUTDIR}/visual_bev_count_{FRAME_ID:06d}.npy\n  - {OUTDIR}/visual_bev_rgb_{FRAME_ID:06d}.png\n  - {OUTDIR}/points_used_{FRAME_ID:06d}.png")

if __name__ == "__main__":
    main()
