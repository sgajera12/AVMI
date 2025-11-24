#!/usr/bin/env python3
import os
from pathlib import Path
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

BASE = Path("/home/pinaka/dataset/rellis3d/Rellis-3D")
SEQ  = "00000"
frame_id = 200

calib_file = BASE / f"calibration/{SEQ}/transforms.yaml"
intr_file  = BASE / f"intrinsic/{SEQ}/camera_info.txt"
lidar_dir  = BASE / f"lidar/{SEQ}/os1_cloud_node_kitti_bin"
image_dir  = BASE / f"image/{SEQ}/pylon_camera_node"

# SAM config
USE_MOBILE_SAM = False  # True = MobileSAM (faster); False = original SAM
SAM_CKPT = "/home/pinaka/models/sam_checkpoints/mobile_sam.pt" if USE_MOBILE_SAM else "/home/pinaka/models/sam_checkpoints/sam_vit_b_01ec64.pth"

OUTDIR = Path("./dino_sam_out"); OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------
def load_calib(calib_file, intr_file):
    with open(calib_file, "r") as f:
        tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]
    q, t = tf["q"], tf["t"]
    R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
    t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)
    # invert to LiDAR->Cam
    R_l2c = R_c2l.T
    t_l2c = -R_c2l.T @ t_c2l
    fx, fy, cx, cy = np.loadtxt(intr_file)
    K = np.array([[fx, 0,  cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float32)
    return R_l2c, t_l2c, K

@torch.no_grad()
def extract_dino_features(img_bgr, model_name="facebook/dinov2-base", device="cuda"):
    """
    Returns:
      feat_bchw: torch [1,C,Hf,Wf]   (feature map)
      meta: {H,W, Hp,Wp, patch, stride, resized}
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    Hp, Wp = inputs["pixel_values"].shape[-2:]
    out = model(**inputs)
    tok = out.last_hidden_state  # [1, N+1, C] for ViT (CLS + patches)
    C = tok.shape[-1]
    # drop CLS if present
    if tok.shape[1] == (Hp//14)*(Wp//14):
        spatial = tok
    else:
        spatial = tok[:,1:,:]
    Ht = Hp // 14; Wt = Wp // 14
    spatial = spatial.reshape(1, Ht, Wt, C).permute(0,3,1,2).contiguous()  # [1,C,Ht,Wt]
    meta = dict(H=H, W=W, Hp=Hp, Wp=Wp, patch=14, stride=14, resized=True, model=model_name)
    return spatial, meta

def upsample_to_image(feat_bchw, meta):
    """Upsample feature map to original image size for easy (u,v) sampling at integer pixels."""
    H, W = meta["H"], meta["W"]
    feat_up = F.interpolate(feat_bchw, size=(H, W), mode="bilinear", align_corners=True)  # [1,C,H,W]
    return feat_up  # torch

def run_sam(img_bgr, use_mobile=True, ckpt_path=SAM_CKPT, device="cuda"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if use_mobile:
        # MobileSAM API
        from mobile_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_t"](checkpoint=ckpt_path).to(device)
        predictor = SamPredictor(sam)
    else:
        # Original SAM API
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=ckpt_path).to(device)
        predictor = SamPredictor(sam)

    predictor.set_image(img_rgb)
    # no prompts; use the automatic mask generator instead (coarse regions)
    # For stability and speed, we use predictor.get_image_embedding + simple slic:
    # But simplest is AutomaticMaskGenerator (original SAM):
    if not use_mobile:
        from segment_anything import SamAutomaticMaskGenerator
        mask_generator = SamAutomaticMaskGenerator(sam,
            points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=200)
        masks = mask_generator.generate(img_rgb)  # list of dicts with 'segmentation'
        if len(masks) == 0:
            return None
        # Build a label map
        H, W = img_bgr.shape[:2]
        label = np.zeros((H,W), np.int32)
        for i, m in enumerate(masks, start=1):
            label[m["segmentation"]] = i
        return label
    else:
        # MobileSAM lacks the auto generator; we do a quick SLIC over image for regions, or skip SAM step.
        # Here, fallback: return None to skip SAM pooling if MobileSAM auto-gen not available.
        return None

def sample_feat_at_uv(feat_up_bchw, u, v):
    """feat_up_bchw [1,C,H,W]; u,v int arrays on image grid -> returns [N,C] numpy"""
    C, H, W = feat_up_bchw.shape[1:]
    # grid_sample expects normalized coords [-1,1]
    x = (u / (W-1)) * 2 - 1
    y = (v / (H-1)) * 2 - 1
    grid = torch.from_numpy(np.stack([x, y], axis=-1)).float()[None, None].to(feat_up_bchw.device)
    samp = F.grid_sample(feat_up_bchw, grid, mode="bilinear", align_corners=True)  # [1,C,1,N]
    samp = samp.squeeze(2).squeeze(0).permute(1,0).contiguous().cpu().numpy()  # [N,C]
    return samp

def region_pool_features(feat_up_bchw, label_map):
    """
    Average DINO features inside each SAM region.
    Returns:
      region_vecs: (R, C)
      region_of_pixel: (H,W) int in [0..R-1] (0 means background/unlabeled)
    """
    if label_map is None:
        return None, None
    feat = feat_up_bchw.squeeze(0).permute(1,2,0).contiguous().cpu().numpy()  # [H,W,C]
    H, W, C = feat.shape
    labels = label_map
    Rmax = labels.max()
    if Rmax == 0:
        return None, None
    region_vecs = np.zeros((Rmax+1, C), np.float32)  # 0-th unused
    counts = np.zeros(Rmax+1, np.int64)
    for rid in range(1, Rmax+1):
        mask = (labels == rid)
        if mask.any():
            region_vecs[rid] = feat[mask].mean(axis=0)
            counts[rid] = mask.sum()
    # map every pixel to its region embedding id (keep same id)
    return region_vecs, labels

# ---------- MAIN ----------
def main():
    # 1) IO
    bins = sorted([p for p in lidar_dir.iterdir() if p.suffix == ".bin"])
    imgs = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg",".png"]])
    assert bins and imgs, "Missing files"
    lidar_file = bins[frame_id]; image_file = imgs[frame_id]

    # 2) Load data
    pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1,4)[:,:3]
    img = cv2.imread(str(image_file)); assert img is not None
    H, W = img.shape[:2]

    # 3) Calib + project LiDAR->Camera
    R_l2c, t_l2c, K = load_calib(calib_file, intr_file)
    Pc = (R_l2c @ pts.T + t_l2c.reshape(3,1)).T
    Z = Pc[:,2]; front = Z > 0
    Pc = Pc[front]; Z = Z[front]; pts_front = pts[front]
    uv = (K @ (Pc.T / Z)).T
    u = uv[:,0]; v = uv[:,1]
    inside = (u>=0)&(u<W)&(v>=0)&(v<H)
    u = u[inside].astype(np.int32)
    v = v[inside].astype(np.int32)
    pts_vis = pts_front[inside]  # Nx3

    print(f"Projected {pts.shape[0]} pts → {front.sum()} in front → {u.shape[0]} on image.")

    # 4) DINO features (dense)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_bchw, meta = extract_dino_features(img, device=device)
    dino_up = upsample_to_image(dino_bchw, meta)  # [1,C,H,W] aligned to image grid

    # 5) (Optional) SAM region label map, then region-pooled DINO
    label_map = run_sam(img, use_mobile=USE_MOBILE_SAM, ckpt_path=SAM_CKPT, device=device)
    region_vecs, region_of_pixel = region_pool_features(dino_up, label_map)

    # 6) Sample features per LiDAR point (two flavors)
    # 6a) per-pixel DINO (always available)
    feat_per_point = sample_feat_at_uv(dino_up, u, v)        # (N, C)

    # 6b) SAM region-pooled DINO (if SAM succeeded)
    if region_vecs is not None:
        # assign region id to each (u,v), then fetch its mean embedding
        rid = region_of_pixel[v, u]  # (N,)
        # rid==0 means unlabeled; fallback to per-pixel feature
        feat_region = np.where(rid[:,None] > 0, region_vecs[rid], feat_per_point)
    else:
        feat_region = feat_per_point

    print(f"Per-point feature shapes: pixel={feat_per_point.shape}, region={feat_region.shape}")

    # 7) Save results for next step (BEV splat)
    np.save(OUTDIR / f"points_cam_{frame_id:06d}.npy", pts_vis)           # (N,3)
    np.save(OUTDIR / f"u_{frame_id:06d}.npy", u)
    np.save(OUTDIR / f"v_{frame_id:06d}.npy", v)
    np.save(OUTDIR / f"dino_perpix_{frame_id:06d}.npy", feat_per_point)   # (N,C)
    np.save(OUTDIR / f"dino_region_{frame_id:06d}.npy", feat_region)      # (N,C)
    if label_map is not None:
        cv2.imwrite(str(OUTDIR / f"sam_labels_{frame_id:06d}.png"), (label_map.astype(np.float32)/label_map.max()*255).astype(np.uint8))

    # (Optional) write a quick colored overlay to visually check projections
    # colorize via depth
    Zpts = Z[inside]
    z_norm = (Zpts - Zpts.min()) / (Zpts.max()-Zpts.min() + 1e-6)
    z_color = cv2.applyColorMap((z_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = img.copy()
    for (x,y,c) in zip(u, v, z_color):
        cv2.circle(overlay, (int(x),int(y)), 1, tuple(int(vv) for vv in c[0]), -1)
    cv2.imwrite(str(OUTDIR / f"overlay_depth_{frame_id:06d}.png"), overlay)

    print(f"[OK] Saved per-point features + overlay to {OUTDIR}")

if __name__ == "__main__":
    main()
