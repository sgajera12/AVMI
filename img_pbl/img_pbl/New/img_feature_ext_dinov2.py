#!/usr/bin/env python3
import os
import cv2
import math
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import torch
from transformers import AutoImageProcessor, AutoModel

# ---------------- USER SETTINGS ----------------
IMG_INDEX = 250   # <— change this number to pick which image to use 
DATASET_ROOT = Path("/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/")
OUTDIR = Path("./out_dino_features")
MODEL_NAME = "facebook/dinov2-base"
TARGET_SIZE = 518  # multiple of 14 (ViT patch size)

def find_image_by_index(idx: int) -> Path:
    """Find the image in the folder by numeric index"""
    imgs = sorted(DATASET_ROOT.glob("frame*.jpg"))
    if idx < 0 or idx >= len(imgs):
        raise IndexError(f"Index {idx} out of range (found {len(imgs)} images)")
    return imgs[idx]

def load_image(img_path: Path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

@torch.no_grad()
def extract_dino_features(img_rgb: np.ndarray, model_name=MODEL_NAME, target_size=TARGET_SIZE, device=None):
    H, W = img_rgb.shape[:2]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor + model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Resize to DINO-friendly square
    img_sq = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    inputs = processor(images=img_sq, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Remove CLS token and reshape
    tokens = outputs.last_hidden_state[0, 1:, :].cpu().numpy()
    C = tokens.shape[-1]
    side = int(math.sqrt(tokens.shape[0]))
    feat_small = tokens.reshape(side, side, C)

    # Upsample to original resolution
    feat_full = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        feat_full[:, :, c] = cv2.resize(feat_small[:, :, c], (W, H), interpolation=cv2.INTER_LINEAR)

    return feat_full, feat_small

def save_pca_visualization(feat_full, out_path):
    """Save PCA→RGB visualization to quickly view feature patterns"""
    H, W, C = feat_full.shape
    flat = feat_full.reshape(-1, C)
    idx = np.random.choice(flat.shape[0], size=min(200_000, flat.shape[0]), replace=False)
    pca = PCA(n_components=3).fit(flat[idx])
    rgb = pca.transform(flat)
    rgb -= rgb.min(0, keepdims=True)
    rgb /= rgb.max(0, keepdims=True) + 1e-8
    rgb_img = (rgb.reshape(H, W, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    img_path = find_image_by_index(IMG_INDEX)
    print(f"▶ Using image {IMG_INDEX}: {img_path.name}")

    img_bgr, img_rgb = load_image(img_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ Running DINOv2 on {device}...")

    feat_full, feat_small = extract_dino_features(img_rgb, MODEL_NAME, TARGET_SIZE, device)

    name = img_path.stem
    np.save(OUTDIR / f"dino_feat_{name}.npy", feat_full)
    np.save(OUTDIR / f"dino_tokens_{name}.npy", feat_small)
    print(f"Saved: {OUTDIR}/dino_feat_{name}.npy  shape={feat_full.shape}")

    out_pca = OUTDIR / f"dino_pca_{name}.png"
    save_pca_visualization(feat_full, out_pca)
    print(f"Saved PCA visualization: {out_pca}")

if __name__ == "__main__":
    main()
