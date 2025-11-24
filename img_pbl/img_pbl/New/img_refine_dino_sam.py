#!/usr/bin/env python3
"""
Fuse DINOv2 + SAM: refine coarse segmentation with crisp SAM boundaries.
Input: segmentation_color.png + original RGB image
Output: dino_sam_fused.png (crisp segmented overlay)
"""

import numpy as np
import cv2
from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------------- USER SETTINGS ----------------
IMG_PATH = Path("/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000250-1581624677_749.jpg")
SEG_PATH = Path("./out_segmentation/segmentation_color.png")
OUTDIR = Path("./out_fusion")

# SAM checkpoint (MobileSAM or full SAM)
USE_MOBILE_SAM = True
SAM_CKPT = Path("/home/pinaka/models/sam_checkpoints/sam_vit_b_01ec64.pth")
# ------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("▶ Loading images...")
    rgb = cv2.cvtColor(cv2.imread(str(IMG_PATH)), cv2.COLOR_BGR2RGB)
    seg = cv2.cvtColor(cv2.imread(str(SEG_PATH)), cv2.COLOR_BGR2RGB)

    H, W, _ = rgb.shape
    print(f"Image shape: {H}x{W}")

    # ---------------- Load SAM ----------------
    print("▶ Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=str(SAM_CKPT))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # ---------------- Generate SAM masks ----------------
    print("▶ Generating SAM masks...")
    masks = mask_generator.generate(rgb)
    print(f"Total masks generated: {len(masks)}")

    # ---------------- Fuse SAM with DINO segmentation ----------------
    fused = seg.copy()
    edge_overlay = rgb.copy()

    for m in masks:
        mask = m["segmentation"].astype(np.uint8)
        edges = cv2.Canny(mask * 255, 100, 200)
        fused[edges > 0] = (255, 255, 255)  # white edges overlay

    # Blend with original image for visualization
    overlay = cv2.addWeighted(rgb, 0.5, fused, 0.5, 0)

    # ---------------- Save results ----------------
    out_path = OUTDIR / "dino_sam_fused.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved fused segmentation: {out_path}")

if __name__ == "__main__":
    main()
