#!/usr/bin/env python3

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# DINOv2 model and image path
DINO_MODEL = "facebook/dinov2-base"
IMG_PATH = Path("/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000250-1581624677_749.jpg")

# SAM (full model for now)
SAM_TYPE = "vit_b"
SAM_CKPT = Path("/home/pinaka/models/sam_checkpoints/sam_vit_b_01ec64.pth")

# Output path
OUT_DIR = Path("./out_refine")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------
# STEP 1: Load models
print("🔹 Loading DINOv2 model...")
processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
dino_model = AutoModel.from_pretrained(DINO_MODEL)
dino_model.eval()

print("🔹 Loading SAM model...")
sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
mask_generator = SamAutomaticMaskGenerator(sam)

# -----------------------------
# STEP 2.1: Load and preprocess image
# -----------------------------
print("🔹 Loading image...")
bgr = cv2.imread(str(IMG_PATH))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape
print(f"Image shape: {H}x{W}")
# -----------------------------
# STEP 2.2: Extract dense feature map safely
# -----------------------------
print("🔹 Extracting DINO features...")
inputs = processor(images=rgb, return_tensors="pt")

with torch.no_grad():
    outputs = dino_model(**inputs)
    feats = outputs.last_hidden_state  # (1, tokens, dim)

# Drop CLS token and flatten properly
tokens = feats[:, 1:, :].squeeze(0).cpu().numpy()
num_tokens = tokens.shape[0]
feat_dim = tokens.shape[1]

# Approximate square grid
side = int(np.sqrt(num_tokens))
feat_map = tokens[: side * side].reshape(side, side, feat_dim)
print("Tokens shape:", tokens.shape)
print("Num tokens:", num_tokens, "→ side:", side)
print("Feature dim:", feat_dim)
print("Target resize:", (W, H))
print("feat_map shape before resize:", feat_map.shape)

# Resize to match original image
# feat_map = cv2.resize(feat_map.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
print("Resizing each feature channel separately...")
feat_map_resized = np.zeros((H, W, feat_dim), dtype=np.float32)
for i in range(feat_dim):
    feat_map_resized[:, :, i] = cv2.resize(feat_map[:, :, i], (W, H), interpolation=cv2.INTER_LINEAR)
feat_map = feat_map_resized
print("✅ Resized feature map shape:", feat_map.shape)

# -----------------------------
# STEP 3: Generate SAM masks
# -----------------------------
print("Generating SAM masks...")
masks = mask_generator.generate(rgb)
print(f"Generated {len(masks)} masks.")

# -----------------------------
# STEP 4: Compute semantic mean per mask
# -----------------------------
print("Computing mean DINO feature for each SAM mask...")
feat_map_flat = feat_map.reshape(-1, feat_dim)
semantic_img = np.zeros((H, W, 3), dtype=np.float32)

colors = np.random.randint(0, 255, (len(masks), 3))

for i, m in enumerate(masks):
    mask = m["segmentation"].astype(bool)
    if mask.sum() < 100:  # skip very tiny masks
        continue

    # Mean feature vector for this mask region
    feat_mean = feat_map[mask].mean(axis=0)

    # Optionally cluster or PCA reduce here
    color = colors[i]
    semantic_img[mask] = color

# -----------------------------
# STEP 5: Save visualization
# -----------------------------
overlay = (0.5 * rgb + 0.5 * semantic_img).astype(np.uint8)
out_path = OUT_DIR / "dino_sam_refined_fusion.png"

cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Saved refined fusion visualization at: {out_path}")

# -----------------------------
# Optional: show image
# -----------------------------
plt.figure(figsize=(10, 6))
plt.imshow(overlay)
plt.axis("off")
plt.title("Refined DINO + SAM Fusion (Instance-Aware)")
plt.show()
