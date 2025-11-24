#!/usr/bin/env python3
"""
Unsupervised segmentation using DINOv2 + PCA + KMeans.
Based on Velociraptor-style feature clustering (no human labels).
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -----------------------------
# User settings
# -----------------------------
IMG_PATH = Path("/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000250-1581624677_749.jpg")
OUT_DIR = Path("./out_unsupervised")
OUT_DIR.mkdir(exist_ok=True, parents=True)

DINO_MODEL = "facebook/dinov2-base"
N_PCA_COMPONENTS = 20
N_CLUSTERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# STEP 1: Load DINO model
# -----------------------------
print("🔹 Loading DINOv2 model...")
processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
model = AutoModel.from_pretrained(DINO_MODEL).to(DEVICE).eval()

# -----------------------------
# STEP 2: Load image
# -----------------------------
print("🔹 Loading image...")
bgr = cv2.imread(str(IMG_PATH))
if bgr is None:
    raise FileNotFoundError(f"Could not read image {IMG_PATH}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W, _ = rgb.shape
print(f"Image shape: {H}x{W}")

# -----------------------------
# STEP 3: Extract dense features
# -----------------------------
print("🔹 Extracting DINO features...")
inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
tokens = outputs.last_hidden_state[0, 1:, :].cpu().numpy()  # remove CLS
feat_dim = tokens.shape[-1]
num_tokens = tokens.shape[0]
side = int(np.sqrt(num_tokens))
print(f"Tokens: {num_tokens}, Dim: {feat_dim}, Side: {side}")

# Reshape and resize to image size
feat_map = tokens[: side * side].reshape(side, side, feat_dim)
feat_map_resized = np.zeros((H, W, feat_dim), dtype=np.float32)
for i in range(feat_dim):
    feat_map_resized[:, :, i] = cv2.resize(feat_map[:, :, i], (W, H), interpolation=cv2.INTER_LINEAR)
print("✅ Feature map ready:", feat_map_resized.shape)

# -----------------------------
# STEP 4: PCA reduction
# -----------------------------
print("🔹 Reducing feature dimension via PCA...")
flat_feats = feat_map_resized.reshape(-1, feat_dim)
pca = PCA(n_components=N_PCA_COMPONENTS, random_state=0)
feat_pca = pca.fit_transform(flat_feats)
print("✅ Reduced feature shape:", feat_pca.shape)

# -----------------------------
# STEP 5: KMeans clustering
# -----------------------------
print("🔹 Clustering with KMeans...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5)
labels = kmeans.fit_predict(feat_pca)
seg_map = labels.reshape(H, W)
print("✅ Segmentation map shape:", seg_map.shape)

# -----------------------------
# STEP 6: Visualization
# -----------------------------
# Random colors for clusters
colors = np.random.randint(0, 255, (N_CLUSTERS, 3), dtype=np.uint8)
seg_color = np.zeros((H, W, 3), np.uint8)
for i in range(N_CLUSTERS):
    seg_color[seg_map == i] = colors[i]

cv2.imwrite(str(OUT_DIR / "unsupervised_segmentation.png"), cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
print("💾 Saved: unsupervised_segmentation.png")

# Overlay with RGB
overlay = cv2.addWeighted(rgb, 0.5, seg_color, 0.5, 0)
cv2.imwrite(str(OUT_DIR / "overlay_segmentation.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("💾 Saved: overlay_segmentation.png")

# Show in matplotlib
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Unsupervised Segmentation (DINOv2 + PCA + KMeans)")
plt.tight_layout()
plt.show()
