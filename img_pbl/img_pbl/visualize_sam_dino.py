#!/usr/bin/env python3
import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path

# ---------------- Paths ----------------
img_path = "/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000200-1581624672_750.jpg"
sam_mask_path = "./dino_sam_out/sam_labels_000200.png"
OUTDIR = Path("./dino_sam_out")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load Image ----------------
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
# ---------------- DINO Feature Extraction ----------------
print("🔹 Loading DINOv2-base...")
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

# Resize image for DINO (DINOv2 works best on 518×518 or 448×448)
target_size = 518
img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)

inputs = processor(images=img_resized, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

tokens = outputs.last_hidden_state[0, 1:, :]  # remove CLS token
feat_dim = tokens.shape[-1]
num_patches = tokens.shape[0]
print(f"Total tokens from DINO: {num_patches}, dim={feat_dim}")

# ✅ Compute feature map dimensions from num_patches automatically
side = int(np.sqrt(num_patches))
if side * side != num_patches:
    # Some models (like DINOv2) may not output a perfect square number of tokens
    # so pad to the next square to make reshaping safe
    padded = np.zeros((side * side, feat_dim))
    valid = min(num_patches, padded.shape[0])
    padded[:valid] = tokens[:valid].cpu().numpy()
    feat_map = padded.reshape(side, side, feat_dim)
    print(f"⚠️ Adjusted patch grid to {side}×{side} (padded {side*side - valid} tokens).")
else:
    feat_map = tokens.reshape(side, side, feat_dim).cpu().numpy()

print(f"Feature map shape before resizing: {feat_map.shape}")

# ✅ Ensure feat_map is non-empty
if feat_map.size == 0:
    raise RuntimeError("❌ DINO feature map is empty! Something went wrong in token extraction.")

# ✅ Resize feature map to match original image shape
feat_map_resized = np.zeros((H, W, feat_dim), dtype=np.float32)
for i in range(feat_dim):
    feat_map_resized[:, :, i] = cv2.resize(feat_map[:, :, i], (W, H), interpolation=cv2.INTER_LINEAR)

feat_map_flat = feat_map_resized.reshape(-1, feat_dim)
print(f"✅ Resized each feature channel separately: {feat_map_resized.shape}")
# ---------------- PCA + KMeans ----------------
print("🔹 Reducing features and clustering...")
pca = PCA(n_components=20)
feat_pca = pca.fit_transform(feat_map_flat)

K = 6  # semantic clusters (adjust between 4–8)
kmeans = KMeans(n_clusters=K, random_state=0, n_init=5)
labels = kmeans.fit_predict(feat_pca)

# We already know the image size, so just reshape directly
seg_map = labels.reshape(H, W)
print(f"✅ Segmentation map created with shape: {seg_map.shape}")

# Random colors for each cluster
colors = np.random.randint(0, 255, (K, 3), dtype=np.uint8)
seg_color = np.zeros((H, W, 3), np.uint8)
for k in range(K):
    seg_color[seg_map == k] = colors[k]

# ---------------- Add SAM Boundaries ----------------
if Path(sam_mask_path).exists():
    sam_mask = cv2.imread(sam_mask_path, 0)
    if sam_mask is not None:
        sam_edges = cv2.Canny(sam_mask.astype(np.uint8), 50, 150)
        sam_edges = cv2.dilate(sam_edges, np.ones((2, 2), np.uint8))
        seg_color[sam_edges > 0] = (255, 255, 255)
    else:
        print(f"⚠️ Could not load SAM mask at {sam_mask_path}")

# ---------------- Blend with RGB ----------------
overlay = cv2.addWeighted(img_bgr, 0.5, seg_color, 0.5, 0)
save_path = OUTDIR / "dense_dino_sam_segmentation.png"
cv2.imwrite(str(save_path), overlay)
print(f"✅ Saved segmentation visualization: {save_path}")

# ---------------- Display ----------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.title("Original RGB Image")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Dense DINO + SAM Segmentation")
plt.tight_layout()
plt.show()
