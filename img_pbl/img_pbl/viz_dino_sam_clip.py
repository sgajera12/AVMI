#!/usr/bin/env python3
import numpy as np
import torch
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from pathlib import Path

# ---------------- PATHS ----------------
img_path = "/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000200-1581624672_750.jpg"
sam_mask_path = "./dino_sam_out/sam_labels_000200.png"
OUTDIR = Path("./dino_sam_out")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD IMAGE ----------------
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

# ---------------- DINO FEATURE EXTRACTION ----------------
print("🔹 Extracting DINO features...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

target_size = 518
img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
inputs = processor(images=img_resized, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

tokens = outputs.last_hidden_state[0, 1:, :].cpu().numpy()   # remove CLS token
feat_dim = tokens.shape[-1]
side = int(np.sqrt(tokens.shape[0]))
feat_map = tokens.reshape(side, side, feat_dim)
feat_map_resized = np.zeros((H, W, feat_dim), dtype=np.float32)
for i in range(feat_dim):
    feat_map_resized[:, :, i] = cv2.resize(feat_map[:, :, i], (W, H), interpolation=cv2.INTER_LINEAR)
feat_map_flat = feat_map_resized.reshape(-1, feat_dim)
print(f"✅ Feature map shape: {feat_map_resized.shape}")

# ---------------- CLUSTERING ----------------
print("🔹 Clustering DINO features...")
pca = PCA(n_components=20)
feat_pca = pca.fit_transform(feat_map_flat)
K = 6
kmeans = KMeans(n_clusters=K, random_state=0, n_init=5)
labels = kmeans.fit_predict(feat_pca)
seg_map = labels.reshape(H, W)

# ---------------- CLIP TEXT EMBEDDINGS ----------------
print("🔹 Loading CLIP for automatic labeling...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_prompts = ["sky", "tree", "grass", "road", "mud", "building", "vegetation", "person"]
with torch.no_grad():
    text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ---------------- MATCH CLUSTERS TO TEXT ----------------
print("🔹 Matching clusters to text prompts...")
cluster_labels = {}

# Project PCA features up to 512-D using a simple linear projection
proj = torch.nn.Linear(feat_pca.shape[1], 512, bias=False).to(device)
proj.weight.data.normal_(0, 0.02)  # small random weights

for k in range(K):
    mask = (seg_map == k)
    if not np.any(mask):
        continue

    # Average DINO cluster feature and project to 512-D
    cluster_feat = torch.tensor(feat_pca[labels == k].mean(axis=0), dtype=torch.float32, device=device)
    cluster_feat = proj(cluster_feat.unsqueeze(0))  # shape (1,512)
    cluster_feat = cluster_feat / cluster_feat.norm(dim=-1, keepdim=True)

    # Compute cosine similarity with each CLIP text embedding
    sims = torch.matmul(text_features, cluster_feat.T).squeeze()  # (8,)
    best_idx = int(torch.argmax(sims))
    best_label = text_prompts[best_idx]
    cluster_labels[k] = best_label
    print(f"Cluster {k} → {best_label} (similarity={sims[best_idx]:.3f})")


# ---------------- COLOR & LABEL MAP ----------------
colors = np.random.randint(0, 255, (K, 3), dtype=np.uint8)
seg_color = np.zeros((H, W, 3), np.uint8)
for k in range(K):
    seg_color[seg_map == k] = colors[k]

# Add SAM boundaries if available
if Path(sam_mask_path).exists():
    sam_mask = cv2.imread(sam_mask_path, 0)
    sam_edges = cv2.Canny(sam_mask, 50, 150)
    sam_edges = cv2.dilate(sam_edges, np.ones((2,2), np.uint8))
    seg_color[sam_edges > 0] = (255, 255, 255)

# ---------------- OVERLAY & TEXT ----------------
overlay = cv2.addWeighted(img_bgr, 0.5, seg_color, 0.5, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
for k, label in cluster_labels.items():
    pts = np.argwhere(seg_map == k)
    if len(pts) == 0: continue
    y, x = np.mean(pts, axis=0).astype(int)
    cv2.putText(overlay, label, (x - 40, y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

save_path = OUTDIR / "dense_dino_sam_clip_labels.png"
cv2.imwrite(str(save_path), overlay)
print(f"✅ Saved labeled segmentation: {save_path}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("DINO + SAM + CLIP Labels")
plt.tight_layout()
plt.show()
