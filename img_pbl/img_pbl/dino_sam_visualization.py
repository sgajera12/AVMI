import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("./dino_sam_out")
frame_id = 200

# Load features
features = np.load(OUTDIR / f"dino_perpix_{frame_id:06d}.npy")  # (N,768)
u = np.load(OUTDIR / f"u_{frame_id:06d}.npy")
v = np.load(OUTDIR / f"v_{frame_id:06d}.npy")
img_path = "/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000200-1581624672_750.jpg"
# Load original image for reference
img = cv2.imread(img_path)
# img = cv2.imread(str(OUTDIR.parent / f"image/{str(frame_id).zfill(5)}.png"))
h, w = img.shape[:2]

# 1. Reduce 768-D to 3D with PCA for visualization
print("Reducing DINO features with PCA...")
pca = PCA(n_components=3)
feat_3d = pca.fit_transform(features)  # (N,3)
feat_3d = (feat_3d - feat_3d.min()) / (feat_3d.max() - feat_3d.min() + 1e-6)
colors = (feat_3d * 255).astype(np.uint8)

# 2. Create empty image and colorize projected points
seg_vis = np.zeros((h, w, 3), np.uint8)
for (x, y, c) in zip(u, v, colors):
    seg_vis[y, x] = c

# 3. Overlay with RGB image for comparison
overlay = cv2.addWeighted(img, 0.5, seg_vis, 0.5, 0)

# 4. Save and show
cv2.imwrite(str(OUTDIR / f"dino_segmentation_{frame_id:06d}.png"), overlay)
print(f"✅ Saved visualization to {OUTDIR}/dino_segmentation_{frame_id:06d}.png")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("DINO Segmentation (PCA colors)")
plt.show()
