import timm
import torch
import cv2
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image  
from sklearn.cluster import KMeans

MODEL_NAME = "eva02_large_patch14_clip_336"  # Valid TIMM model
IMG_PATH = Path("/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node/frame000250-1581624677_749.jpg")
OUT_PATH = Path("./out_eva_features.png")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading EVA02-CLIP model...")
model = timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()
transform = timm.data.create_transform(input_size=336, is_training=False)

print("🔹 Loading image...")
bgr = cv2.imread(str(IMG_PATH))
if bgr is None:
    raise FileNotFoundError(f"Image not found at {IMG_PATH}")

rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)  

img_t = transform(pil_img).unsqueeze(0).to(device)  

print("🔹 Extracting features...")
with torch.no_grad():
    feats = model.forward_features(img_t) 
feat_map = feats[0].cpu().numpy()

num_tokens, dim = feat_map.shape
side = int(np.sqrt(num_tokens))
print(f"Feature tokens: {num_tokens}, approx grid: {side}x{side}")

feat_map = feat_map[: side * side].reshape(side, side, dim)

print("🔹 Creating PCA visualization...")
flat = feat_map.reshape(-1, dim)
pca = PCA(n_components=3).fit(flat)
rgb = pca.transform(flat)
rgb -= rgb.min(0, keepdims=True)
rgb /= rgb.max(0, keepdims=True) + 1e-8
rgb_img = (rgb.reshape(side, side, 3) * 255).astype(np.uint8)

H, W = bgr.shape[:2]  # (1200, 1920) for your image

print(f"Upsampling feature map from {feat_map.shape[:2]} → {H}x{W}")
feat_map = np.ascontiguousarray(feat_map.astype(np.float32))   # make memory continuous
feat_map_resized = cv2.resize(feat_map, (W, H), interpolation=cv2.INTER_LINEAR)
flat_feats = feat_map_resized.reshape(-1, dim)

NUM_CLUSTERS = 6                     # tweak as you like (4–8 usually works well)
print(f"🔹 Running KMeans with {NUM_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=5)
labels = kmeans.fit_predict(flat_feats)
seg_map = labels.reshape(H, W)

colors = np.random.randint(0, 255, (NUM_CLUSTERS, 3), dtype=np.uint8)
seg_color = np.zeros((H, W, 3), dtype=np.uint8)
for i in range(NUM_CLUSTERS):
    seg_color[seg_map == i] = colors[i]

overlay = cv2.addWeighted(rgb, 0.5, seg_color, 0.5, 0)

cv2.imwrite("eva02_segmentation_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
plt.imshow(overlay)
plt.title(f"EVA02-CLIP unsupervised segmentation ({NUM_CLUSTERS} clusters)")
plt.axis("off")
plt.show()

cv2.imwrite(str(OUT_PATH), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
print(f"Saved PCA feature map at: {OUT_PATH}")

plt.imshow(rgb_img)
plt.title("EVA02-CLIP-336 feature PCA visualization")
plt.axis("off")
plt.show()
