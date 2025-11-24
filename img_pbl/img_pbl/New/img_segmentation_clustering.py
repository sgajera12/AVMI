#!/usr/bin/env python3
"""
Unsupervised segmentation using KMeans on DINOv2 features.
Input: dino_feat_*.npy from previous step
Output: segmentation_color.png and seg_labels.npy
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
FEAT_PATH = Path("./out_dino_features/dino_feat_frame000250-1581624677_749.npy")
NUM_CLUSTERS = 5
REDUCE_DIM = 30
OUTDIR = Path("./out_segmentation")
# ------------------------------------------------

def colorize_labels(label_map, num_classes):
    """Assign random colors to each class"""
    H, W = label_map.shape
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(num_classes):
        color_img[label_map == i] = colors[i]
    return color_img

def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    print(f"Loading features from {FEAT_PATH.name}...")
    feat = np.load(FEAT_PATH)
    H, W, C = feat.shape
    print(f"Feature shape: {feat.shape}")

    # Flatten
    flat_feat = feat.reshape(-1, C)
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=REDUCE_DIM)
    flat_pca = pca.fit_transform(flat_feat)

    print(f"Clustering into {NUM_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=5)
    labels = kmeans.fit_predict(flat_pca)
    seg_map = labels.reshape(H, W)

    # Save label map
    np.save(OUTDIR / "seg_labels.npy", seg_map)
    print(f"Saved label map: {OUTDIR}/seg_labels.npy")

    # Colorize
    seg_color = colorize_labels(seg_map, NUM_CLUSTERS)
    out_path = OUTDIR / "segmentation_color.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
    print(f"Saved segmentation visualization: {out_path}")

    # Display
    plt.figure(figsize=(10,5))
    plt.imshow(seg_color)
    plt.title("Unsupervised Segmentation (DINO + KMeans)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
