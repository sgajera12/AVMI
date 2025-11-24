import cv2
import matplotlib.pyplot as plt

mask = cv2.imread("dino_sam_out/sam_labels_000200.png", 0)
plt.imshow(mask, cmap='tab20')
plt.title("SAM Region Masks")
plt.show()