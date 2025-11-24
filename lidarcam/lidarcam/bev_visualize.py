import numpy as np
import matplotlib.pyplot as plt
import cv2

data = np.load('bev_out_2_5D/bev_tensor_frame000250.npz')
height = data['height']
intensity = data['intensity']
density = data['density']
rgb = data['rgb']

gradient_x, gradient_y = np.gradient(np.nan_to_num(height, nan=0.0))
slope = np.sqrt(gradient_x**2 + gradient_y**2)

slope_norm = np.clip(slope / np.nanmax(slope), 0, 1)

height_norm = (height - np.nanmin(height)) / (np.nanmax(height) - np.nanmin(height))
height_vis = cv2.applyColorMap((height_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

slope_mask = (slope_norm * 255).astype(np.uint8)
slope_color = cv2.applyColorMap(slope_mask, cv2.COLORMAP_HOT)
fused = cv2.addWeighted(height_vis, 0.6, slope_color, 0.4, 0)
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.imshow(height_vis[..., ::-1])
plt.title("Height Map (Z in meters)")
plt.subplot(1,3,2)
plt.imshow(slope_color[..., ::-1])
plt.title("Density Map")
plt.subplot(1,3,3)
plt.imshow(fused[..., ::-1])
plt.title("Combined Height + Density")
plt.show()
