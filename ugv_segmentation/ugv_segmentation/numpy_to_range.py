#!/usr/bin/env python3
"""
Convert numpy point cloud to range image
Super simple - no ROS, no parsing, just math!
"""

import numpy as np
import cv2
import sys

def numpy_to_range_image(points, H=64, W=1024):
    """
    Convert Nx4 point cloud [x,y,z,i] to range image
    
    Simple explanation:
    1. For each point, calculate its direction (angles)
    2. Map that direction to a pixel location
    3. Store the distance at that pixel
    """
    
    print(f"\nConverting {len(points)} points to range image...")
    
    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Calculate distance from sensor (range)
    range_vals = np.sqrt(x**2 + y**2 + z**2)
    
    print(f"Range: {range_vals.min():.2f}m to {range_vals.max():.2f}m")
    
    # Calculate direction angles
    # Azimuth: horizontal angle (like compass direction)
    azimuth_rad = np.arctan2(y, x)
    azimuth_deg = np.degrees(azimuth_rad)
    
    # Elevation: vertical angle (like looking up/down)
    elevation_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
    elevation_deg = np.degrees(elevation_rad)
    
    print(f"Azimuth: {azimuth_deg.min():.1f}° to {azimuth_deg.max():.1f}°")
    print(f"Elevation: {elevation_deg.min():.1f}° to {elevation_deg.max():.1f}°")
    
    # Map angles to image pixels
    # Width = 360° horizontal sweep
    col = ((azimuth_deg + 180) / 360 * W).astype(int)
    col = np.clip(col, 0, W - 1)
    
    # Height = vertical field of view
    elev_min = elevation_deg.min() - 1
    elev_max = elevation_deg.max() + 1
    row = H - 1 - ((elevation_deg - elev_min) / (elev_max - elev_min) * H).astype(int)
    row = np.clip(row, 0, H - 1)
    
    # Create empty image
    image = np.zeros((H, W), dtype=np.float32)
    
    # Fill image: for each point, put its distance at the corresponding pixel
    for i in range(len(points)):
        r_idx = row[i]
        c_idx = col[i]
        
        # If pixel is empty OR this point is closer, update it
        if image[r_idx, c_idx] == 0 or range_vals[i] < image[r_idx, c_idx]:
            image[r_idx, c_idx] = range_vals[i]
    
    # Statistics
    filled = np.count_nonzero(image)
    total = H * W
    print(f"\nFilled {filled}/{total} pixels ({100*filled/total:.1f}%)")
    
    return image

def save_visualizations(img, prefix='range_image'):
    """Save grayscale and colored versions"""
    
    # Check if we have data
    if not (img > 0).any():
        print("ERROR: Image is empty!")
        return
    
    # Normalize to 0-255
    mask = img > 0
    normalized = np.zeros_like(img)
    normalized[mask] = (img[mask] - img[mask].min()) / (img[mask].max() - img[mask].min())
    img_u8 = (normalized * 255).astype(np.uint8)
    
    # Save grayscale
    cv2.imwrite(f'{prefix}_gray.png', img_u8)
    
    # Save colored (JET colormap: blue=close, red=far)
    img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
    cv2.imwrite(f'{prefix}_color.png', img_color)
    
    # Save raw numpy
    np.save(f'{prefix}.npy', img)
    
    print(f"\n✓ Saved:")
    print(f"  {prefix}_gray.png  (grayscale depth)")
    print(f"  {prefix}_color.png (colored depth)")
    print(f"  {prefix}.npy (raw data)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 numpy_to_range.py <pointcloud.npy>")
        return
    
    npy_file = sys.argv[1]
    
    print(f"Loading: {npy_file}")
    points = np.load(npy_file)
    
    print(f"✓ Loaded {len(points)} points")
    print(f"\nPoint cloud stats:")
    print(f"  X: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}m")
    print(f"  Y: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}m")
    print(f"  Z: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}m")
    
    # Create range image
    img = numpy_to_range_image(points, H=64, W=1024)
    
    # Save visualizations
    save_visualizations(img, 'final_range_image')
    
    print("\n✓ Done!")

if __name__ == "__main__":
    main()
