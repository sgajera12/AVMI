#!/usr/bin/env python3
"""
Test if Z and intensity fields are swapped
Creates two versions so you can compare
"""

import sqlite3
import struct
import numpy as np
import cv2

def read_both_versions(bag_path, topic='/lidar/points2'):
    """
    Read point cloud TWO ways:
    1. Normal: x, y, z at 0,4,8; intensity at 16
    2. Swapped: x, y at 0,4; intensity at 8; z at 16
    """
    
    print("=== Testing Field Ordering ===\n")
    print(f"Reading: {bag_path}\n")
    
    # Read raw data
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM topics WHERE name=?", (topic,))
    result = cursor.fetchone()
    if not result:
        print("Topic not found!")
        return None, None
    
    cursor.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (result[0],))
    msg_data = cursor.fetchone()
    conn.close()
    
    if not msg_data:
        return None, None
    
    data = msg_data[0]
    offset = 108
    step = 20
    
    # Version 1: Normal (z at offset 8)
    print("Version 1: Normal ordering (z at offset 8)")
    points_normal = []
    i = offset
    while i < len(data) - step:
        try:
            x = struct.unpack('<f', data[i:i+4])[0]
            y = struct.unpack('<f', data[i+4:i+8])[0]
            z = struct.unpack('<f', data[i+8:i+12])[0]  # Z at offset 8
            intensity = struct.unpack('<f', data[i+16:i+20])[0]  # I at offset 16
            
            if (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and
                abs(x) < 200 and abs(y) < 200):
                if not np.isfinite(intensity) or intensity < 0 or intensity > 10:
                    intensity = 1.0
                points_normal.append([x, y, z, intensity])
        except:
            pass
        i += step
    
    points_normal = np.array(points_normal)
    
    # Version 2: Swapped (z at offset 16)
    print("Version 2: Swapped ordering (z at offset 16)")
    points_swapped = []
    i = offset
    while i < len(data) - step:
        try:
            x = struct.unpack('<f', data[i:i+4])[0]
            y = struct.unpack('<f', data[i+4:i+8])[0]
            intensity = struct.unpack('<f', data[i+8:i+12])[0]  # I at offset 8
            z = struct.unpack('<f', data[i+16:i+20])[0]  # Z at offset 16
            
            if (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and
                abs(x) < 200 and abs(y) < 200):
                if not np.isfinite(intensity) or intensity < 0 or intensity > 10:
                    intensity = 1.0
                points_swapped.append([x, y, z, intensity])
        except:
            pass
        i += step
    
    points_swapped = np.array(points_swapped)
    
    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"{'='*60}\n")
    
    print(f"Version 1 (Normal):")
    print(f"  Points: {len(points_normal)}")
    print(f"  Z range: {points_normal[:, 2].min():.2f} to {points_normal[:, 2].max():.2f}m")
    z_range_1 = points_normal[:, 2].max() - points_normal[:, 2].min()
    print(f"  Z span: {z_range_1:.2f}m")
    
    # Filter to reasonable outdoor Z
    valid_1 = (points_normal[:, 2] > -5) & (points_normal[:, 2] < 15)
    print(f"  Points with outdoor Z (-5 to +15m): {valid_1.sum()} ({100*valid_1.sum()/len(points_normal):.1f}%)")
    
    print(f"\nVersion 2 (Swapped):")
    print(f"  Points: {len(points_swapped)}")
    print(f"  Z range: {points_swapped[:, 2].min():.2f} to {points_swapped[:, 2].max():.2f}m")
    z_range_2 = points_swapped[:, 2].max() - points_swapped[:, 2].min()
    print(f"  Z span: {z_range_2:.2f}m")
    
    # Filter to reasonable outdoor Z
    valid_2 = (points_swapped[:, 2] > -5) & (points_swapped[:, 2] < 15)
    print(f"  Points with outdoor Z (-5 to +15m): {valid_2.sum()} ({100*valid_2.sum()/len(points_swapped):.1f}%)")
    
    print(f"\n{'='*60}")
    print("VERDICT:")
    print(f"{'='*60}\n")
    
    # Determine which is better
    if valid_2.sum() > valid_1.sum():
        print("✓ Version 2 (SWAPPED) looks better!")
        print(f"  {valid_2.sum()} valid points vs {valid_1.sum()}")
        print("\n  Z and intensity fields appear to be SWAPPED in your data.")
        best = points_swapped
        best_name = "swapped"
    else:
        print("✓ Version 1 (NORMAL) looks better!")
        print(f"  {valid_1.sum()} valid points vs {valid_2.sum()}")
        print("\n  Z and intensity fields are in NORMAL positions.")
        best = points_normal
        best_name = "normal"
    
    return points_normal, points_swapped, best, best_name

def create_range_image(points, name):
    """Create range image"""
    
    # Filter
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    valid = (r > 1) & (r < 100) & (z > -5) & (z < 15)
    filtered = points[valid]
    
    if len(filtered) < 100:
        print(f"Not enough points for {name} version")
        return None
    
    x, y, z = filtered[:, 0], filtered[:, 1], filtered[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    
    H, W = 64, 1024
    col = ((azimuth + 180) / 360 * W).astype(int)
    col = np.clip(col, 0, W - 1)
    
    elev_min = elevation.min() - 2
    elev_max = elevation.max() + 2
    row = H - 1 - ((elevation - elev_min) / (elev_max - elev_min) * H).astype(int)
    row = np.clip(row, 0, H - 1)
    
    img = np.zeros((H, W))
    for i in range(len(r)):
        if img[row[i], col[i]] == 0 or r[i] < img[row[i], col[i]]:
            img[row[i], col[i]] = r[i]
    
    # Save
    if (img > 0).any():
        mask = img > 0
        norm = np.zeros_like(img)
        norm[mask] = (img[mask] - img[mask].min()) / (img[mask].max() - img[mask].min())
        u8 = (norm * 255).astype(np.uint8)
        
        cv2.imwrite(f'test_{name}_gray.png', u8)
        cv2.imwrite(f'test_{name}_color.png', cv2.applyColorMap(u8, cv2.COLORMAP_JET))
    
    return img

def main():
    bag_path = "/home/pinaka/dataset/AVMI/data/rosbag1210.db3"
    
    normal, swapped, best, best_name = read_both_versions(bag_path)
    
    if normal is None:
        return
    
    print(f"\nCreating range images for both versions...\n")
    
    img_normal = create_range_image(normal, "normal")
    img_swapped = create_range_image(swapped, "swapped")
    
    print("\n✓ Created test images:")
    print("  test_normal_gray.png / test_normal_color.png")
    print("  test_swapped_gray.png / test_swapped_color.png")
    
    print(f"\n✓ Based on analysis, the {best_name.upper()} version is correct.")
    print(f"\nSaving best version as final output...")
    
    # Save best version
    np.save('pointcloud_corrected.npy', best)
    print("  ✓ Saved: pointcloud_corrected.npy")
    
    print("\nNow run:")
    print("  python3 numpy_to_range.py pointcloud_corrected.npy")

if __name__ == "__main__":
    main()