#!/usr/bin/env python3
"""
Diagnostic script - checks if fields are in correct order
"""

import numpy as np
import cv2
import sqlite3
import struct

def read_and_diagnose(bag_path, topic):
    """Read bag and show diagnostic info"""
    
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM topics WHERE name=?", (topic,))
    result = cursor.fetchone()
    if not result:
        print("Topic not found!")
        return None
    
    cursor.execute("SELECT data FROM messages WHERE topic_id=? LIMIT 1", (result[0],))
    msg = cursor.fetchone()
    conn.close()
    
    if not msg:
        return None
    
    data = msg[0]
    offset = 108
    step = 20
    
    print("=== Reading Sample Points ===\n")
    
    # Read first 20 points and show them
    sample_points = []
    for i in range(offset, min(offset + step * 20, len(data) - step), step):
        try:
            x = struct.unpack('<f', data[i:i+4])[0]
            y = struct.unpack('<f', data[i+4:i+8])[0]
            z = struct.unpack('<f', data[i+8:i+12])[0]
            val16 = struct.unpack('<f', data[i+16:i+20])[0]
            
            sample_points.append([x, y, z, val16])
            
            if len(sample_points) <= 10:
                print(f"Point {len(sample_points):2d}: x={x:8.2f}, y={y:8.2f}, z={z:8.2f}, field@16={val16:8.3f}")
        except:
            break
    
    print("\n=== Analyzing Point Distribution ===\n")
    
    # Now read all points with robust parsing
    all_points = []
    consecutive_bad = 0
    
    i = offset
    while i < len(data) - step and consecutive_bad < 50:
        try:
            x = struct.unpack('<f', data[i:i+4])[0]
            y = struct.unpack('<f', data[i+4:i+8])[0]
            z = struct.unpack('<f', data[i+8:i+12])[0]
            val16 = struct.unpack('<f', data[i+16:i+20])[0]
            
            # Very lenient check
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                all_points.append([x, y, z, val16])
                consecutive_bad = 0
            else:
                consecutive_bad += 1
        except:
            consecutive_bad += 1
        
        i += step
    
    points = np.array(all_points)
    
    print(f"Loaded {len(points)} points\n")
    
    # Show distribution
    print("RAW data ranges (no filtering):")
    print(f"  X: {points[:, 0].min():.2f} to {points[:, 0].max():.2f} (range: {points[:, 0].max() - points[:, 0].min():.2f}m)")
    print(f"  Y: {points[:, 1].min():.2f} to {points[:, 1].max():.2f} (range: {points[:, 1].max() - points[:, 1].min():.2f}m)")
    print(f"  Z: {points[:, 2].min():.2f} to {points[:, 2].max():.2f} (range: {points[:, 2].max() - points[:, 2].min():.2f}m)")
    print(f"  Field@16: {points[:, 3].min():.3f} to {points[:, 3].max():.3f}")
    
    # Check what makes sense for a typical outdoor LiDAR scene
    print("\n=== Filtering Analysis ===\n")
    
    # Typical outdoor scene should have:
    # - Horizontal extent (X, Y): 5-100m
    # - Vertical extent (Z): -2m to +10m (ground to trees)
    
    # Filter 1: Reasonable XY range (typical LiDAR range)
    xy_range = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    valid_range = (xy_range > 1.0) & (xy_range < 100)
    
    print(f"Points with reasonable XY range (1-100m): {valid_range.sum()}/{len(points)}")
    
    # Filter 2: Reasonable Z (for outdoor scene with trees)
    valid_z_outdoor = (points[:, 2] > -5) & (points[:, 2] < 15)
    print(f"Points with outdoor-reasonable Z (-5 to +15m): {valid_z_outdoor.sum()}/{len(points)}")
    
    # Combined filter
    valid = valid_range & valid_z_outdoor
    filtered = points[valid]
    
    print(f"\nAfter filtering: {len(filtered)} points")
    
    if len(filtered) > 100:
        print("\nFiltered data ranges:")
        print(f"  X: {filtered[:, 0].min():.2f} to {filtered[:, 0].max():.2f}")
        print(f"  Y: {filtered[:, 1].min():.2f} to {filtered[:, 1].max():.2f}")
        print(f"  Z: {filtered[:, 2].min():.2f} to {filtered[:, 2].max():.2f}")
        
        # Check if this looks like reasonable LiDAR data
        z_range = filtered[:, 2].max() - filtered[:, 2].min()
        print(f"\nZ range: {z_range:.2f}m")
        
        if z_range < 20:
            print("✓ Z range looks reasonable for outdoor scene")
        else:
            print("⚠ Z range still seems too large")
        
        return filtered
    else:
        print("\n⚠ Not enough valid points after filtering!")
        print("\nTrying alternative interpretation...")
        
        # Maybe Z and intensity are swapped?
        print("\n=== Trying Swapped Fields (Z ↔ Intensity) ===\n")
        
        # Treat field@16 as Z instead
        swapped = points.copy()
        swapped[:, 2] = points[:, 3]  # Put field@16 into Z
        swapped[:, 3] = points[:, 2]  # Put old Z into intensity
        
        # Filter with swapped data
        valid_range_sw = (np.sqrt(swapped[:, 0]**2 + swapped[:, 1]**2) > 1.0) & \
                         (np.sqrt(swapped[:, 0]**2 + swapped[:, 1]**2) < 100)
        valid_z_sw = (swapped[:, 2] > -5) & (swapped[:, 2] < 15)
        valid_sw = valid_range_sw & valid_z_sw
        
        filtered_sw = swapped[valid_sw]
        
        print(f"Swapped interpretation: {len(filtered_sw)} valid points")
        
        if len(filtered_sw) > len(filtered):
            print("✓ Swapped version has MORE valid points!")
            print("\nSwapped data ranges:")
            print(f"  X: {filtered_sw[:, 0].min():.2f} to {filtered_sw[:, 0].max():.2f}")
            print(f"  Y: {filtered_sw[:, 1].min():.2f} to {filtered_sw[:, 1].max():.2f}")
            print(f"  Z: {filtered_sw[:, 2].min():.2f} to {filtered_sw[:, 2].max():.2f}")
            return filtered_sw
        else:
            return filtered if len(filtered) > 0 else None

def make_range_image(points, H=64, W=1024):
    """Create range image"""
    
    if points is None or len(points) < 100:
        print("Not enough points!")
        return None
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    
    print(f"\n=== Creating Range Image ===")
    print(f"Using {len(points)} points")
    print(f"Range: {r.min():.2f}m to {r.max():.2f}m")
    
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    
    print(f"Azimuth: {azimuth.min():.1f}° to {azimuth.max():.1f}°")
    print(f"Elevation: {elevation.min():.1f}° to {elevation.max():.1f}°")
    
    # Map to image
    col = ((azimuth + 180) / 360 * W).astype(int)
    col = np.clip(col, 0, W - 1)
    
    elev_min = max(elevation.min() - 2, -90)
    elev_max = min(elevation.max() + 2, 90)
    
    row = H - 1 - ((elevation - elev_min) / (elev_max - elev_min) * H).astype(int)
    row = np.clip(row, 0, H - 1)
    
    img = np.zeros((H, W))
    for i in range(len(r)):
        if img[row[i], col[i]] == 0 or r[i] < img[row[i], col[i]]:
            img[row[i], col[i]] = r[i]
    
    filled = np.count_nonzero(img)
    print(f"Filled: {filled}/{H*W} pixels ({100*filled/(H*W):.1f}%)")
    
    return img

def save_viz(img, name='range'):
    """Save images"""
    
    if img is None or not (img > 0).any():
        return
    
    mask = img > 0
    norm = np.zeros_like(img)
    norm[mask] = (img[mask] - img[mask].min()) / (img[mask].max() - img[mask].min())
    u8 = (norm * 255).astype(np.uint8)
    
    cv2.imwrite(f'{name}_gray.png', u8)
    cv2.imwrite(f'{name}_jet.png', cv2.applyColorMap(u8, cv2.COLORMAP_JET))
    np.save(f'{name}.npy', img)
    
    print(f"\n✓ Saved {name}_gray.png, {name}_jet.png, {name}.npy")

def main():
    bag = "/home/pinaka/dataset/AVMI/data/rosbag1210.db3"
    topic = "/lidar/points2"
    
    print(f"Reading: {bag}\n")
    
    points = read_and_diagnose(bag, topic)
    
    if points is not None:
        img = make_range_image(points)
        save_viz(img, 'filtered_range')
    
    print("\n✓ Done!")

if __name__ == "__main__":
    main()