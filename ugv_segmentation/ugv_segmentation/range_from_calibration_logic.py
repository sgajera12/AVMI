#!/usr/bin/env python3
"""
Range Image Generator - Maximize Resolution
Fits the data range to the full image height for best detail
"""

import sqlite3
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2
import sys

def load_sample_from_bag(bag_path, frame_number=0):
    """Load point cloud from bag"""
    
    print(f"Opening bag: {bag_path}")
    
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name FROM topics WHERE name LIKE '%lidar%' OR name LIKE '%point%'")
    topics = cursor.fetchall()
    
    if not topics:
        print("ERROR: No LiDAR topic found!")
        return None
    
    topic_id, topic_name = topics[0]
    print(f"Using topic: {topic_name}")
    
    cursor.execute(
        "SELECT data FROM messages WHERE topic_id=? LIMIT 1 OFFSET ?",
        (topic_id, frame_number)
    )
    
    msg_data = cursor.fetchone()
    conn.close()
    
    if not msg_data:
        print(f"ERROR: Frame {frame_number} not found!")
        return None
    
    msg = deserialize_message(msg_data[0], PointCloud2)
    
    available_fields = [field.name for field in msg.fields]
    
    fields_to_extract = ['x', 'y', 'z']
    has_intensity = 'intensity' in available_fields or 'i' in available_fields
    
    if has_intensity:
        fields_to_extract.append('intensity' if 'intensity' in available_fields else 'i')
    
    points = []
    for point in pc2.read_points(msg, field_names=fields_to_extract, skip_nans=True):
        if has_intensity:
            points.append([point[0], point[1], point[2], point[3]])
        else:
            points.append([point[0], point[1], point[2], 1.0])
    
    return np.array(points)

def create_range_image_zoomed(points, width=2048, height=512, min_padding=2.0):
    """
    Create range image that ZOOMS IN on the data range
    Uses full image height for the actual data range (not full 180 degree sphere)
    
    Args:
        points: Nx4 array [x, y, z, intensity]
        width: Horizontal resolution (more = more detail)
        height: Vertical resolution (more = more detail in your limited FOV)
        min_padding: Minimum degrees of padding (just to avoid clipping edges)
    """
    
    print("\n" + "="*60)
    print(f"Creating ZOOMED range image from {len(points)} points")
    print("="*60)
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Calculate spherical coordinates
    range_vals = np.sqrt(x**2 + y**2 + z**2)
    azimuth_deg = np.degrees(np.arctan2(y, x))
    elevation_deg = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    
    print(f"\nData ranges:")
    print(f"  X: {x.min():.2f}m to {x.max():.2f}m")
    print(f"  Y: {y.min():.2f}m to {y.max():.2f}m")
    print(f"  Z: {z.min():.2f}m to {z.max():.2f}m")
    print(f"  Distance: {range_vals.min():.2f}m to {range_vals.max():.2f}m")
    
    print(f"\nAngular ranges:")
    print(f"  Azimuth: {azimuth_deg.min():.1f} to {azimuth_deg.max():.1f} deg")
    print(f"  Elevation: {elevation_deg.min():.1f} to {elevation_deg.max():.1f} deg")
    
    elevation_span = elevation_deg.max() - elevation_deg.min()
    print(f"  Elevation span: {elevation_span:.1f} deg")
    
    # Use minimal padding - just enough to avoid edge clipping
    elev_min = elevation_deg.min() - min_padding
    elev_max = elevation_deg.max() + min_padding
    
    total_span = elev_max - elev_min
    
    print(f"\nZOOMED view (using full image height for data):")
    print(f"  Elevation range: {elev_min:.1f} to {elev_max:.1f} deg")
    print(f"  Total span: {total_span:.1f} deg")
    print(f"  Image height: {height} pixels")
    print(f"  Vertical resolution: {total_span/height:.3f} deg per pixel")
    print(f"  This is much better than spreading 180 deg across {height} pixels!")
    
    # Map to pixels - using FULL image height for the zoomed range
    col = ((azimuth_deg + 180) / 360 * width).astype(int)
    col = np.clip(col, 0, width - 1)
    
    # Map elevation to full height range
    row = height - 1 - ((elevation_deg - elev_min) / (elev_max - elev_min) * height).astype(int)
    row = np.clip(row, 0, height - 1)
    
    # Create image
    range_image = np.zeros((height, width), dtype=np.float32)
    point_count = np.zeros((height, width), dtype=np.int32)
    
    for i in range(len(points)):
        r_idx, c_idx = row[i], col[i]
        point_count[r_idx, c_idx] += 1
        
        if range_image[r_idx, c_idx] == 0 or range_vals[i] < range_image[r_idx, c_idx]:
            range_image[r_idx, c_idx] = range_vals[i]
    
    # Statistics
    filled = np.count_nonzero(range_image)
    total = height * width
    avg_points_per_pixel = point_count[point_count > 0].mean()
    
    print(f"\nImage statistics:")
    print(f"  Total pixels: {total:,}")
    print(f"  Filled pixels: {filled:,} ({100*filled/total:.2f}%)")
    print(f"  Points per filled pixel: {avg_points_per_pixel:.1f}")
    
    # Check vertical usage
    rows_with_data = np.any(range_image > 0, axis=1)
    first_row = np.argmax(rows_with_data)
    last_row = len(rows_with_data) - 1 - np.argmax(rows_with_data[::-1])
    rows_used = last_row - first_row + 1
    
    print(f"\nVertical space usage:")
    print(f"  Rows with data: {rows_used}/{height} ({100*rows_used/height:.1f}%)")
    print(f"  Empty at top: {first_row} rows")
    print(f"  Empty at bottom: {height - last_row - 1} rows")
    
    if rows_used / height > 0.8:
        print("  Good! Most of image height is used for data")
    else:
        print("  Note: Could reduce min_padding to use even more vertical space")
    
    return range_image

def save_visualizations(range_image, prefix='range_zoom'):
    """Save visualizations"""
    
    import os
    
    if not (range_image > 0).any():
        print("\nERROR: No data in image!")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    mask = range_image > 0
    normalized = np.zeros_like(range_image)
    normalized[mask] = (range_image[mask] - range_image[mask].min()) / \
                       (range_image[mask].max() - range_image[mask].min())
    
    img_uint8 = (normalized * 255).astype(np.uint8)
    
    # Save grayscale
    cv2.imwrite(f'{prefix}_gray.png', img_uint8)
    print(f"\nSaved: {prefix}_gray.png")
    
    # Save colored
    img_color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(f'{prefix}_color.png', img_color)
    print(f"Saved: {prefix}_color.png")
    
    # Save raw
    np.save(f'{prefix}.npy', range_image)
    print(f"Saved: {prefix}.npy")

def main():
    bag_path = "/home/pinaka/dataset/AVMI/data/rosbag1210.db3"
    frame_number = 500
    output_prefix = 'results/images/range_zoom'
    
    # Higher resolution for better detail
    width = 2048   # Horizontal resolution
    height = 512   # Vertical resolution (more pixels for your ~30 deg FOV)
    
    if len(sys.argv) > 1:
        frame_number = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]
    
    print("="*60)
    print("Range Image Generator - ZOOMED VERSION")
    print("Maximizes vertical resolution by fitting data to full height")
    print("="*60)
    print(f"\nBag: {bag_path}")
    print(f"Frame: {frame_number}")
    print(f"Resolution: {width} x {height}")
    
    points = load_sample_from_bag(bag_path, frame_number)
    
    if points is None:
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} [frame_number] [output_prefix]")
        print("\nExamples:")
        print(f"  python3 {sys.argv[0]} 500")
        print(f"  python3 {sys.argv[0]} 500 results/images/frame_500")
        return
    
    print(f"\nLoaded {len(points)} points")
    
    # Create zoomed range image
    range_image = create_range_image_zoomed(
        points, 
        width=width,
        height=height,
        min_padding=2.0
    )
    
    # Save
    save_visualizations(range_image, output_prefix)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print("\nThis version ZOOMS IN on your LiDAR's actual field of view")
    print("instead of trying to show the full 180 degree sphere.")
    print("\nYour LiDAR scans about 30 degrees vertically.")
    print("Now those 30 degrees are spread across the full image height,")
    print("giving you much better vertical resolution and detail.")

if __name__ == "__main__":
    main()