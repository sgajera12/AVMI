#!/usr/bin/env python3
"""
PRACTICAL SEMANTIC LABEL GENERATOR

Takes approximately aligned LiDAR projection and creates usable semantic masks!

Key insight: Don't need perfect alignment - use dilation and filling!
"""

import numpy as np
import cv2
import sqlite3
import struct
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import math
from pathlib import Path


# Your calibration values (from tuner - adjust these!)
TX = -0.261
TY = 0.0
TZ = 2.020
PITCH = -72.0
YAW = 0.0
ROLL = 90.0
FOV = 90.0

# Color to class mapping (from your data)
COLOR_TO_CLASS = {
    (139, 69, 19): 1,    # Brown → ground
    (0, 255, 0): 2,      # Green → tree
    (255, 0, 0): 3,      # Red → rock
    (243, 156, 18): 1,   # Orange → ground (tree trunk)
}

def color_to_class_id(r, g, b, tolerance=30):
    """Map RGB to class ID with tolerance"""
    min_dist = float('inf')
    best_class = 0
    
    for (ref_r, ref_g, ref_b), class_id in COLOR_TO_CLASS.items():
        dist = abs(r - ref_r) + abs(g - ref_g) + abs(b - ref_b)
        if dist < min_dist:
            min_dist = dist
            best_class = class_id
    
    return best_class if min_dist < tolerance else 0


def get_rotation_matrix(pitch, yaw, roll):
    """Create rotation matrix"""
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)
    
    R_pitch = np.array([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])
    
    R_yaw = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad), math.cos(roll_rad)]
    ])
    
    return (R_yaw @ R_pitch @ R_roll).astype(np.float32)


def project_points(points, img_shape):
    """Project LiDAR points to image with current calibration"""
    
    h, w = img_shape[:2]
    
    # Get transformation
    R = get_rotation_matrix(PITCH, YAW, ROLL)
    t = np.array([TX, TY, TZ], dtype=np.float32)
    
    # Extract XYZ and RGB
    xyz = points[:, :3]
    rgb = points[:, 3:6].astype(np.uint8)
    
    # Transform to camera frame
    Pc = (R @ xyz.T + t.reshape(3, 1)).T
    Z = Pc[:, 2]
    front = Z > 0.1
    
    if front.sum() == 0:
        return None
    
    Pc = Pc[front]
    Z = Z[front]
    rgb = rgb[front]
    
    # Camera intrinsics
    focal = (w / 2.0) / math.tan(math.radians(FOV / 2.0))
    K = np.array([
        [focal, 0, w/2],
        [0, focal, h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Project
    dist_coeffs = np.zeros(5, dtype=np.float32)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    
    image_points, _ = cv2.projectPoints(Pc, rvec, tvec, K, dist_coeffs)
    image_points = image_points.reshape(-1, 2)
    u, v = image_points[:, 0], image_points[:, 1]
    
    # Keep points in image
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    if keep.sum() == 0:
        return None
    
    u = u[keep].astype(np.int32)
    v = v[keep].astype(np.int32)
    rgb = rgb[keep]
    
    return list(zip(u, v, rgb[:, 0], rgb[:, 1], rgb[:, 2]))


def create_semantic_mask(img_shape, projected_points, dilation_size=5, fill_gaps=True):
    """
    Create semantic mask from sparse projected points
    
    Args:
        img_shape: Image shape (H, W, 3)
        projected_points: List of (u, v, r, g, b)
        dilation_size: How much to expand each point (pixels)
        fill_gaps: Whether to fill gaps with nearest neighbor
    
    Returns:
        mask: (H, W) semantic mask with class IDs
        confidence: (H, W) confidence map (0-255)
    """
    
    h, w = img_shape[:2]
    
    # Initialize
    mask = np.zeros((h, w), dtype=np.uint8)
    confidence = np.zeros((h, w), dtype=np.uint8)
    
    # Step 1: Place points with dilation
    for u, v, r, g, b in projected_points:
        class_id = color_to_class_id(r, g, b)
        
        # Dilate point into small circle
        cv2.circle(mask, (u, v), dilation_size, class_id, -1)
        cv2.circle(confidence, (u, v), dilation_size, 255, -1)
    
    print(f"  After dilation: {(mask > 0).sum()} pixels labeled")
    
    # Step 2: Fill gaps (optional)
    if fill_gaps:
        # Use inpainting or nearest neighbor to fill
        # Create mask of unknown regions
        unknown = (mask == 0).astype(np.uint8) * 255
        
        # Dilate known regions to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Fill only small gaps
        small_gaps = cv2.erode(unknown, kernel, iterations=1)
        mask[small_gaps == 0] = mask_dilated[small_gaps == 0]
        
        print(f"  After gap filling: {(mask > 0).sum()} pixels labeled")
    
    return mask, confidence


def visualize_results(img, mask, confidence, projected_points):
    """Create visualization"""
    
    h, w = img.shape[:2]
    
    # Color map for classes
    class_colors = {
        0: [0, 0, 0],        # Background - black
        1: [139, 69, 19],    # Ground - brown
        2: [0, 255, 0],      # Tree - green
        3: [255, 0, 0],      # Rock - red
    }
    
    # Create colored mask
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask_colored[mask == class_id] = color
    
    # Overlay on image
    overlay = img.copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
    
    # Draw projected points
    for u, v, r, g, b in projected_points:
        cv2.circle(overlay, (u, v), 2, (int(b), int(g), int(r)), -1)
    
    # Create composite
    result = np.hstack([
        img,
        mask_colored,
        overlay
    ])
    
    # Add labels
    cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Overlay", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result


def process_bag(bag_path, output_dir, num_frames=50, dilation=5):
    """Process bag and generate labels"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("SEMANTIC LABEL GENERATOR")
    print("="*70)
    print(f"Bag: {bag_path}")
    print(f"Output: {output_dir}")
    print(f"Calibration: TX={TX:.3f}, TY={TY:.3f}, TZ={TZ:.3f}")
    print(f"             Pitch={PITCH:.1f}°, Yaw={YAW:.1f}°, Roll={ROLL:.1f}°")
    print(f"             FOV={FOV:.1f}°")
    print(f"Dilation: {dilation}px")
    print("="*70)
    
    # Open bag
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name FROM topics")
    topics = {name: id for id, name in cursor.fetchall()}
    
    camera_id = topics['/camera/left/image_raw']
    lidar_id = topics['/lidar/points2']
    
    # Count total frames
    cursor.execute(f"SELECT COUNT(*) FROM messages WHERE topic_id = {camera_id}")
    total_frames = cursor.fetchone()[0]
    
    print(f"\nTotal frames in bag: {total_frames}")
    print(f"Sampling {num_frames} frames evenly across duration")
    
    # Calculate frame indices to sample (evenly spaced)
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        # Sample evenly across the bag
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    print(f"Frame indices: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
    print("="*70)
    
    bridge = CvBridge()
    
    success = 0
    
    for idx, frame_idx in enumerate(frame_indices):
        print(f"\nProcessing frame {idx+1}/{num_frames} (bag frame {frame_idx})...")
        
        try:
            # Get camera
            cursor.execute(
                f"SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET {frame_idx}",
                (camera_id,)
            )
            camera_result = cursor.fetchone()
            if not camera_result:
                print(f"  ⚠ Frame {frame_idx} not found")
                continue
            
            camera_msg = deserialize_message(camera_result[0], Image)
            img = bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
            
            # Get LiDAR (try to get closest in time)
            cursor.execute(
                f"SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET {frame_idx}",
                (lidar_id,)
            )
            lidar_result = cursor.fetchone()
            if not lidar_result:
                print(f"  ⚠ LiDAR frame {frame_idx} not found")
                continue
            
            lidar_msg = deserialize_message(lidar_result[0], PointCloud2)
            
            # Parse points with RGB
            points_list = []
            point_step = lidar_msg.point_step
            data = lidar_msg.data
            
            for i in range(0, min(len(data), 50000), point_step):
                try:
                    x = struct.unpack_from('f', data, i)[0]
                    y = struct.unpack_from('f', data, i + 4)[0]
                    z = struct.unpack_from('f', data, i + 8)[0]
                    rgb_int = struct.unpack_from('I', data, i + 12)[0]
                    
                    r = (rgb_int >> 16) & 0xFF
                    g = (rgb_int >> 8) & 0xFF
                    b = rgb_int & 0xFF
                    
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        points_list.append([x, y, z, r, g, b])
                except:
                    continue
            
            points = np.array(points_list, dtype=np.float32)
            
            # Project
            projected = project_points(points, img.shape)
            
            if projected is None or len(projected) < 50:
                print(f"  ⚠ Too few points projected: {len(projected) if projected else 0}")
                continue
            
            print(f"  ✓ Projected {len(projected)} points")
            
            # Create mask
            mask, confidence = create_semantic_mask(
                img.shape, projected, 
                dilation_size=dilation, 
                fill_gaps=True
            )
            
            # Check class distribution
            unique, counts = np.unique(mask, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"  Classes: ", end="")
            for cls, cnt in class_dist.items():
                if cls > 0:
                    print(f"{cls}:{cnt} ", end="")
            print()
            
            # Visualize
            result = visualize_results(img, mask, confidence, projected)
            
            # Save with sequential numbering
            cv2.imwrite(str(output_dir / f"image_{idx:04d}.jpg"), img)
            cv2.imwrite(str(output_dir / f"mask_{idx:04d}.png"), mask)
            cv2.imwrite(str(output_dir / f"confidence_{idx:04d}.png"), confidence)
            cv2.imwrite(str(output_dir / f"result_{idx:04d}.jpg"), result)
            
            success += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    conn.close()
    
    print("\n" + "="*70)
    print(f"✓ COMPLETE! Generated {success}/{num_frames} labeled frames")
    print("="*70)
    print(f"\nOutput files in {output_dir}:")
    print(f"  image_XXXX.jpg - Original camera images")
    print(f"  mask_XXXX.png - Semantic masks (class IDs 0-3)")
    print(f"  result_XXXX.jpg - Visualization (original | mask | overlay)")
    print(f"\nClass mapping:")
    print(f"  0 = background (unlabeled)")
    print(f"  1 = ground (brown)")
    print(f"  2 = tree/vegetation (green)")
    print(f"  3 = rock (red)")
    print(f"\nReady to train U-Net!")


if __name__ == '__main__':
    import sys
    
    bag_path = '/home/pinaka/dataset/AVMI/data/rosbag1210.db3'
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = './auto_labeled_dataset'
    
    dilation = 5  # Adjust this if points too sparse/dense
    
    print("\nUSAGE:")
    print("  python3 generate_labels.py [output_dir] [dilation_size]")
    print("\nEXAMPLE:")
    print("  python3 generate_labels.py ./my_dataset 7")
    print("")
    
    if len(sys.argv) > 2:
        dilation = int(sys.argv[2])
    
    process_bag(bag_path, output_dir, num_frames=50, dilation=dilation)