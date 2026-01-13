#!/usr/bin/env python3
"""
SEMANTIC LABEL GENERATOR - TIMESTAMP SYNCHRONIZED VERSION

Properly synchronizes camera and LiDAR by timestamp!
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


# Your calibration values (adjust these!)
TX = -0.261
TY = 0.0
TZ = 2.020
PITCH = -72.0
YAW = 0.0
ROLL = 90.0
FOV = 90.0

# Color to class mapping
COLOR_TO_CLASS = {
    (139, 69, 19): 1,    # Brown → ground
    (0, 255, 0): 2,      # Green → tree
    (255, 0, 0): 3,      # Red → rock
    (243, 156, 18): 1,   # Orange → ground
}

def color_to_class_id(r, g, b, tolerance=30):
    """Map RGB to class ID"""
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
    """Project LiDAR points to image"""
    
    h, w = img_shape[:2]
    
    R = get_rotation_matrix(PITCH, YAW, ROLL)
    t = np.array([TX, TY, TZ], dtype=np.float32)
    
    xyz = points[:, :3]
    rgb = points[:, 3:6].astype(np.uint8)
    
    Pc = (R @ xyz.T + t.reshape(3, 1)).T
    Z = Pc[:, 2]
    front = Z > 0.1
    
    if front.sum() == 0:
        return None
    
    Pc = Pc[front]
    Z = Z[front]
    rgb = rgb[front]
    
    focal = (w / 2.0) / math.tan(math.radians(FOV / 2.0))
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.zeros(5, dtype=np.float32)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    
    image_points, _ = cv2.projectPoints(Pc, rvec, tvec, K, dist_coeffs)
    image_points = image_points.reshape(-1, 2)
    u, v = image_points[:, 0], image_points[:, 1]
    
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    
    if keep.sum() == 0:
        return None
    
    u = u[keep].astype(np.int32)
    v = v[keep].astype(np.int32)
    rgb = rgb[keep]
    
    return list(zip(u, v, rgb[:, 0], rgb[:, 1], rgb[:, 2]))


def create_semantic_mask(img_shape, projected_points, dilation_size=5, fill_gaps=True):
    """Create semantic mask from sparse points"""
    
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    confidence = np.zeros((h, w), dtype=np.uint8)
    
    for u, v, r, g, b in projected_points:
        class_id = color_to_class_id(r, g, b)
        cv2.circle(mask, (u, v), dilation_size, class_id, -1)
        cv2.circle(confidence, (u, v), dilation_size, 255, -1)
    
    if fill_gaps:
        unknown = (mask == 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        small_gaps = cv2.erode(unknown, kernel, iterations=1)
        mask[small_gaps == 0] = mask_dilated[small_gaps == 0]
    
    return mask, confidence


def visualize_results(img, mask, confidence, projected_points):
    """Create visualization"""
    
    h, w = img.shape[:2]
    
    class_colors = {
        0: [0, 0, 0],
        1: [139, 69, 19],
        2: [0, 255, 0],
        3: [255, 0, 0],
    }
    
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask_colored[mask == class_id] = color
    
    overlay = img.copy()
    overlay = cv2.addWeighted(overlay, 0.5, mask_colored, 0.5, 0)
    
    for u, v, r, g, b in projected_points:
        cv2.circle(overlay, (u, v), 2, (int(b), int(g), int(r)), -1)
    
    result = np.hstack([img, mask_colored, overlay])
    
    cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Overlay", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result


def process_bag_synchronized(bag_path, output_dir, num_frames=50, dilation=5, max_time_diff_ns=100000000):
    """Process bag with proper timestamp synchronization"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("SEMANTIC LABEL GENERATOR - TIMESTAMP SYNCHRONIZED")
    print("="*70)
    print(f"Bag: {bag_path}")
    print(f"Output: {output_dir}")
    print(f"Max time diff: {max_time_diff_ns/1e6:.1f} ms")
    print("="*70)
    
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name FROM topics")
    topics = {name: id for id, name in cursor.fetchall()}
    
    camera_id = topics['/camera/left/image_raw']
    lidar_id = topics['/lidar/points2']
    
    # Get ALL camera timestamps
    cursor.execute(f"SELECT timestamp FROM messages WHERE topic_id = {camera_id} ORDER BY timestamp")
    camera_times = [row[0] for row in cursor.fetchall()]
    
    # Get ALL lidar timestamps
    cursor.execute(f"SELECT timestamp FROM messages WHERE topic_id = {lidar_id} ORDER BY timestamp")
    lidar_times = np.array([row[0] for row in cursor.fetchall()])
    
    print(f"\nCamera frames: {len(camera_times)}")
    print(f"LiDAR scans: {len(lidar_times)}")
    
    # Sample camera frames evenly
    if num_frames >= len(camera_times):
        selected_cam_indices = list(range(len(camera_times)))
    else:
        step = len(camera_times) / num_frames
        selected_cam_indices = [int(i * step) for i in range(num_frames)]
    
    print(f"Sampling {len(selected_cam_indices)} camera frames")
    print("="*70)
    
    bridge = CvBridge()
    success = 0
    
    for idx, cam_idx in enumerate(selected_cam_indices):
        cam_time = camera_times[cam_idx]
        
        # Find closest LiDAR timestamp
        time_diffs = np.abs(lidar_times - cam_time)
        closest_lidar_idx = np.argmin(time_diffs)
        time_diff = time_diffs[closest_lidar_idx]
        
        if time_diff > max_time_diff_ns:
            print(f"\nFrame {idx+1}/{len(selected_cam_indices)}: Time diff too large ({time_diff/1e6:.1f}ms), skipping")
            continue
        
        print(f"\nFrame {idx+1}/{len(selected_cam_indices)}: cam={cam_idx}, lidar={closest_lidar_idx}, Δt={time_diff/1e6:.1f}ms")
        
        try:
            # Get camera frame
            cursor.execute(
                f"SELECT data FROM messages WHERE topic_id = {camera_id} AND timestamp = {cam_time}"
            )
            camera_data = cursor.fetchone()[0]
            camera_msg = deserialize_message(camera_data, Image)
            img = bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
            
            # Get LiDAR scan at closest_lidar_idx
            lidar_time = lidar_times[closest_lidar_idx]
            cursor.execute(
                f"SELECT data FROM messages WHERE topic_id = {lidar_id} AND timestamp = {lidar_time}"
            )
            lidar_data = cursor.fetchone()[0]
            lidar_msg = deserialize_message(lidar_data, PointCloud2)
            
            # Parse points
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
                print(f"  ⚠ Too few points: {len(projected) if projected else 0}")
                continue
            
            print(f"  ✓ {len(projected)} points projected")
            
            # Create mask
            mask, confidence = create_semantic_mask(img.shape, projected, dilation, True)
            
            # Class distribution
            unique, counts = np.unique(mask[mask > 0], return_counts=True)
            print(f"  Classes: {dict(zip(unique, counts))}")
            
            # Visualize
            result = visualize_results(img, mask, confidence, projected)
            
            # Save
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
    print(f"✓ COMPLETE! {success}/{len(selected_cam_indices)} frames")
    print("="*70)
    print(f"\nClass mapping: 0=bg, 1=ground, 2=tree, 3=rock")
    print(f"Ready to train U-Net!")


if __name__ == '__main__':
    import sys
    
    bag_path = '/home/pinaka/dataset/AVMI/data/rosbag1210.db3'
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './synced_dataset'
    dilation = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    process_bag_synchronized(bag_path, output_dir, num_frames, dilation)
