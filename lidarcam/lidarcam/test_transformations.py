#!/usr/bin/env python3
"""
Coordinate Transformation Tester

This script will try different coordinate transformations and show you
which one produces the best results.

It will test 8 different combinations of coordinate flips to find
the correct transformation for your sensor setup.
"""

import os
import numpy as np
import cv2
import sqlite3
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import math


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def setup_camera_intrinsics():
    """Setup camera intrinsics from specs."""
    width = 640
    height = 480
    fov_deg = 90.0
    
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    return K, width, height


def get_sample_data(bag_path):
    """Load one sample image and LiDAR scan."""
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    # Get topic IDs
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {name: (id, msg_type) for id, name, msg_type in cursor.fetchall()}
    
    camera_topic = '/camera/left/image_raw'
    lidar_topic = '/lidar/points2'
    
    # Get one camera message
    camera_id, _ = topics[camera_topic]
    cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (camera_id,))
    camera_data = cursor.fetchone()[0]
    camera_msg = deserialize_message(camera_data, Image)
    
    # Get one LiDAR message
    lidar_id, _ = topics[lidar_topic]
    cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (lidar_id,))
    lidar_data = cursor.fetchone()[0]
    lidar_msg = deserialize_message(lidar_data, PointCloud2)
    
    conn.close()
    
    # Convert to usable formats
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
    
    points_list = []
    for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True):
        points_list.append([point[0], point[1], point[2]])
    
    points = np.array(points_list, dtype=np.float32)
    
    return img, points


def test_transformation(img, points, K, R, t, name, width, height):
    """Test a specific transformation and return statistics."""
    
    # Transform to camera frame
    Pc = (R @ points.T + t.reshape(3, 1)).T
    Z = Pc[:, 2]
    
    # Count points in front
    front = Z > 0.1
    num_front = front.sum()
    
    if num_front == 0:
        return {
            'name': name,
            'in_front': 0,
            'in_image': 0,
            'success': False,
            'image': None
        }
    
    # Keep points in front
    Pc = Pc[front]
    Z = Z[front]
    
    # Project to 2D
    uv = (K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    
    # Count points in image
    keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    num_in_image = keep.sum()
    
    # Create visualization if we have points
    output = img.copy()
    if num_in_image > 0:
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        Z_vis = Z[keep]
        
        # Color by depth
        Zn = (Z_vis - Z_vis.min()) / (Z_vis.max() - Z_vis.min() + 1e-6)
        Zc = (Zn * 255).astype(np.uint8)
        colors = cv2.applyColorMap(Zc, cv2.COLORMAP_JET)
        
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(output, (x, y), 2, tuple(int(a) for a in c[0]), -1)
    
    return {
        'name': name,
        'R': R.copy(),
        't': t.copy(),
        'in_front': num_front,
        'in_image': num_in_image,
        'percentage': (num_in_image / num_front * 100) if num_front > 0 else 0,
        'success': num_in_image > 0,
        'image': output
    }


def main():
    print_section("COORDINATE TRANSFORMATION TESTER")
    
    bag_path = '/home/pinaka/dataset/AVMI/mrzr_run_02_0-001.db3'
    
    if not os.path.exists(bag_path):
        print(f"Bag file not found: {bag_path}")
        return
    
    print("Loading sample data...")
    img, points = get_sample_data(bag_path)
    print(f"Loaded image: {img.shape}")
    print(f"Loaded {len(points)} LiDAR points")
    
    # Setup camera
    K, width, height = setup_camera_intrinsics()
    
    # Sensor positions from Unreal (in cm)
    lidar_pos_cm = np.array([-31.08193, -0.000001, 179.706462])
    camera_pos_cm = np.array([156.729406, 0.0, 103.342601])
    
    # Calculate base translation (in meters)
    relative_pos_m = (camera_pos_cm - lidar_pos_cm) / 100.0
    
    print(f"\nBase translation (Camera - LiDAR):")
    print(f"  X: {relative_pos_m[0]:.3f}m (forward)")
    print(f"  Y: {relative_pos_m[1]:.3f}m (sideways)")
    print(f"  Z: {relative_pos_m[2]:.3f}m (vertical)")
    
    print_section("TESTING 8 COORDINATE TRANSFORMATIONS")
    
    # Define 8 different transformations to test
    # These cover all possible coordinate system interpretations
    transformations = []
    
    # Original: Assume Unreal X,Y,Z → ROS X,-Y,Z (standard conversion)
    transformations.append({
        'name': 'Option 1: X, -Y, Z (Standard ROS)',
        't': np.array([relative_pos_m[0], -relative_pos_m[1], relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    # Flip X
    transformations.append({
        'name': 'Option 2: -X, -Y, Z (Flip forward)',
        't': np.array([-relative_pos_m[0], -relative_pos_m[1], relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    # Flip Z
    transformations.append({
        'name': 'Option 3: X, -Y, -Z (Flip up)',
        't': np.array([relative_pos_m[0], -relative_pos_m[1], -relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    # Flip both X and Z
    transformations.append({
        'name': 'Option 4: -X, -Y, -Z (Flip both)',
        't': np.array([-relative_pos_m[0], -relative_pos_m[1], -relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    # Don't flip Y (maybe Unreal and ROS Y are same)
    transformations.append({
        'name': 'Option 5: X, Y, Z (No Y flip)',
        't': np.array([relative_pos_m[0], relative_pos_m[1], relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    transformations.append({
        'name': 'Option 6: -X, Y, Z',
        't': np.array([-relative_pos_m[0], relative_pos_m[1], relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    transformations.append({
        'name': 'Option 7: X, Y, -Z',
        't': np.array([relative_pos_m[0], relative_pos_m[1], -relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    transformations.append({
        'name': 'Option 8: -X, Y, -Z',
        't': np.array([-relative_pos_m[0], relative_pos_m[1], -relative_pos_m[2]]),
        'R': np.eye(3)
    })
    
    # Test all transformations
    results = []
    for i, trans in enumerate(transformations, 1):
        print(f"Testing {trans['name']}...")
        result = test_transformation(
            img.copy(), points, K, trans['R'], trans['t'], 
            trans['name'], width, height
        )
        results.append(result)
        
        print(f"  Points in front: {result['in_front']}")
        print(f"  Points in image: {result['in_image']}")
        if result['success']:
            print(f"  ✅ SUCCESS! {result['percentage']:.1f}% of front points visible")
        else:
            print(f"  ❌ Failed - no points in image")
    
    # Find best result
    print_section("RESULTS SUMMARY")
    
    best_result = max(results, key=lambda x: x['in_image'])
    
    print("All options ranked by points in image:\n")
    sorted_results = sorted(results, key=lambda x: x['in_image'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        status = "✅ BEST" if result == best_result and result['success'] else "✅" if result['success'] else "❌"
        print(f"{i}. {result['name']}")
        print(f"   {status} {result['in_image']} points in image ({result['percentage']:.1f}%)")
        print(f"   Translation: {result['t']}")
        print()
    
    if best_result['success']:
        print_section("RECOMMENDED SOLUTION")
        print(f"✅ Use: {best_result['name']}\n")
        print(f"Translation vector:")
        print(f"  {best_result['t']}\n")
        print(f"In your ugv_fusion_node.py, update line ~239:")
        print(f"  self.t_l2c = np.array([{best_result['t'][0]:.6f}, "
              f"{best_result['t'][1]:.6f}, {best_result['t'][2]:.6f}], dtype=np.float32)")
        
        # Save best visualization
        if best_result['image'] is not None:
            output_path = '/home/claude/best_transformation.jpg'
            cv2.imwrite(output_path, best_result['image'])
            print(f"\n📸 Best result saved to: {output_path}")
        
        # Save all visualizations for comparison
        print("\n📸 Saving all results for comparison...")
        for i, result in enumerate(results, 1):
            if result['image'] is not None:
                output_path = f'/home/claude/option_{i}_transformation.jpg'
                cv2.imwrite(output_path, result['image'])
                print(f"  Option {i}: {output_path}")
        
    else:
        print_section("NO SOLUTION FOUND")
        print("None of the coordinate transformations produced points in the image.")
        print("\nPossible issues:")
        print("1. LiDAR and camera FOV don't overlap")
        print("2. Sensor positions in Unreal are incorrect")
        print("3. Camera intrinsics (FOV) are wrong")
        print("4. LiDAR is pointing in completely wrong direction")
        print("\nNext steps:")
        print("- Verify sensor positions in Unreal Engine")
        print("- Check if LiDAR and camera are facing same direction")
        print("- Verify camera FOV is actually 90 degrees")


if __name__ == '__main__':
    main()
