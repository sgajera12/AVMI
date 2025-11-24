#!/usr/bin/env python3
"""
Advanced Diagnostic Tool for LiDAR-Camera Fusion

This script will:
1. Show WHERE points are projecting (even if outside image)
2. Test different FOV values
3. Analyze LiDAR scan pattern
4. Check if sensors actually see the same area
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def get_sample_data(bag_path):
    """Load one sample image and LiDAR scan."""
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {name: (id, msg_type) for id, name, msg_type in cursor.fetchall()}
    
    camera_topic = '/camera/image_raw'
    lidar_topic = '/lidar/points2'
    
    camera_id, _ = topics[camera_topic]
    cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (camera_id,))
    camera_data = cursor.fetchone()[0]
    camera_msg = deserialize_message(camera_data, Image)
    
    lidar_id, _ = topics[lidar_topic]
    cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (lidar_id,))
    lidar_data = cursor.fetchone()[0]
    lidar_msg = deserialize_message(lidar_data, PointCloud2)
    
    conn.close()
    
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
    
    points_list = []
    for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True):
        points_list.append([point[0], point[1], point[2]])
    
    points = np.array(points_list, dtype=np.float32)
    
    return img, points


def analyze_lidar_scan(points):
    """Analyze the LiDAR scan pattern to understand its coverage."""
    print_section("LIDAR SCAN ANALYSIS")
    
    print(f"Total points: {len(points)}")
    print(f"\nSpatial distribution:")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
    
    # Calculate distance and angles
    distances = np.linalg.norm(points, axis=1)
    print(f"\nDistance statistics:")
    print(f"  Min: {distances.min():.2f} m")
    print(f"  Max: {distances.max():.2f} m")
    print(f"  Mean: {distances.mean():.2f} m")
    print(f"  Median: {np.median(distances):.2f} m")
    
    # Horizontal angle (in XY plane)
    horizontal_angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
    print(f"\nHorizontal angle distribution:")
    print(f"  Range: [{horizontal_angles.min():.1f}°, {horizontal_angles.max():.1f}°]")
    print(f"  Coverage: {horizontal_angles.max() - horizontal_angles.min():.1f}°")
    
    # Count points in different sectors
    front = np.sum((horizontal_angles > -45) & (horizontal_angles < 45))
    back = np.sum((horizontal_angles < -135) | (horizontal_angles > 135))
    left = np.sum((horizontal_angles > 45) & (horizontal_angles < 135))
    right = np.sum((horizontal_angles > -135) & (horizontal_angles < -45))
    
    print(f"\nPoint distribution by direction:")
    print(f"  Front (-45° to +45°): {front} points ({front/len(points)*100:.1f}%)")
    print(f"  Left (45° to 135°): {left} points ({left/len(points)*100:.1f}%)")
    print(f"  Back: {back} points ({back/len(points)*100:.1f}%)")
    print(f"  Right: {right} points ({right/len(points)*100:.1f}%)")
    
    if front < len(points) * 0.1:
        print(f"\n⚠️  WARNING: Only {front/len(points)*100:.1f}% of points are in front!")
        print(f"   LiDAR might not be facing forward or has limited forward coverage.")
    
    # Vertical angle
    horizontal_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    vertical_angles = np.arctan2(points[:, 2], horizontal_dist) * 180 / np.pi
    print(f"\nVertical angle distribution:")
    print(f"  Range: [{vertical_angles.min():.1f}°, {vertical_angles.max():.1f}°]")
    
    return horizontal_angles, vertical_angles


def test_with_fov(img, points, fov_deg, t):
    """Test projection with a specific FOV."""
    width, height = 640, 480
    
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
    
    # Transform to camera frame (no rotation)
    R = np.eye(3)
    Pc = (R @ points.T + t.reshape(3, 1)).T
    Z = Pc[:, 2]
    
    # Keep points in front
    front = Z > 0.1
    if front.sum() == 0:
        return 0, K
    
    Pc = Pc[front]
    Z = Z[front]
    
    # Project
    uv = (K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    
    # Count points in image
    keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return keep.sum(), K


def analyze_projection_locations(img, points, K, t):
    """Analyze WHERE points are projecting, even if outside image."""
    print_section("PROJECTION LOCATION ANALYSIS")
    
    width, height = 640, 480
    
    # Transform to camera frame
    R = np.eye(3)
    Pc = (R @ points.T + t.reshape(3, 1)).T
    Z = Pc[:, 2]
    
    # Keep points in front
    front = Z > 0.1
    print(f"Points in front of camera (Z > 0.1m): {front.sum()} / {len(points)}")
    
    if front.sum() == 0:
        print("❌ No points in front of camera!")
        return
    
    Pc = Pc[front]
    Z = Z[front]
    
    # Project ALL points (even if outside image)
    uv = (K @ (Pc.T / Z)).T
    u, v = uv[:, 0], uv[:, 1]
    
    print(f"\nProjected pixel coordinates:")
    print(f"  U range: [{u.min():.1f}, {u.max():.1f}] (image width: 0-{width})")
    print(f"  V range: [{v.min():.1f}, {v.max():.1f}] (image height: 0-{height})")
    
    # Analyze distribution
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    left_of_image = u < 0
    right_of_image = u >= width
    above_image = v < 0
    below_image = v >= height
    
    print(f"\nPoint locations relative to image:")
    print(f"  Inside image: {inside.sum()} ({inside.sum()/len(u)*100:.1f}%)")
    print(f"  Left of image: {left_of_image.sum()} ({left_of_image.sum()/len(u)*100:.1f}%)")
    print(f"  Right of image: {right_of_image.sum()} ({right_of_image.sum()/len(u)*100:.1f}%)")
    print(f"  Above image: {above_image.sum()} ({above_image.sum()/len(u)*100:.1f}%)")
    print(f"  Below image: {below_image.sum()} ({below_image.sum()/len(u)*100:.1f}%)")
    
    # Create visualization showing projection beyond image bounds
    scale = 0.5  # Scale down for visualization
    vis_width = int(width * 3)  # 3x wider to show points outside
    vis_height = int(height * 3)  # 3x taller
    vis_img = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw image boundary
    img_left = vis_width // 2 - width // 2
    img_top = vis_height // 2 - height // 2
    cv2.rectangle(vis_img, (img_left, img_top), (img_left + width, img_top + height), (0, 255, 0), 2)
    cv2.putText(vis_img, "Camera Image", (img_left + 10, img_top + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Plot projected points (offset to center of visualization)
    u_vis = (u + vis_width // 2 - width // 2).astype(np.int32)
    v_vis = (v + vis_height // 2 - height // 2).astype(np.int32)
    
    # Keep only points that fit in visualization
    vis_keep = (u_vis >= 0) & (u_vis < vis_width) & (v_vis >= 0) & (v_vis < vis_height)
    
    for (x, y) in zip(u_vis[vis_keep], v_vis[vis_keep]):
        color = (0, 0, 255) if (x >= img_left and x < img_left + width and 
                                 y >= img_top and y < img_top + height) else (255, 0, 0)
        cv2.circle(vis_img, (x, y), 1, color, -1)
    
    output_path = '/home/ros2_ws/src/lidarcam/lidarcam/ugv/projection_map.png'
    cv2.imwrite(output_path, vis_img)
    print(f"\n📸 Projection map saved to: {output_path}")
    print(f"   Green box = camera image")
    print(f"   Red dots = points outside image")
    print(f"   Blue dots = points inside image")
    
    # Check if points are clustered in a specific area
    if inside.sum() == 0:
        u_mean = u.mean()
        v_mean = v.mean()
        print(f"\n💡 INSIGHT: Points are centered at pixel ({u_mean:.1f}, {v_mean:.1f})")
        
        if abs(u_mean - width/2) > width:
            print(f"   Points are {abs(u_mean - width/2)/width:.1f}x image width away horizontally!")
            print(f"   This suggests a MAJOR calibration or FOV issue.")
        
        if abs(v_mean - height/2) > height:
            print(f"   Points are {abs(v_mean - height/2)/height:.1f}x image height away vertically!")


def test_different_fovs(img, points, t):
    """Test different FOV values to see if camera FOV is wrong."""
    print_section("TESTING DIFFERENT FOV VALUES")
    
    fov_values = [30, 45, 60, 75, 90, 110, 130, 150, 170]
    results = []
    
    print("Testing various FOV values:")
    for fov in fov_values:
        num_points, K = test_with_fov(img, points, fov, t)
        results.append((fov, num_points))
        status = "✅" if num_points > 0 else "❌"
        print(f"  {status} FOV {fov:3d}°: {num_points:5d} points in image")
    
    best_fov, best_count = max(results, key=lambda x: x[1])
    
    if best_count > 0:
        print(f"\n💡 SOLUTION FOUND!")
        print(f"   FOV {best_fov}° produces {best_count} points in image")
        print(f"\n   Try changing the FOV in ugv_fusion_node.py:")
        print(f"   Line ~216, change:")
        print(f"   fov_degrees = 90.0")
        print(f"   to:")
        print(f"   fov_degrees = {best_fov}")
        return best_fov
    else:
        print(f"\n❌ No FOV value produced points in image")
        print(f"   The problem is not just FOV - it's likely:")
        print(f"   1. Sensor positions are wrong")
        print(f"   2. Sensors are not facing same direction")
        print(f"   3. LiDAR and camera FOVs don't overlap at all")
        return None


def create_top_down_view(points, t):
    """Create a top-down view showing LiDAR coverage and camera FOV."""
    print_section("TOP-DOWN VIEW ANALYSIS")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot LiDAR points (top-down, XY plane)
    ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5, c='blue', label='LiDAR points')
    
    # Mark LiDAR position
    ax.plot(0, 0, 'ro', markersize=10, label='LiDAR')
    
    # Mark camera position (from translation)
    cam_x, cam_y = t[0], t[1]
    ax.plot(cam_x, cam_y, 'go', markersize=10, label='Camera')
    
    # Draw camera FOV cone (90 degrees, forward direction)
    fov_angle = 90  # degrees
    fov_distance = 20  # meters
    
    # Camera looks in +X direction (forward) from its position
    left_angle = np.radians(fov_angle / 2)
    right_angle = np.radians(-fov_angle / 2)
    
    # FOV cone points
    fov_left_x = cam_x + fov_distance * np.cos(left_angle)
    fov_left_y = cam_y + fov_distance * np.sin(left_angle)
    fov_right_x = cam_x + fov_distance * np.cos(right_angle)
    fov_right_y = cam_y + fov_distance * np.sin(right_angle)
    
    # Draw FOV cone
    ax.plot([cam_x, fov_left_x], [cam_y, fov_left_y], 'g--', alpha=0.5)
    ax.plot([cam_x, fov_right_x], [cam_y, fov_right_y], 'g--', alpha=0.5)
    ax.plot([fov_left_x, fov_right_x], [fov_left_y, fov_right_y], 'g--', alpha=0.5)
    
    # Fill FOV cone
    fov_cone = np.array([[cam_x, cam_y], [fov_left_x, fov_left_y], [fov_right_x, fov_right_y]])
    ax.fill(fov_cone[:, 0], fov_cone[:, 1], 'green', alpha=0.2, label='Camera FOV')
    
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.set_title('Top-Down View: LiDAR Points and Camera FOV')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    output_path = '/home/ros2_ws/src/lidarcam/lidarcam/ugv/projection_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Top-down view saved to: {output_path}")
    print(f"\nThis shows:")
    print(f"  - Blue dots = LiDAR scan points")
    print(f"  - Red circle = LiDAR position (0, 0)")
    print(f"  - Green circle = Camera position ({cam_x:.2f}, {cam_y:.2f})")
    print(f"  - Green cone = Camera 90° FOV")
    print(f"\nIf blue dots don't overlap with green cone:")
    print(f"  → LiDAR and camera aren't seeing the same area!")


def main():
    print_section("ADVANCED LIDAR-CAMERA DIAGNOSTIC")
    
    bag_path = '/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3'
    
    if not os.path.exists(bag_path):
        print(f"❌ Bag file not found: {bag_path}")
        return
    
    print("Loading data...")
    img, points = get_sample_data(bag_path)
    print(f"✅ Loaded image: {img.shape}")
    print(f"✅ Loaded {len(points)} LiDAR points")
    
    # Analyze LiDAR scan pattern
    h_angles, v_angles = analyze_lidar_scan(points)
    
    # Setup calibration (use standard transformation)
    lidar_pos_cm = np.array([-31.08193, -0.000001, 179.706462])
    camera_pos_cm = np.array([156.729406, 0.0, 103.342601])
    relative_pos_m = (camera_pos_cm - lidar_pos_cm) / 100.0
    t = np.array([relative_pos_m[0], -relative_pos_m[1], relative_pos_m[2]], dtype=np.float32)
    
    # Setup camera with 90° FOV
    width, height = 640, 480
    fov_deg = 90.0
    fov_rad = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    K = np.array([[fx, 0, width/2], [0, fx, height/2], [0, 0, 1]], dtype=np.float32)
    
    # Analyze where points project
    analyze_projection_locations(img, points, K, t)
    
    # Test different FOVs
    best_fov = test_different_fovs(img, points, t)
    
    # Create top-down view
    create_top_down_view(points, t)
    
    print_section("DIAGNOSIS SUMMARY")
    
    print("Based on the analysis:")
    print("\n1. Check the projection_map.jpg:")
    print("   - Shows where LiDAR points project (even if off-screen)")
    print("   - Green box = camera image area")
    print("   - Red/Blue dots = projected LiDAR points")
    
    print("\n2. Check the top_down_view.png:")
    print("   - Shows if LiDAR coverage overlaps with camera FOV")
    print("   - Blue points should be inside green cone")
    
    if best_fov:
        print(f"\n3. ✅ SOLUTION: Use FOV = {best_fov}° instead of 90°")
    else:
        print("\n3. ⚠️  No FOV value works - fundamental calibration issue")
        print("\nNext steps:")
        print("  - Verify sensor positions in Unreal Engine")
        print("  - Check if LiDAR and camera face same direction")
        print("  - Confirm camera FOV setting in Unreal")
        print("  - Check if sensors are on same vehicle/robot")


if __name__ == '__main__':
    main()
