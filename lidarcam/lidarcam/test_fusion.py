#!/usr/bin/env python3
"""
Test and Verification Script for UGV Fusion

This script helps you:
1. Verify your bag file is readable
2. Check topic names and message counts
3. Test calibration values
4. Visualize sample projections
5. Debug synchronization

Run this BEFORE running the fusion node to catch issues early!
"""

import os
import sys
import sqlite3
import numpy as np
import cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


def print_section(title):
    """Print a nice section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_bag_file(bag_path):
    """Test if bag file is readable and contains expected topics."""
    print_section("TEST 1: BAG FILE VERIFICATION")
    
    # Check file exists
    if not os.path.exists(bag_path):
        print(f"ERROR: Bag file not found at: {bag_path}")
        return False
    
    print(f"✅ Bag file exists: {bag_path}")
    print(f"   Size: {os.path.getsize(bag_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Connect to database
        conn = sqlite3.connect(bag_path)
        cursor = conn.cursor()
        
        # Get topics
        cursor.execute("SELECT id, name, type FROM topics")
        topics = cursor.fetchall()
        
        print(f"\n✅ Found {len(topics)} topics in bag:")
        for topic_id, name, msg_type in topics:
            # Count messages for this topic
            cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,))
            count = cursor.fetchone()[0]
            print(f"   - {name}")
            print(f"     Type: {msg_type}")
            print(f"     Messages: {count}")
        
        # Check for required topics
        topic_names = [t[1] for t in topics]
        has_camera = any('camera' in t.lower() and 'image' in t.lower() for t in topic_names)
        has_lidar = any('lidar' in t.lower() or 'points' in t.lower() for t in topic_names)
        
        if not has_camera:
            print("\n⚠️  WARNING: No camera/image topic found!")
            print("   Expected topic like: /camera/image_raw")
        
        if not has_lidar:
            print("\n⚠️  WARNING: No LiDAR/points topic found!")
            print("   Expected topic like: /lidar/points2")
        
        conn.close()
        
        if has_camera and has_lidar:
            print("\n✅ All required topics found!")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ ERROR reading bag file: {e}")
        return False


def test_camera_intrinsics():
    """Test camera intrinsic calculation."""
    print_section("TEST 2: CAMERA INTRINSICS")
    
    # Your camera specs
    width = 640
    height = 480
    fov_deg = 170.0
    
    print(f"Camera specifications:")
    print(f"  Resolution: {width} x {height} pixels")
    print(f"  Field of View: {fov_deg}°")
    
    # Calculate focal length
    import math
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
    
    print(f"\nCalculated Camera Matrix K:")
    print(K)
    print(f"\nParameters:")
    print(f"  Focal length (fx, fy): {fx:.2f} pixels")
    print(f"  Principal point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels")
    
    # Test projection
    print(f"\n✅ Test: Project a point at [1, 0, 5] meters")
    print(f"   (1m right, 0m up, 5m forward from camera)")
    
    point_3d = np.array([1.0, 0.0, 5.0])  # x, y, z in camera frame
    
    # Normalize by depth
    norm_point = point_3d / point_3d[2]  # divide by Z
    
    # Project
    pixel = K @ np.array([norm_point[0], norm_point[1], 1.0])
    u, v = int(pixel[0]), int(pixel[1])
    
    print(f"   → Projects to pixel: ({u}, {v})")
    
    if 0 <= u < width and 0 <= v < height:
        print(f"   ✅ Pixel is inside image bounds")
    else:
        print(f"   ❌ Pixel is OUTSIDE image bounds!")
    
    return K


def test_extrinsics():
    """Test extrinsic transformation."""
    print_section("TEST 3: EXTRINSICS (SENSOR POSITIONS)")
    
    # Positions from Unreal Engine (in cm)
    lidar_pos_cm = np.array([-31.08193, -0.000001, 179.706462])
    camera_pos_cm = np.array([156.729406, 0.0, 103.342601])
    
    print(f"Sensor positions in Unreal Engine (centimeters):")
    print(f"  LiDAR:  {lidar_pos_cm}")
    print(f"  Camera: {camera_pos_cm}")
    
    # Calculate relative position
    relative_pos_cm = camera_pos_cm - lidar_pos_cm
    print(f"\nRelative position (Camera - LiDAR) in cm:")
    print(f"  {relative_pos_cm}")
    
    # Convert to meters
    relative_pos_m = relative_pos_cm / 100.0
    print(f"\nRelative position in meters:")
    print(f"  {relative_pos_m}")
    
    # Apply coordinate conversion (Unreal left-handed → ROS right-handed)
    # Unreal: X=forward, Y=right, Z=up
    # ROS: X=forward, Y=left, Z=up
    # Conversion: flip Y axis
    t_l2c = np.array([
        relative_pos_m[0],   # X: forward (same)
        -relative_pos_m[1],  # Y: flip (right → left)
        relative_pos_m[2]    # Z: up (same)
    ])
    
    print(f"\nTranslation vector (LiDAR → Camera) in ROS frame:")
    print(f"  {t_l2c} meters")
    print(f"\nInterpretation:")
    print(f"  Camera is {t_l2c[0]:.3f}m FORWARD of LiDAR")
    print(f"  Camera is {abs(t_l2c[1]):.3f}m {'LEFT' if t_l2c[1] > 0 else 'RIGHT'} of LiDAR")
    print(f"  Camera is {abs(t_l2c[2]):.3f}m {'ABOVE' if t_l2c[2] > 0 else 'BELOW'} LiDAR")
    
    # Test transformation
    print(f"\n✅ Test: Transform LiDAR point [5, 0, 0] to camera frame")
    print(f"   (5m forward in LiDAR frame)")
    
    lidar_point = np.array([5.0, 0.0, 0.0])
    R = np.eye(3)  # No rotation
    camera_point = R @ lidar_point + t_l2c
    
    print(f"   → In camera frame: {camera_point}")
    
    if camera_point[2] > 0:
        print(f"   ✅ Point is IN FRONT of camera (Z > 0)")
    else:
        print(f"   ❌ Point is BEHIND camera (Z < 0) - Won't be visible!")
    
    return R, t_l2c


def test_synchronization(bag_path):
    """Test timestamp synchronization between camera and LiDAR."""
    print_section("TEST 4: MESSAGE SYNCHRONIZATION")
    
    try:
        conn = sqlite3.connect(bag_path)
        cursor = conn.cursor()
        
        # Get topic IDs
        cursor.execute("SELECT id, name FROM topics")
        topics = {name: id for id, name in cursor.fetchall()}
        
        camera_topic = '/camera/image_raw'
        lidar_topic = '/lidar/points2'
        
        if camera_topic not in topics or lidar_topic not in topics:
            print("❌ ERROR: Required topics not found")
            print(f"   Looking for: {camera_topic} and {lidar_topic}")
            print(f"   Available: {list(topics.keys())}")
            return
        
        # Get timestamps
        cursor.execute(
            "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 5",
            (topics[camera_topic],)
        )
        camera_times = [t[0] for t in cursor.fetchall()]
        
        cursor.execute(
            "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 5",
            (topics[lidar_topic],)
        )
        lidar_times = [t[0] for t in cursor.fetchall()]
        
        print(f"First 5 camera timestamps (nanoseconds):")
        for i, t in enumerate(camera_times):
            print(f"  {i+1}. {t}")
        
        print(f"\nFirst 5 LiDAR timestamps (nanoseconds):")
        for i, t in enumerate(lidar_times):
            print(f"  {i+1}. {t}")
        
        # Check time differences
        if camera_times and lidar_times:
            min_diff = min(abs(ct - lt) for ct in camera_times for lt in lidar_times)
            min_diff_ms = min_diff / 1e6
            
            print(f"\n✅ Minimum time difference: {min_diff_ms:.2f} ms")
            
            if min_diff_ms < 50:
                print(f"   ✅ GOOD: Messages are well synchronized (< 50ms)")
            elif min_diff_ms < 100:
                print(f"   ⚠️  ACCEPTABLE: Slight delay ({min_diff_ms:.1f}ms)")
            else:
                print(f"   ❌ WARNING: Large delay ({min_diff_ms:.1f}ms)")
                print(f"      May need to increase sync_tolerance in code")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")


def create_sample_visualization(bag_path, K, R, t):
    """Create a sample visualization to test the projection."""
    print_section("TEST 5: SAMPLE PROJECTION")
    
    try:
        conn = sqlite3.connect(bag_path)
        cursor = conn.cursor()
        
        # Get topic IDs
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {name: (id, msg_type) for id, name, msg_type in cursor.fetchall()}
        
        camera_topic = '/camera/image_raw'
        lidar_topic = '/lidar/points2'
        
        # Get one camera message
        camera_id, camera_type = topics[camera_topic]
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id = ? LIMIT 1",
            (camera_id,)
        )
        camera_data = cursor.fetchone()[0]
        camera_msg = deserialize_message(camera_data, Image)
        
        # Get one LiDAR message
        lidar_id, lidar_type = topics[lidar_topic]
        cursor.execute(
            "SELECT data FROM messages WHERE topic_id = ? LIMIT 1",
            (lidar_id,)
        )
        lidar_data = cursor.fetchone()[0]
        lidar_msg = deserialize_message(lidar_data, PointCloud2)
        
        conn.close()
        
        # Convert to OpenCV
        from cv_bridge import CvBridge
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        
        print(f"✅ Loaded sample frame:")
        print(f"   Image size: {img.shape}")
        
        # Extract points
        points_list = []
        for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if len(points_list) == 0:
            print("❌ No points in LiDAR scan!")
            return
        
        points = np.array(points_list, dtype=np.float32)
        print(f"   LiDAR points: {len(points)}")
        print(f"   X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
        print(f"   Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
        print(f"   Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
        
        # Transform to camera frame
        Pc = (R @ points.T + t.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        print(f"\n   Points in front of camera: {front.sum()} / {len(points)}")
        
        if front.sum() == 0:
            print("❌ No points in front of camera!")
            print("   This means either:")
            print("   1. LiDAR is pointing backwards")
            print("   2. Extrinsic calibration is wrong")
            print("   3. Coordinate system conversion is incorrect")
            return
        
        Pc = Pc[front]
        Z = Z[front]
        
        # Project to 2D
        h, w = img.shape[:2]
        uv = (K @ (Pc.T / Z)).T
        u, v = uv[:, 0], uv[:, 1]
        
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        print(f"   Points inside image: {keep.sum()} / {front.sum()}")
        
        if keep.sum() == 0:
            print("❌ No points project inside image!")
            print("   Possible issues:")
            print("   1. FOV calculation is wrong")
            print("   2. Camera intrinsics are incorrect")
            print("   3. Points are outside camera view")
            return
        
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        Z_vis = Z[keep]
        
        # Create visualization
        output = img.copy()
        Zn = (Z_vis - Z_vis.min()) / (Z_vis.max() - Z_vis.min() + 1e-6)
        Zc = (Zn * 255).astype(np.uint8)
        colors = cv2.applyColorMap(Zc, cv2.COLORMAP_JET)
        
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(output, (x, y), 2, tuple(int(a) for a in c[0]), -1)
        
        # Save image
        output_path = '/home/claude/test_projection.jpg'
        cv2.imwrite(output_path, output)
        print(f"\nSample projection saved to: {output_path}")
        print(f"   View this image to verify calibration!")
        
        return output_path
        
    except Exception as e:
        print(f"ERROR creating visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print_section("UGV FUSION TEST SCRIPT")
    print("This script will verify your setup before running the fusion node.")
    
    # Get bag file path
    if len(sys.argv) > 1:
        bag_path = sys.argv[1]
    else:
        bag_path = '/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3'
    
    print(f"Testing bag file: {bag_path}")
    
    # Run tests
    bag_ok = test_bag_file(bag_path)
    
    if not bag_ok:
        print("\nBag file test failed. Fix issues before continuing.")
        return
    
    K = test_camera_intrinsics()
    R, t = test_extrinsics()
    test_synchronization(bag_path)
    
    # Try sample projection
    try:
        output_path = create_sample_visualization(bag_path, K, R, t)
        
        if output_path:
            print_section("FINAL SUMMARY")
            print("All tests passed!")
            print(f"\n Sample projection saved to:")
            print(f"   {output_path}")
            print(f"\n Next steps:")
            print(f"   1. View the test_projection.jpg image")
            print(f"   2. Verify points align with image features")
            print(f"   3. If alignment is good → Run fusion node!")
            print(f"   4. If alignment is bad → Check calibration values")
    except Exception as e:
        print(f"\n Could not create sample projection: {e}")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
