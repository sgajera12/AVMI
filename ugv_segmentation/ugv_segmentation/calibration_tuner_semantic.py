#!/usr/bin/env python3
"""
Interactive Calibration Tuner - SEMANTIC COLORS VERSION

Shows colored LiDAR semantic segmentation points projected onto camera!
- Brown = Ground
- Green = Trees/Grass  
- Red = Rocks
- Orange/Yellow = Tree trunks

Adjust calibration until colored points align with image features!
"""

import os
import numpy as np
import cv2
import sqlite3
import struct
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import math


class CalibrationTuner:
    def __init__(self, bag_path, frame_number=0):
        self.bag_path = bag_path
        self.bridge = CvBridge()
        self.frame_number = frame_number
        
        # Load specific frame
        print(f"\nLoading frame {frame_number}...")
        self.img, self.points = self.load_sample(frame_number)
        
        # Starting parameters (from diagnostic results!)
        # Location from UE screenshot
        self.tx = -31.08 / 100.0  # -0.3108 m (backward)
        self.ty = -0.00 / 100.0   # 0.0 m
        self.tz = 202.0 / 100.0   # 2.02 m (up!)
        
        # Camera orientation (FROM DIAGNOSTIC - camera looks straight down!)
        self.pitch = -72.0  # Look straight down! (diagnostic found this works)
        self.yaw = -6.0      # No pan
        self.roll = 96.0     # No roll
        
        # Camera intrinsics
        self.fov = 95.0  # Field of view is 90 degrees
        self.update_intrinsics()
        
        # Distortion
        self.k1 = 0.0
        self.k2 = 0.0
        
        print("\n" + "="*70)
        print("  INTERACTIVE CALIBRATION TUNER - SEMANTIC COLORS")
        print("="*70)
        print("\nStarting with UE values from your screenshot:")
        print(f"  Location (UE): X=-31.08cm, Y=-0.00cm, Z=202.0cm")
        print(f"  Translation: ({self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}) meters")
        print(f"  Rotation (UE): all 0.0°")
        print(f"  Rotation: pitch={self.pitch:.1f}°, yaw={self.yaw:.1f}°, roll={self.roll:.1f}°")
        print(f"  FOV: {self.fov:.1f}° (left camera, 640x480)")
        print("\nNote: Camera is 2.02m ABOVE LiDAR (Z=202cm)")
        print("\nControls:")
        print("  LEFT/RIGHT Arrow: Previous/Next frame")
        print("  Q/A: Adjust translation X (forward/back)")
        print("  W/S: Adjust translation Y (left/right)")
        print("  E/D: Adjust translation Z (up/down)")
        print("  R/F: Adjust pitch (tilt up/down)")
        print("  T/G: Adjust yaw (pan left/right)")
        print("  Y/H: Adjust roll")
        print("  U/J: Adjust FOV")
        print("  I/K: Adjust k1 distortion")
        print("  O/L: Adjust k2 distortion")
        print("  P: Print current parameters")
        print("  ESC: Quit")
        print("="*70 + "\n")
        
        self.run()
    
    def load_sample(self, frame_number=0):
        """Load a specific frame from bag."""
        conn = sqlite3.connect(self.bag_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name FROM topics")
        topics = {name: id for id, name in cursor.fetchall()}
        
        camera_id = topics['/camera/left/image_raw']
        lidar_id = topics['/lidar/points2']
        
        # Load specific frame by offset
        cursor.execute(
            f"SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET {frame_number}", 
            (camera_id,)
        )
        camera_result = cursor.fetchone()
        if not camera_result:
            print(f"ERROR: Frame {frame_number} not found in camera data!")
            print(f"Falling back to frame 0...")
            cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (camera_id,))
            camera_result = cursor.fetchone()
        
        camera_data = camera_result[0]
        camera_msg = deserialize_message(camera_data, Image)
        
        cursor.execute(
            f"SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET {frame_number}",
            (lidar_id,)
        )
        lidar_result = cursor.fetchone()
        if not lidar_result:
            print(f"ERROR: Frame {frame_number} not found in LiDAR data!")
            cursor.execute("SELECT data FROM messages WHERE topic_id = ? LIMIT 1", (lidar_id,))
            lidar_result = cursor.fetchone()
        
        lidar_data = lidar_result[0]
        lidar_msg = deserialize_message(lidar_data, PointCloud2)
        
        conn.close()
        
        img = self.bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        
        points_list = []
        # Try to read RGB from point cloud - multiple methods
        
        # Method 1: Try reading 'rgb' field
        try:
            for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True):
                x, y, z, rgb = point
                
                # Decode RGB from packed float
                # RGB might be packed as uint32 in a float
                rgb_packed = struct.unpack('I', struct.pack('f', rgb))[0]
                r = (rgb_packed >> 16) & 0xFF
                g = (rgb_packed >> 8) & 0xFF
                b = rgb_packed & 0xFF
                
                points_list.append([x, y, z, r, g, b])
            
            if len(points_list) > 0:
                print(f"✅ Method 1 worked: Extracted RGB from 'rgb' field")
                # Check if we have actual colors (not all same)
                points_array = np.array(points_list, dtype=np.float32)
                unique_colors = len(np.unique(points_array[:, 3:6], axis=0))
                print(f"   Found {unique_colors} unique colors")
                if unique_colors > 1:
                    points = points_array
                    return img, points
        except Exception as e:
            print(f"Method 1 failed: {e}")
            points_list = []
        
        # Method 2: Try reading r, g, b fields separately
        try:
            for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z', 'r', 'g', 'b'), skip_nans=True):
                x, y, z, r, g, b = point
                points_list.append([x, y, z, int(r), int(g), int(b)])
            
            if len(points_list) > 0:
                print(f"✅ Method 2 worked: Extracted from separate r,g,b fields")
                points_array = np.array(points_list, dtype=np.float32)
                unique_colors = len(np.unique(points_array[:, 3:6], axis=0))
                print(f"   Found {unique_colors} unique colors")
                if unique_colors > 1:
                    points = points_array
                    return img, points
        except Exception as e:
            print(f"Method 2 failed: {e}")
            points_list = []
        
        # Method 3: Manual parsing from raw data
        print("Trying Method 3: Manual parsing...")
        import struct as st
        
        # Get point step and data
        point_step = lidar_msg.point_step
        data = lidar_msg.data
        
        for i in range(0, len(data) - point_step, point_step):
            try:
                # Assuming: x,y,z (float32, 12 bytes) + rgb (uint32, 4 bytes) = 16 bytes
                x = st.unpack_from('f', data, i)[0]
                y = st.unpack_from('f', data, i + 4)[0]
                z = st.unpack_from('f', data, i + 8)[0]
                rgb_int = st.unpack_from('I', data, i + 12)[0]
                
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF
                
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    points_list.append([x, y, z, r, g, b])
            except:
                continue
        
        if len(points_list) > 0:
            print(f"✅ Method 3 worked: Manual parsing")
            points_array = np.array(points_list, dtype=np.float32)
            unique_colors = len(np.unique(points_array[:, 3:6], axis=0))
            print(f"   Found {unique_colors} unique colors")
            
            # Show sample colors
            print("\n   Sample RGB values:")
            for i in range(min(10, len(points_array))):
                r, g, b = points_array[i, 3:6].astype(int)
                print(f"     Point {i}: RGB({r}, {g}, {b})")
            
            points = points_array
            return img, points
        
        print("❌ All methods failed - no RGB data found!")
        # Fallback: just XYZ
        points_list = []
        for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2], 128, 128, 128])  # Gray
        
        points = np.array(points_list, dtype=np.float32)
        
        print(f"✅ Loaded frame {frame_number}: {len(points)} LiDAR points, image shape {img.shape}")
        
        return img, points
    
    def update_intrinsics(self):
        """Update camera matrix from FOV."""
        width, height = 640, 480
        fov_rad = math.radians(self.fov)
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)
    
    def get_rotation_matrix(self):
        """Get rotation matrix from pitch, yaw, roll."""
        # Pitch (around Y-axis, tilt up/down)
        pitch_rad = math.radians(self.pitch)
        R_pitch = np.array([
            [math.cos(pitch_rad),  0, math.sin(pitch_rad)],
            [0,                    1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        # Yaw (around Z-axis, pan left/right)
        yaw_rad = math.radians(self.yaw)
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
            [0,                  0,                  1]
        ])
        
        # Roll (around X-axis)
        roll_rad = math.radians(self.roll)
        R_roll = np.array([
            [1, 0,                   0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad),  math.cos(roll_rad)]
        ])
        
        # Combined rotation
        R = R_yaw @ R_pitch @ R_roll
        return R.astype(np.float32)
    
    def project_points(self):
        """Project points with current parameters and show semantic colors."""
        h, w = self.img.shape[:2]
        
        # Get transformation
        R = self.get_rotation_matrix()
        t = np.array([self.tx, self.ty, self.tz], dtype=np.float32)
        
        # Extract XYZ and RGB
        xyz = self.points[:, :3]
        rgb = self.points[:, 3:6].astype(np.uint8)
        
        # Transform to camera frame
        Pc = (R @ xyz.T + t.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() == 0:
            return self.img.copy(), 0
        
        Pc = Pc[front]
        Z = Z[front]
        rgb = rgb[front]
        
        # Project with distortion
        dist_coeffs = np.array([self.k1, self.k2, 0.0, 0.0, 0.0], dtype=np.float32)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        
        image_points, _ = cv2.projectPoints(Pc, rvec, tvec, self.K, dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        # Keep points in image
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if keep.sum() == 0:
            return self.img.copy(), 0
        
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        rgb = rgb[keep]
        
        # Create visualization
        output = self.img.copy()
        
        # Draw points with their semantic colors (BGR for OpenCV)
        for (x, y, color) in zip(u, v, rgb):
            r, g, b = color
            cv2.circle(output, (x, y), 3, (int(b), int(g), int(r)), -1)
        
        return output, keep.sum()
    
    def print_params(self):
        """Print current parameters."""
        print("\n" + "="*70)
        print("CURRENT PARAMETERS:")
        print("="*70)
        print(f"Translation (tx, ty, tz): ({self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}) meters")
        print(f"Rotation (pitch, yaw, roll): ({self.pitch:.1f}°, {self.yaw:.1f}°, {self.roll:.1f}°)")
        print(f"FOV: {self.fov:.1f}°")
        print(f"Distortion (k1, k2): ({self.k1:.3f}, {self.k2:.3f})")
        print("="*70)
        print("\nTo use these values in ugv_fusion_node.py:")
        print(f"  self.t_l2c = np.array([{self.tx:.6f}, {self.ty:.6f}, {self.tz:.6f}], dtype=np.float32)")
        print(f"  # Add rotation with pitch={self.pitch:.1f}°, yaw={self.yaw:.1f}°, roll={self.roll:.1f}°")
        print(f"  fov_degrees = {self.fov:.1f}")
        print(f"  self.dist_coeffs = np.array([{self.k1:.3f}, {self.k2:.3f}, 0.0, 0.0, 0.0], dtype=np.float32)")
        print("="*70 + "\n")
    
    def run(self):
        """Main loop."""
        step_trans = 0.01  # 5cm steps
        step_rot = 1.0     # 1 degree steps
        step_fov = 5.0     # 5 degree steps
        step_dist = 0.01   # Distortion steps
        
        while True:
            # Generate projection
            result, num_points = self.project_points()
            
            # Add info overlay
            cv2.putText(result, f"Frame: {self.frame_number} | Points: {num_points}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"FOV: {self.fov:.0f} deg", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Pitch: {self.pitch:.1f} deg", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, "Press 'P' for params, ESC to quit", (10, result.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show
            cv2.imshow('Calibration Tuner', result)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle keys
            if key == 27:  # ESC
                break
            elif key == 81:  # Left arrow
                self.frame_number = max(0, self.frame_number - 1)
                print(f"Loading frame {self.frame_number}...")
                self.img, self.points = self.load_sample(self.frame_number)
            elif key == 83:  # Right arrow
                self.frame_number += 1
                print(f"Loading frame {self.frame_number}...")
                try:
                    self.img, self.points = self.load_sample(self.frame_number)
                except:
                    print(f"No more frames! Staying at frame {self.frame_number - 1}")
                    self.frame_number -= 1
            elif key == ord('q'):
                self.tx += step_trans
                print(f"TX: {self.tx:.3f}m (forward)")
            elif key == ord('a'):
                self.tx -= step_trans
                print(f"TX: {self.tx:.3f}m (backward)")
            elif key == ord('w'):
                self.ty += step_trans
                print(f"TY: {self.ty:.3f}m (left)")
            elif key == ord('s'):
                self.ty -= step_trans
                print(f"TY: {self.ty:.3f}m (right)")
            elif key == ord('e'):
                self.tz += step_trans
                print(f"TZ: {self.tz:.3f}m (up)")
            elif key == ord('d'):
                self.tz -= step_trans
                print(f"TZ: {self.tz:.3f}m (down)")
            elif key == ord('r'):
                self.pitch += step_rot
                print(f"Pitch: {self.pitch:.1f}° (tilt up)")
            elif key == ord('f'):
                self.pitch -= step_rot
                print(f"Pitch: {self.pitch:.1f}° (tilt down)")
            elif key == ord('t'):
                self.yaw += step_rot
                print(f"Yaw: {self.yaw:.1f}° (pan left)")
            elif key == ord('g'):
                self.yaw -= step_rot
                print(f"Yaw: {self.yaw:.1f}° (pan right)")
            elif key == ord('y'):
                self.roll += step_rot
                print(f"Roll: {self.roll:.1f}°")
            elif key == ord('h'):
                self.roll -= step_rot
                print(f"Roll: {self.roll:.1f}°")
            elif key == ord('u'):
                self.fov += step_fov
                self.update_intrinsics()
                print(f"FOV: {self.fov:.1f}°")
            elif key == ord('j'):
                self.fov -= step_fov
                self.update_intrinsics()
                print(f"FOV: {self.fov:.1f}°")
            elif key == ord('i'):
                self.k1 += step_dist
                print(f"k1: {self.k1:.3f}")
            elif key == ord('k'):
                self.k1 -= step_dist
                print(f"k1: {self.k1:.3f}")
            elif key == ord('o'):
                self.k2 += step_dist
                print(f"k2: {self.k2:.3f}")
            elif key == ord('l'):
                self.k2 -= step_dist
                print(f"k2: {self.k2:.3f}")
            elif key == ord('p'):
                self.print_params()
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    bag_path = '/home/pinaka/dataset/AVMI/data/rosbag1210.db3'
    
    # Get frame number from command line argument
    frame_number = 500
    if len(sys.argv) > 1:
        try:
            frame_number = int(sys.argv[1])
            print(f"Loading frame {frame_number}")
        except ValueError:
            print(f"Invalid frame number: {sys.argv[1]}, using frame 0")
    
    print(f"\n{'='*70}")
    print(f"  CALIBRATION TUNER - SEMANTIC COLORS - Frame {frame_number}")
    print(f"{'='*70}")
    print(f"Bag file: {bag_path}")
    print(f"Frame: {frame_number}")
    print(f"Camera: Left (640x480, FOV=90°)")
    print(f"\nStarting with UE values from screenshot:")
    print(f"  Location: X=-31.08cm, Y=0.0cm, Z=202cm")
    print(f"  Rotation: 0°, 0°, 0°")
    print(f"\nUsage: python3 calibration_tuner_semantic.py [frame_number]")
    print(f"Example: python3 calibration_tuner_semantic.py 50")
    print(f"{'='*70}\n")
    
    tuner = CalibrationTuner(bag_path, frame_number)