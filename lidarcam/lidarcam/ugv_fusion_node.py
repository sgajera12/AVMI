#!/usr/bin/env python3
"""
UGV LiDAR-Camera Fusion Node for Unreal Engine Dataset

This node reads from a ROS2 bag file containing LiDAR and camera data,
projects LiDAR points onto the camera image, and publishes the fused results.

Key Concepts:
1. Coordinate Transformation: Converting LiDAR 3D points to camera frame
2. Projection: Using camera intrinsics to map 3D points to 2D image pixels
3. Visualization: Creating two types of outputs:
   - Depth-colored projection (points colored by distance)
   - RGB-colored projection (points colored by image pixel values)
"""

import os
import numpy as np
import cv2
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import sqlite3
from rosidl_runtime_py.utilities import get_message


def create_cloud(header, points):
    """
    Create a PointCloud2 message from a numpy array.
    
    WHY: ROS2 uses PointCloud2 format to transmit 3D point data
    WHAT: Converts numpy array (N, 4) with [x, y, z, intensity] to ROS message
    HOW: Defines field structure and serializes the data
    
    Args:
        header: ROS message header with timestamp and frame info
        points: numpy array of shape (N, 4) containing [x, y, z, intensity]
    
    Returns:
        PointCloud2 message ready to be published
    """
    fields = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    data = points.astype(np.float32).tobytes()
    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=16,
        row_step=16 * points.shape[0],
        data=data,
    )


class UGVFusionNode(Node):
    """
    Main node that handles LiDAR-Camera fusion for UGV data.
    
    This node:
    1. Reads messages from a ROS2 bag file
    2. Computes the transformation between LiDAR and camera
    3. Projects LiDAR points onto camera images
    4. Publishes synchronized camera, LiDAR, and fusion results
    """
    
    def __init__(self):
        super().__init__('ugv_fusion_node')
        self.bridge = CvBridge()
        
        # ============================================================
        # PARAMETERS - You can modify these as needed
        # ============================================================
        self.declare_parameters(
            namespace='',
            parameters=[
                ('bag_file', '/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3'),
                ('camera_topic', '/camera/image_raw'),
                ('lidar_topic', '/lidar/points2'),
                ('fusion_topic', '/fusion/projected_image'),
                ('color_proj_topic', '/fusion/color_projection_image'),
                ('republish_rate_hz', 10.0),
            ]
        )
        
        # Get parameters
        self.bag_file = self.get_parameter('bag_file').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.fusion_topic = self.get_parameter('fusion_topic').get_parameter_value().string_value
        self.color_proj_topic = self.get_parameter('color_proj_topic').get_parameter_value().string_value
        
        # ============================================================
        # PUBLISHERS
        # ============================================================
        self.image_pub = self.create_publisher(Image, self.camera_topic, 10)
        self.lidar_pub = self.create_publisher(PointCloud2, self.lidar_topic, 10)
        self.fusion_pub = self.create_publisher(Image, self.fusion_topic, 10)
        self.color_proj_pub = self.create_publisher(Image, self.color_proj_topic, 10)
        
        # ============================================================
        # CALIBRATION SETUP
        # ============================================================
        self._setup_calibration()
        
        # ============================================================
        # LOAD BAG FILE DATA
        # ============================================================
        self._load_bag_data()
        
        # ============================================================
        # TIMER FOR PLAYBACK
        # ============================================================
        rate_hz = float(self.get_parameter('republish_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.index = 0
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"UGV Fusion Node Initialized Successfully!\n"
            f"{'='*70}\n"
            f"Bag file: {self.bag_file}\n"
            f"Total synchronized frames: {len(self.sync_data)}\n"
            f"Playback rate: {rate_hz} Hz\n"
            f"Camera topic: {self.camera_topic}\n"
            f"LiDAR topic: {self.lidar_topic}\n"
            f"Fusion topic: {self.fusion_topic}\n"
            f"Color projection topic: {self.color_proj_topic}\n"
            f"{'='*70}\n"
        )
    
    def _setup_calibration(self):
        """
        Set up camera intrinsics and LiDAR-to-Camera extrinsics.
        
        CAMERA INTRINSICS:
        ------------------
        WHY: We need to know HOW the camera converts 3D points to 2D pixels
        WHAT: The camera matrix K contains focal lengths and principal point
        HOW: For a 90° FOV camera with 640x480 resolution:
        
        Explanation:
        - FOV (Field of View) = 90 degrees means the camera sees a 90° cone
        - Focal length relates FOV to image size: f = (width/2) / tan(FOV/2)
        - For 90° FOV: tan(45°) = 1, so f = width/2 = 320 pixels
        - Principal point (cx, cy) is usually at the image center
        
        Camera Matrix K:
            [fx  0  cx]     [320   0  320]
            [ 0 fy  cy]  =  [  0 320  240]
            [ 0  0   1]     [  0   0    1]
        
        where:
        - fx, fy = focal lengths in pixels (320 for 90° FOV)
        - cx, cy = principal point (image center: 320, 240)
        """
        
        # Camera specs (CALIBRATED VALUES from calibration_tuner.py)
        image_width = 640
        image_height = 480
        fov_degrees = 100.0  # ✅ CALIBRATED: Found to be 100°, not 90°
        
        # Calculate focal length from FOV
        # Formula: f = (width / 2) / tan(FOV / 2)
        fov_rad = math.radians(fov_degrees)
        focal_length = (image_width / 2.0) / math.tan(fov_rad / 2.0)
        
        fx = fy = focal_length
        cx = image_width / 2.0
        cy = image_height / 2.0
        
        self.K = np.array([
            [fx,  0,  cx],
            [ 0, fy,  cy],
            [ 0,  0,   1]
        ], dtype=np.float32)
        
        # Distortion coefficients (CALIBRATED)
        # k1, k2 are radial distortion, p1, p2 are tangential, k3 is higher-order radial
        self.dist_coeffs = np.array([0.000, 0.100, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"CAMERA INTRINSICS\n"
            f"{'='*70}\n"
            f"Image size: {image_width} x {image_height} pixels\n"
            f"Field of View: {fov_degrees}°\n"
            f"Focal length (fx, fy): {fx:.2f} pixels\n"
            f"Principal point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels\n"
            f"Distortion (k1, k2): ({self.dist_coeffs[0]:.3f}, {self.dist_coeffs[1]:.3f})\n"
            f"Camera Matrix K:\n{self.K}\n"
            f"{'='*70}\n"
        )
        
        """
        EXTRINSICS: LiDAR to Camera Transformation (CALIBRATED)
        ---------------------------------------------------------
        WHY: LiDAR and Camera have different positions AND orientations
        WHAT: We need rotation (R) and translation (t) to convert between them
        HOW: Found via interactive calibration_tuner.py
        
        CALIBRATED VALUES:
        ------------------
        Translation (meters): tx=-0.922, ty=-0.750, tz=-1.864
        Rotation (degrees): pitch=-90.0°, yaw=0.0°, roll=90.0°
        
        What these mean:
        - Translation: Camera is at (-0.922, -0.750, -1.864) relative to LiDAR
        - Rotation: Camera is tilted DOWN 90° (pitch=-90°) and rolled 90°
        
        The KEY insight: The camera's viewing angle is VERY different from
        what the raw Unreal positions suggested. This rotation is what fixes
        the circular arc projection problem!
        """
        
        # CALIBRATED translation (found via calibration_tuner.py)
        # These values create natural feature distribution, not circular arcs
        self.t_l2c = np.array([-0.922000, -0.750000, -1.864000], dtype=np.float32)
        
        # CALIBRATED rotation (pitch=-90°, yaw=0°, roll=90°)
        # This orients the camera frame to match how it actually views the world
        pitch_deg = -90.0
        yaw_deg = 0.0
        roll_deg = 90.0
        
        # Convert to radians
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)
        
        # Build rotation matrix from Euler angles (ZYX convention)
        # Pitch (around Y-axis, tilt up/down)
        R_pitch = np.array([
            [math.cos(pitch_rad),  0, math.sin(pitch_rad)],
            [0,                    1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ], dtype=np.float32)
        
        # Yaw (around Z-axis, pan left/right)
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
            [0,                  0,                  1]
        ], dtype=np.float32)
        
        # Roll (around X-axis)
        R_roll = np.array([
            [1, 0,                   0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad),  math.cos(roll_rad)]
        ], dtype=np.float32)
        
        # Combined rotation: R = R_yaw @ R_pitch @ R_roll
        self.R_l2c = R_yaw @ R_pitch @ R_roll
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"EXTRINSICS: LiDAR to Camera Transformation (CALIBRATED)\n"
            f"{'='*70}\n"
            f"Translation (tx, ty, tz): {self.t_l2c} meters\n"
            f"Rotation:\n"
            f"  - Pitch: {pitch_deg:.1f}° (tilt down)\n"
            f"  - Yaw:   {yaw_deg:.1f}° (no pan)\n"
            f"  - Roll:  {roll_deg:.1f}° (rotated)\n"
            f"\nRotation Matrix:\n{self.R_l2c}\n"
            f"\nThese values were found via calibration_tuner.py\n"
            f"They fix the circular arc projection problem!\n"
            f"{'='*70}\n"
        )
    
    def _load_bag_data(self):
        """
        Load and synchronize camera and LiDAR messages from ROS2 bag file.
        
        WHY: We need to pair camera images with LiDAR scans from the same time
        WHAT: Reads SQLite database and matches messages by timestamp
        HOW: Uses timestamp-based synchronization with tolerance
        
        ROS2 bags are SQLite databases with:
        - 'messages' table: Contains serialized message data
        - 'topics' table: Maps topic names to IDs
        
        Process:
        1. Find topic IDs for camera and LiDAR
        2. Read all messages for these topics
        3. Pair messages with close timestamps (within 50ms)
        """
        self.get_logger().info(f"Loading bag file: {self.bag_file}")
        
        if not os.path.exists(self.bag_file):
            raise FileNotFoundError(f"Bag file not found: {self.bag_file}")
        
        # Connect to SQLite database
        conn = sqlite3.connect(self.bag_file)
        cursor = conn.cursor()
        
        # Get topic IDs
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {name: (id, type) for id, name, type in cursor.fetchall()}
        
        self.get_logger().info(f"Available topics in bag: {list(topics.keys())}")
        
        # Find our topics
        camera_topic_name = '/camera/image_raw'
        lidar_topic_name = '/lidar/points2'
        
        if camera_topic_name not in topics:
            raise ValueError(f"Camera topic '{camera_topic_name}' not found in bag")
        if lidar_topic_name not in topics:
            raise ValueError(f"LiDAR topic '{lidar_topic_name}' not found in bag")
        
        camera_topic_id, camera_type = topics[camera_topic_name]
        lidar_topic_id, lidar_type = topics[lidar_topic_name]
        
        # Load messages
        camera_msgs = []
        lidar_msgs = []
        
        # Load camera messages
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (camera_topic_id,)
        )
        for timestamp, data in cursor.fetchall():
            msg = deserialize_message(data, Image)
            camera_msgs.append((timestamp, msg))
        
        # Load LiDAR messages
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (lidar_topic_id,)
        )
        for timestamp, data in cursor.fetchall():
            msg = deserialize_message(data, PointCloud2)
            lidar_msgs.append((timestamp, msg))
        
        conn.close()
        
        self.get_logger().info(
            f"Loaded {len(camera_msgs)} camera messages and {len(lidar_msgs)} LiDAR messages"
        )
        
        # Synchronize messages by timestamp
        self.sync_data = []
        sync_tolerance = 50_000_000  # 50ms in nanoseconds
        
        for cam_ts, cam_msg in camera_msgs:
            # Find closest LiDAR message
            best_match = None
            best_diff = float('inf')
            
            for lidar_ts, lidar_msg in lidar_msgs:
                diff = abs(cam_ts - lidar_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_match = (lidar_ts, lidar_msg)
                    
                # If we're getting further away, stop searching
                if lidar_ts > cam_ts + sync_tolerance:
                    break
            
            if best_match and best_diff < sync_tolerance:
                self.sync_data.append({
                    'camera': cam_msg,
                    'lidar': best_match[1],
                    'timestamp': cam_ts
                })
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"Synchronized {len(self.sync_data)} frames\n"
            f"Synchronization tolerance: {sync_tolerance/1e6:.1f} ms\n"
            f"{'='*70}\n"
        )
        
        if len(self.sync_data) == 0:
            raise RuntimeError("No synchronized frames found!")
    
    def _tick(self):
        """
        Timer callback that publishes one synchronized frame at a time.
        
        This simulates real-time playback of the recorded data.
        """
        if self.index >= len(self.sync_data):
            self.index = 0  # Loop back to start
        
        data = self.sync_data[self.index]
        
        # Convert camera message to OpenCV image
        try:
            img = self.bridge.imgmsg_to_cv2(data['camera'], desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert camera image: {e}")
            self.index += 1
            return
        
        # Extract LiDAR points from PointCloud2 message
        points = self._extract_points_from_cloud(data['lidar'])
        
        if points is None or len(points) == 0:
            self.get_logger().warn("No valid LiDAR points")
            self.index += 1
            return
        
        # Create timestamp
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        
        # Publish original messages
        self.image_pub.publish(data['camera'])
        self.lidar_pub.publish(data['lidar'])
        
        # Project LiDAR points onto camera image
        fused_depth = self._project_points_depth_colored(img.copy(), points[:, :3])
        fused_rgb = self._project_points_rgb_colored(img.copy(), points[:, :3])
        
        # Publish fusion results
        try:
            fuse_depth_msg = self.bridge.cv2_to_imgmsg(fused_depth, encoding='bgr8')
            fuse_rgb_msg = self.bridge.cv2_to_imgmsg(fused_rgb, encoding='bgr8')
            
            self.fusion_pub.publish(fuse_depth_msg)
            self.color_proj_pub.publish(fuse_rgb_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish fusion images: {e}")
        
        # Log progress
        if self.index % 10 == 0:
            self.get_logger().info(f"Processing frame {self.index + 1}/{len(self.sync_data)}")
        
        self.index += 1
    
    def _extract_points_from_cloud(self, cloud_msg):
        """
        Extract xyz points from PointCloud2 message.
        
        WHY: PointCloud2 is a serialized format, we need numpy arrays
        WHAT: Converts ROS PointCloud2 to numpy array of shape (N, 4)
        HOW: Uses sensor_msgs_py library to parse the binary data
        
        Args:
            cloud_msg: sensor_msgs/PointCloud2 message
        
        Returns:
            numpy array of shape (N, 4) with [x, y, z, intensity]
        """
        try:
            # Read points as list of [x, y, z, intensity]
            points_list = []
            for point in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True):
                points_list.append([point[0], point[1], point[2], 1.0])  # Add dummy intensity
            
            if len(points_list) == 0:
                return None
            
            points = np.array(points_list, dtype=np.float32)
            return points
            
        except Exception as e:
            self.get_logger().error(f"Failed to extract points: {e}")
            return None
    
    def _project_points_depth_colored(self, img_bgr, lidar_xyz):
        """
        Project LiDAR points onto image and color by depth (distance).
        
        WHAT THIS DOES:
        ---------------
        1. Transform LiDAR points to camera frame
        2. Project 3D points to 2D image coordinates
        3. Color points by their distance (closer = blue, farther = red)
        4. Draw colored circles on the image
        
        WHY: This helps visualize depth information from LiDAR
        
        HOW IT WORKS:
        -------------
        Step 1: Transform points from LiDAR frame to Camera frame
            P_camera = R_l2c × P_lidar + t_l2c
        
        Step 2: Keep only points in front of camera (Z > 0)
        
        Step 3: Project to image using camera matrix
            [u]       [X/Z]
            [v] = K × [Y/Z]
            [1]       [ 1 ]
        
        Step 4: Keep only points inside image boundaries
        
        Step 5: Color by depth using JET colormap
            - Blue: Close (small Z)
            - Green: Medium
            - Red: Far (large Z)
        
        Args:
            img_bgr: Original camera image (will be modified)
            lidar_xyz: LiDAR points (N, 3) array [x, y, z]
        
        Returns:
            Image with projected colored points
        """
        h, w = img_bgr.shape[:2]
        
        # Step 1: Transform LiDAR points to Camera frame
        # Formula: P_camera = R_l2c × P_lidar^T + t_l2c
        # Shape: (3, N) = (3, 3) × (3, N) + (3, 1)
        Pc = (self.R_l2c @ lidar_xyz.T + self.t_l2c.reshape(3, 1)).T  # (N, 3)
        
        # Step 2: Filter points in front of camera
        Z = Pc[:, 2]  # Depth values
        front = Z > 0.1  # Keep points at least 10cm in front
        
        if front.sum() == 0:
            return img_bgr  # No points to project
        
        Pc = Pc[front]
        Z = Z[front]
        
        # Step 3: Project to 2D using OpenCV's projectPoints (handles distortion!)
        # This is KEY: cv2.projectPoints applies both intrinsics AND distortion
        # Input: 3D points in camera frame
        # Output: 2D pixel coordinates with distortion correction
        
        # cv2.projectPoints needs rvec and tvec, but since points are ALREADY
        # in camera frame, we use zero rotation and translation
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        
        # Project with distortion
        image_points, _ = cv2.projectPoints(Pc, rvec, tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        # Step 4: Keep points inside image boundaries
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if keep.sum() == 0:
            return img_bgr
        
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        Z = Z[keep]
        
        # Step 5: Color by depth using JET colormap
        # Normalize depth to [0, 1]
        Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
        # Convert to [0, 255] and apply colormap
        Zc = (Zn * 255).astype(np.uint8)
        colors = cv2.applyColorMap(Zc, cv2.COLORMAP_JET)
        
        # Draw circles for each projected point
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(img_bgr, (x, y), 2, tuple(int(a) for a in c[0]), -1)
        
        return img_bgr
    
    def _project_points_rgb_colored(self, img_bgr, lidar_xyz):
        """
        Project LiDAR points and color by corresponding image RGB values.
        
        WHAT THIS DOES:
        ---------------
        1. Project LiDAR points to image coordinates (same as depth version)
        2. Sample RGB color from the image at projected locations
        3. Draw colored points on BLACK background
        
        WHY: This shows which image pixels correspond to LiDAR measurements
        
        HOW: Same projection as depth-colored, but:
        - Use image RGB values instead of depth for coloring
        - Draw on black background instead of original image
        
        This is useful for:
        - Seeing spatial distribution of LiDAR coverage
        - Identifying which objects have LiDAR measurements
        - Debugging calibration (misaligned colors = bad calibration)
        
        Args:
            img_bgr: Original camera image (for sampling colors)
            lidar_xyz: LiDAR points (N, 3) array [x, y, z]
        
        Returns:
            Black image with RGB-colored projected points
        """
        h, w = img_bgr.shape[:2]
        black_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Transform and project (same as depth version)
        Pc = (self.R_l2c @ lidar_xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() == 0:
            return black_map
        
        Pc = Pc[front]
        Z = Z[front]
        
        # Project with distortion (same as depth version)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        image_points, _ = cv2.projectPoints(Pc, rvec, tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if keep.sum() == 0:
            return black_map
        
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        
        # Sample RGB color from original image at projected coordinates
        colors = img_bgr[v, u, :]
        
        # Draw colored points on black background
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(black_map, (x, y), 2, tuple(int(a) for a in c), -1)
        
        return black_map


def main(args=None):
    """
    Main entry point for the node.
    
    This function:
    1. Initializes ROS2
    2. Creates the fusion node
    3. Spins (processes callbacks) until shutdown
    4. Cleans up resources
    """ 
    rclpy.init(args=args)
    
    try:
        node = UGVFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()