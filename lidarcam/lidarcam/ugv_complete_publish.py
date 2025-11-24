#!/usr/bin/env python3
"""
UGV Complete Fusion Publisher - All Visualizations in One

Publishes:
1. /camera/image_raw              - Raw camera image
2. /lidar/points                  - Raw LiDAR point cloud (intensity)
3. /fusion/depth_projection       - Depth-colored projection on image
4. /fusion/rgb_projection         - RGB-colored projection on black
5. /fusion/colored_pointcloud     - Full colored point cloud (RGB + gray)
6. /fusion/bev_map                - Bird's Eye View fusion map

View all in RViz!
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import sqlite3
from rclpy.serialization import deserialize_message
import sensor_msgs_py.point_cloud2 as pc2


def create_standard_cloud(header, points):
    """Create standard PointCloud2 (x,y,z,intensity)"""
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
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


def create_colored_cloud(header, points):
    """Create RGB PointCloud2 (x,y,z,rgb)"""
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    
    # Pack RGB into uint32
    rgb_packed = (
        (points[:, 3].astype(np.uint32) << 16) |  # R
        (points[:, 4].astype(np.uint32) << 8)  |  # G
        (points[:, 5].astype(np.uint32))          # B
    )
    
    cloud_data = np.zeros(len(points), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32),
    ])
    cloud_data['x'] = points[:, 0]
    cloud_data['y'] = points[:, 1]
    cloud_data['z'] = points[:, 2]
    cloud_data['rgb'] = rgb_packed
    
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=16,
        row_step=16 * len(points),
        data=cloud_data.tobytes(),
    )


class CompleteFusionPublisher(Node):
    def __init__(self):
        super().__init__('ugv_complete_fusion_publisher')
        self.bridge = CvBridge()
        
        # ============================================================
        # PARAMETERS
        # ============================================================
        self.declare_parameters(
            namespace='',
            parameters=[
                ('bag_file', '/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3'),
                ('fps', 10.0),
                ('bev_res', 0.1),
                ('bev_x_range', [-30.0, 30.0]),
                ('bev_y_range', [-30.0, 30.0]),
            ]
        )
        
        self.bag_file = self.get_parameter('bag_file').value
        self.fps = self.get_parameter('fps').value
        self.bev_res = self.get_parameter('bev_res').value
        self.bev_x_range = self.get_parameter('bev_x_range').value
        self.bev_y_range = self.get_parameter('bev_y_range').value
        
        # ============================================================
        # PUBLISHERS (6 topics)
        # ============================================================
        self.pub_camera = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_lidar = self.create_publisher(PointCloud2, '/lidar/points', 10)
        self.pub_depth_proj = self.create_publisher(Image, '/fusion/depth_projection', 10)
        self.pub_rgb_proj = self.create_publisher(Image, '/fusion/rgb_projection', 10)
        self.pub_colored_pc = self.create_publisher(PointCloud2, '/fusion/colored_pointcloud', 10)
        self.pub_bev = self.create_publisher(Image, '/fusion/bev_map', 10)
        
        # ============================================================
        # CALIBRATION
        # ============================================================
        self._setup_calibration()
        
        # ============================================================
        # LOAD DATA
        # ============================================================
        self._load_bag_data()
        
        # ============================================================
        # TIMER
        # ============================================================
        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("UGV COMPLETE FUSION PUBLISHER - STARTED")
        self.get_logger().info("="*70)
        self.get_logger().info(f"Bag file: {self.bag_file}")
        self.get_logger().info(f"Total frames: {len(self.sync_data)}")
        self.get_logger().info(f"Publishing at {self.fps} Hz")
        self.get_logger().info("")
        self.get_logger().info("Topics published:")
        self.get_logger().info("  1. /camera/image_raw           - Raw camera")
        self.get_logger().info("  2. /lidar/points               - Raw LiDAR")
        self.get_logger().info("  3. /fusion/depth_projection    - Depth colored")
        self.get_logger().info("  4. /fusion/rgb_projection      - RGB colored")
        self.get_logger().info("  5. /fusion/colored_pointcloud  - Full RGB cloud")
        self.get_logger().info("  6. /fusion/bev_map             - BEV top-down")
        self.get_logger().info("="*70)
    
    def _setup_calibration(self):
        """Load calibrated parameters"""
        # Camera intrinsics
        fov_degrees = 100.0
        fov_rad = math.radians(fov_degrees)
        focal_length = (640 / 2.0) / math.tan(fov_rad / 2.0)
        
        self.K = np.array([
            [focal_length, 0, 320.0],
            [0, focal_length, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([0.000, 0.100, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Extrinsics
        self.t_l2c = np.array([-0.922000, -0.750000, -1.864000], dtype=np.float32)
        
        pitch_rad = math.radians(-90.0)
        yaw_rad = math.radians(0.0)
        roll_rad = math.radians(90.0)
        
        R_pitch = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ], dtype=np.float32)
        
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ], dtype=np.float32)
        
        self.R_l2c = R_yaw @ R_pitch @ R_roll
        
        self.get_logger().info("Calibration loaded: FOV=100°, pitch=-90°, roll=90°")
    
    def _load_bag_data(self):
        """Load synchronized data from bag"""
        self.get_logger().info(f"Loading bag: {self.bag_file}")
        
        conn = sqlite3.connect(self.bag_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name FROM topics")
        topics = {name: id for id, name in cursor.fetchall()}
        
        camera_id = topics['/camera/image_raw']
        lidar_id = topics['/lidar/points2']
        
        # Load messages
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (camera_id,)
        )
        camera_msgs = [(t, deserialize_message(d, Image)) for t, d in cursor.fetchall()]
        
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (lidar_id,)
        )
        lidar_msgs = [(t, deserialize_message(d, PointCloud2)) for t, d in cursor.fetchall()]
        
        conn.close()
        
        # Synchronize
        self.sync_data = []
        for cam_t, cam_msg in camera_msgs:
            closest = min(lidar_msgs, key=lambda x: abs(x[0] - cam_t))
            if abs(closest[0] - cam_t) < 50_000_000:  # 50ms
                self.sync_data.append({'camera': cam_msg, 'lidar': closest[1]})
        
        self.get_logger().info(f"Synchronized {len(self.sync_data)} frames")
    
    def timer_callback(self):
        """Publish all visualizations for current frame"""
        if self.frame_idx >= len(self.sync_data):
            self.frame_idx = 0  # Loop
        
        data = self.sync_data[self.frame_idx]
        self.frame_idx += 1
        
        # Convert image
        img = self.bridge.imgmsg_to_cv2(data['camera'], desired_encoding='bgr8')
        
        # Extract LiDAR points
        points_list = []
        for point in pc2.read_points(data['lidar'], field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2], 1.0])  # Add intensity
        points = np.array(points_list, dtype=np.float32)
        xyz = points[:, :3]
        
        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'lidar_link'
        
        # ============================================================
        # PUBLISH ALL OUTPUTS
        # ============================================================
        
        # 1. Raw camera
        camera_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        camera_msg.header = header
        self.pub_camera.publish(camera_msg)
        
        # 2. Raw LiDAR
        lidar_msg = create_standard_cloud(header, points)
        self.pub_lidar.publish(lidar_msg)
        
        # 3. Depth projection
        depth_proj = self._create_depth_projection(img.copy(), xyz)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_proj, encoding='bgr8')
        depth_msg.header = header
        self.pub_depth_proj.publish(depth_msg)
        
        # 4. RGB projection
        rgb_proj = self._create_rgb_projection(img.copy(), xyz)
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_proj, encoding='bgr8')
        rgb_msg.header = header
        self.pub_rgb_proj.publish(rgb_msg)
        
        # 5. Colored point cloud
        colored_pc = self._create_colored_pointcloud(img, xyz)
        colored_pc_msg = create_colored_cloud(header, colored_pc)
        self.pub_colored_pc.publish(colored_pc_msg)
        
        # 6. BEV map
        bev_map = self._create_bev_map(img, xyz)
        bev_msg = self.bridge.cv2_to_imgmsg(bev_map, encoding='bgr8')
        bev_msg.header = header
        self.pub_bev.publish(bev_msg)
        
        # Log progress
        if self.frame_idx % 10 == 0:
            self.get_logger().info(
                f"Frame {self.frame_idx}/{len(self.sync_data)} | "
                f"Points: {len(xyz)} | Published 6 topics"
            )
    
    def _create_depth_projection(self, img, xyz):
        """Depth-colored projection on original image"""
        h, w = img.shape[:2]
        
        # Transform and project
        Pc = (self.R_l2c @ xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() == 0:
            return img
        
        Pc = Pc[front]
        Z = Z[front]
        
        # Project with distortion
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        image_points, _ = cv2.projectPoints(Pc, rvec, tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        # Keep inside
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if keep.sum() == 0:
            return img
        
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        Z_vis = Z[keep]
        
        # Color by depth
        Z_norm = (Z_vis - Z_vis.min()) / (Z_vis.max() - Z_vis.min() + 1e-6)
        Z_color = (Z_norm * 255).astype(np.uint8)
        colors = cv2.applyColorMap(Z_color, cv2.COLORMAP_JET)
        
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(img, (x, y), 2, tuple(int(a) for a in c[0]), -1)
        
        return img
    
    def _create_rgb_projection(self, img, xyz):
        """RGB-colored projection on black background"""
        h, w = img.shape[:2]
        black_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Transform and project
        Pc = (self.R_l2c @ xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() == 0:
            return black_map
        
        Pc = Pc[front]
        
        # Project
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
        
        # Sample colors
        colors = img[v, u, :]
        
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(black_map, (x, y), 2, tuple(int(a) for a in c), -1)
        
        return black_map
    
    def _create_colored_pointcloud(self, img, xyz):
        """
        Full colored point cloud:
        - Visible in camera → RGB
        - Not visible → Gray
        """
        h, w = img.shape[:2]
        total_points = len(xyz)
        
        # All start gray
        all_colors = np.full((total_points, 3), 128, dtype=np.float32)
        
        # Transform
        Pc = (self.R_l2c @ xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() > 0:
            Pc_front = Pc[front]
            front_indices = np.where(front)[0]
            
            # Project
            rvec = np.zeros(3, dtype=np.float32)
            tvec = np.zeros(3, dtype=np.float32)
            image_points, _ = cv2.projectPoints(Pc_front, rvec, tvec, self.K, self.dist_coeffs)
            image_points = image_points.reshape(-1, 2)
            u, v = image_points[:, 0], image_points[:, 1]
            
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            
            if inside.sum() > 0:
                u_valid = u[inside].astype(np.int32)
                v_valid = v[inside].astype(np.int32)
                visible_indices = front_indices[inside]
                
                # Sample colors
                colors_bgr = img[v_valid, u_valid]
                colors_rgb = colors_bgr[:, ::-1]
                
                # Assign RGB
                all_colors[visible_indices] = colors_rgb.astype(np.float32)
        
        # Combine (N, 6): [x, y, z, r, g, b]
        colored_points = np.hstack((xyz, all_colors))
        return colored_points
    
    def _create_bev_map(self, img, xyz):
        """Bird's Eye View fusion map"""
        h, w = img.shape[:2]
        
        # Project to find visible points
        Pc = (self.R_l2c @ xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0
        
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        image_points, _ = cv2.projectPoints(Pc, rvec, tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        cam_visible = front & inside
        
        # Color assignment
        colors = np.full((len(xyz), 3), 128, dtype=np.uint8)
        
        if cam_visible.sum() > 0:
            u_vis = u[cam_visible].astype(np.int32)
            v_vis = v[cam_visible].astype(np.int32)
            colors[cam_visible] = img[v_vis, u_vis, :]
        
        # Transform to vehicle frame for BEV
        xyz_v = np.stack([xyz[:, 1], -xyz[:, 0], xyz[:, 2]], axis=1)
        X, Y = xyz_v[:, 0], xyz_v[:, 1]
        
        # BEV grid
        X_MIN, X_MAX = self.bev_x_range
        Y_MIN, Y_MAX = self.bev_y_range
        RES = self.bev_res
        bev_H = int((X_MAX - X_MIN) / RES)
        bev_W = int((Y_MAX - Y_MIN) / RES)
        
        bev_rgb = np.zeros((bev_H, bev_W, 3), np.float32)
        bev_count = np.zeros((bev_H, bev_W), np.int32)
        
        # Filter in range
        mask = (X >= X_MIN) & (X < X_MAX) & (Y >= Y_MIN) & (Y < Y_MAX)
        Xb, Yb, Cb = X[mask], Y[mask], colors[mask]
        
        # Grid indices
        ix = ((Xb - X_MIN) / RES).astype(np.int32)
        iy = ((Yb - Y_MIN) / RES).astype(np.int32)
        ix = np.clip(ix, 0, bev_H - 1)
        iy = np.clip(iy, 0, bev_W - 1)
        
        # Accumulate
        for r, c, col in zip(ix, iy, Cb):
            bev_rgb[r, c] += col.astype(np.float32)
            bev_count[r, c] += 1
        
        # Average
        nz = bev_count > 0
        bev_rgb[nz] /= bev_count[nz][..., None]
        
        bev_img = bev_rgb.astype(np.uint8)
        bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_CLOCKWISE)
        
        return bev_img


def main(args=None):
    rclpy.init(args=args)
    node = CompleteFusionPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()