#!/usr/bin/env python3
"""
UGV BEV Fusion Publisher for RViz
Creates Bird's-Eye-View (top-down) map with LiDAR+Camera fusion
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import sqlite3
from rclpy.serialization import deserialize_message
import sensor_msgs_py.point_cloud2 as pc2


class BevFusionPublisher(Node):
    def __init__(self):
        super().__init__('ugv_bev_fusion_publisher')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('bag_file', '/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3'),
                ('fps', 5.0),
                ('res', 0.1),  # BEV resolution (m/pixel)
                ('x_range', [-30.0, 30.0]),
                ('y_range', [-30.0, 30.0]),
            ]
        )
        
        self.bag_file = self.get_parameter('bag_file').value
        self.fps = self.get_parameter('fps').value
        self.res = self.get_parameter('res').value
        self.x_range = self.get_parameter('x_range').value
        self.y_range = self.get_parameter('y_range').value
        
        # Calibration
        self._setup_calibration()
        
        # Publishers
        self.bev_pub = self.create_publisher(Image, '/bev_fusion_image', 10)
        self.raw_image_pub = self.create_publisher(Image, '/raw_image', 10)
        
        # Load data
        self._load_bag_data()
        
        # Timer
        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        
        self.get_logger().info(f"UGV BEV Fusion Publisher started!")
        self.get_logger().info(f"Loaded {len(self.sync_data)} frames")
        self.get_logger().info(f"BEV range: X={self.x_range}, Y={self.y_range}, res={self.res}m/pixel")
    
    def _setup_calibration(self):
        """Calibrated parameters for UGV"""
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
            [math.cos(pitch_rad),  0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ], dtype=np.float32)
        
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad),  math.cos(roll_rad)]
        ], dtype=np.float32)
        
        self.R_l2c = R_yaw @ R_pitch @ R_roll
    
    def _load_bag_data(self):
        """Load synchronized frames from bag"""
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
            if abs(closest[0] - cam_t) < 50_000_000:  # 50ms tolerance
                self.sync_data.append({'camera': cam_msg, 'lidar': closest[1]})
        
        self.get_logger().info(f"Synchronized {len(self.sync_data)} frames")
    
    def timer_callback(self):
        """Generate and publish BEV for current frame"""
        if self.frame_idx >= len(self.sync_data):
            self.frame_idx = 0  # Loop
        
        data = self.sync_data[self.frame_idx]
        self.frame_idx += 1
        
        # Convert image
        img = self.bridge.imgmsg_to_cv2(data['camera'], desired_encoding='bgr8')
        
        # Extract LiDAR points
        points_list = []
        for point in pc2.read_points(data['lidar'], field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        xyz = np.array(points_list, dtype=np.float32)
        
        # Generate BEV
        bev_img = self.generate_bev(xyz, img)
        
        # Publish
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'bev_frame'
        
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = header
        self.bev_pub.publish(bev_msg)
        
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        img_msg.header = header
        self.raw_image_pub.publish(img_msg)
        
        if self.frame_idx % 10 == 0:
            self.get_logger().info(f"Published BEV frame {self.frame_idx}/{len(self.sync_data)}")
    
    def generate_bev(self, xyz_lidar, img):
        """
        Generate Bird's Eye View with fusion
        - Points visible in camera → RGB from image
        - Other points → Gray based on distance
        """
        h, w = img.shape[:2]
        
        # Project to camera to find visible points
        Pc = (self.R_l2c @ xyz_lidar.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0
        
        # Project
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        image_points, _ = cv2.projectPoints(Pc, rvec, tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        cam_visible = front & inside
        
        # Color assignment
        colors = np.full((len(xyz_lidar), 3), 128, dtype=np.uint8)  # Gray default
        
        if cam_visible.sum() > 0:
            u_vis = u[cam_visible].astype(np.int32)
            v_vis = v[cam_visible].astype(np.int32)
            colors[cam_visible] = img[v_vis, u_vis, :]
        
        # Transform to vehicle frame for BEV
        # Vehicle frame: X=forward, Y=left, Z=up
        # BEV top-down: X=right axis, Y=forward axis
        xyz_v = np.stack([xyz_lidar[:, 1], -xyz_lidar[:, 0], xyz_lidar[:, 2]], axis=1)
        X, Y = xyz_v[:, 0], xyz_v[:, 1]
        
        # BEV grid
        X_MIN, X_MAX = self.x_range
        Y_MIN, Y_MAX = self.y_range
        RES = self.res
        bev_H = int((X_MAX - X_MIN) / RES)
        bev_W = int((Y_MAX - Y_MIN) / RES)
        
        bev_rgb = np.zeros((bev_H, bev_W, 3), np.float32)
        bev_count = np.zeros((bev_H, bev_W), np.int32)
        
        # Filter points in range
        mask = (X >= X_MIN) & (X < X_MAX) & (Y >= Y_MIN) & (Y < Y_MAX)
        Xb, Yb, Cb = X[mask], Y[mask], colors[mask]
        
        # Grid indices
        ix = ((Xb - X_MIN) / RES).astype(np.int32)
        iy = ((Yb - Y_MIN) / RES).astype(np.int32)
        ix = np.clip(ix, 0, bev_H - 1)
        iy = np.clip(iy, 0, bev_W - 1)
        
        # Accumulate colors
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
    node = BevFusionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()