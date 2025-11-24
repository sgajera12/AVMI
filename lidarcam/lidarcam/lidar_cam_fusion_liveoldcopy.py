#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# Utility: Convert PointCloud2 → Nx3 numpy array
def pointcloud2_to_xyz_array(cloud_msg):
    fmt = 'ffff'  # x, y, z, intensity
    step = struct.calcsize(fmt)
    data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    n_points = len(data) // step
    points = np.zeros((n_points, 4), dtype=np.float32)

    for i in range(n_points):
        x, y, z, intensity = struct.unpack_from(fmt, data, i * step)
        points[i] = (x, y, z, intensity)
    return points


class LidarCamFusionLive(Node):
    def __init__(self):
        super().__init__('lidar_cam_fusion_live')
        self.bridge = CvBridge()

        # --- Declare Parameters ---
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('lidar_topic', '/lidar/points')
        self.declare_parameter('fusion_topic', '/fusion/projected_image')
        self.declare_parameter('color_proj_topic', '/fusion/color_projection_image')
        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)
        self.declare_parameter('fov_deg', 90.0)

        # --- Get parameters ---
        self.image_topic = self.get_parameter('image_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.fusion_topic = self.get_parameter('fusion_topic').value
        self.color_proj_topic = self.get_parameter('color_proj_topic').value

        # --- Calibration (from Unreal Engine transform screenshots) ---
        # Camera intrinsics (from 90° FOV, 1920x1080)
        w, h = self.get_parameter('image_width').value, self.get_parameter('image_height').value
        fov = np.deg2rad(self.get_parameter('fov_deg').value)
        fx = fy = (w / 2) / np.tan(fov / 2)
        cx, cy = w / 2, h / 2
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float32)

        # Extrinsics (LiDAR → Camera) from Unreal transforms
        t_l2c = np.array([62.5, 74.7, -38.1], dtype=np.float32) / 100.0  # cm → m (example offset)
        r_l2c = R.from_euler('xyz', [-1.29, -7.23, 180.16], degrees=True).as_matrix().astype(np.float32)
        self.R_l2c = r_l2c
        self.t_l2c = t_l2c

        # self.get_logger().info(f"Calibration loaded:\nK=\n{self.K}\nR=\n{self.R_l2c}\nt={self.t_l2c}")

        # --- Publishers ---
        self.fusion_pub = self.create_publisher(Image, self.fusion_topic, 10)
        self.color_pub = self.create_publisher(Image, self.color_proj_topic, 10)

        # --- Subscriptions ---
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=5)
        self.create_subscription(Image, self.image_topic, self.image_callback, sensor_qos)
        self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, sensor_qos)


        # Buffers
        self.last_image = None
        self.last_lidar = None

        self.get_logger().info("Live LiDAR-Camera Fusion Node Initialized.")

    def image_callback(self, msg):
        """Store latest camera image"""
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_fuse()

    def lidar_callback(self, msg):
        """Store latest LiDAR point cloud"""
        self.last_lidar = pointcloud2_to_xyz_array(msg)
        self.try_fuse()

    def try_fuse(self):
        """Run fusion when both image and lidar are available"""
        if self.last_image is None or self.last_lidar is None:
            print( "Waiting for both image and LiDAR data..." )
            return

        img = self.last_image.copy()
        pts = self.last_lidar[:, :3]

        fused_depth = self.project_depth_colormap(img.copy(), pts)
        fused_color = self.project_truecolor_map(img.copy(), pts)

        # Publish both results
        self.fusion_pub.publish(self.bridge.cv2_to_imgmsg(fused_depth, encoding='bgr8'))
        self.color_pub.publish(self.bridge.cv2_to_imgmsg(fused_color, encoding='bgr8'))

    def project_depth_colormap(self, img, pts):
        """Project LiDAR points onto image, colored by depth"""
        h, w = img.shape[:2]
        Pc = (self.R_l2c @ pts.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        valid = Z > 0
        Pc = Pc[valid]
        Z = Z[valid]
        uv = (self.K @ (Pc.T / Z)).T
        u, v = uv[:, 0], uv[:, 1]
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, Z = u[mask].astype(np.int32), v[mask].astype(np.int32), Z[mask]
        if len(u) == 0:
            return img

        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
        colors = cv2.applyColorMap((Z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(img, (x, y), 2, tuple(int(v) for v in c[0]), -1)
        return img

    def project_truecolor_map(self, img, pts):
        """Project LiDAR points onto a black background using true image colors"""
        h, w = img.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        Pc = (self.R_l2c @ pts.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        valid = Z > 0
        Pc = Pc[valid]
        Z = Z[valid]
        uv = (self.K @ (Pc.T / Z)).T
        u, v = uv[:, 0], uv[:, 1]
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[mask].astype(np.int32), v[mask].astype(np.int32)
        colors = img[v, u, :]
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(canvas, (x, y), 3, tuple(int(a) for a in c), -1)
        return canvas


def main(args=None):
    rclpy.init(args=args)
    node = LidarCamFusionLive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
