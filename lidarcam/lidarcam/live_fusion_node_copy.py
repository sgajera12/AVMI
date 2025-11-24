#!/usr/bin/env python3
import os
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge


def create_cloud(header, points):
    """Create PointCloud2 from (N,4) float32 array [x,y,z,intensity]."""
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

class LiveFusionNode(Node):
    def __init__(self):
        super().__init__('live_fusion_node')
        self.bridge = CvBridge()

        # Declare parameters (can be overridden by launch file)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_dir', '/home/pinaka/dataset/rellis3d/Rellis-3D'),
                ('sequence', '00000'),
                ('image_topic', '/camera/image_raw'),
                ('lidar_topic', '/lidar/points'),
                ('fusion_topic', '/fusion/projected_image'),
                ('color_proj_topic', '/fusion/color_projection_image'),
                ('rate_hz', 10.0),
            ]
        )

        base = self.get_parameter('base_dir').get_parameter_value().string_value
        seq  = self.get_parameter('sequence').get_parameter_value().string_value

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.fusion_topic = self.get_parameter('fusion_topic').get_parameter_value().string_value
        self.color_proj_topic = self.get_parameter('color_proj_topic').get_parameter_value().string_value
        #Paths
        self.lidar_dir = os.path.join(base, f'lidar/{seq}/os1_cloud_node_kitti_bin')
        self.image_dir = os.path.join(base, f'image/{seq}/pylon_camera_node')
        self.calib_file = os.path.join(base, f'calibration/{seq}/transforms.yaml')
        self.intr_file  = os.path.join(base, f'intrinsic/{seq}/camera_info.txt')
        #publishers
        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.lidar_pub = self.create_publisher(PointCloud2, self.lidar_topic, 10)
        self.fusion_pub = self.create_publisher(Image, self.fusion_topic, 10)
        self.color_proj_msg = self.create_publisher(Image, self.color_proj_topic, 10)


        #load calibration and lists
        self._load_calibration()
        self._load_lists()

        rate_hz = float(self.get_parameter('rate_hz').value)
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.index = 0

        self.get_logger().info(
            f"LiveFusion ready — frames: {self.total_frames}, rate: {rate_hz} Hz\n"
            f"Topics: {self.image_topic}, {self.lidar_topic}, {self.fusion_topic},{self.color_proj_topic}"
        )
    def _load_calibration(self):
        with open(self.calib_file, 'r') as f:
            tf = yaml.safe_load(f)['os1_cloud_node-pylon_camera_node']
        q, t = tf['q'], tf['t']

        # Camera->LiDAR in file; we need LiDAR->Camera (to project LiDAR into image)
        R_c2l = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix().astype(np.float32)
        t_c2l = np.array([t['x'], t['y'], t['z']], dtype=np.float32)

        self.R_l2c = R_c2l.T
        self.t_l2c = -R_c2l.T @ t_c2l

        vals = np.loadtxt(self.intr_file).flatten().astype(np.float32)
        fx, fy, cx, cy = vals[:4]
        self.K = np.array([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,  1]], dtype=np.float32)
        self.get_logger().info(f"Loaded intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    def _load_lists(self):
        self.lidar_files = sorted(
            [os.path.join(self.lidar_dir, f)
             for f in os.listdir(self.lidar_dir) if f.endswith('.bin')]
        )
        self.image_files = sorted(
            [os.path.join(self.image_dir, f)
             for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png'))]
        )
        self.total_frames = min(len(self.image_files), len(self.lidar_files))
        if self.total_frames == 0:
            raise RuntimeError("No synchronized frames found. Check dataset paths.")

    def _tick(self):
        if self.index >= self.total_frames:
            self.index = 0

        # Load image
        img_path = self.image_files[self.index]
        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().warn(f"Failed to read image {img_path}")
            self.index += 1
            return

        # Load LiDAR
        lidar_path = self.lidar_files[self.index]
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # (N,4)
        # Publish image
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.image_pub.publish(img_msg)
        # Publish LiDAR
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'lidar_link'
        cloud_msg = create_cloud(header, pts)
        self.lidar_pub.publish(cloud_msg)
        #project LiDAR into camera
        fused = self._project_points(img.copy(), pts[:, :3])
        fuse_msg = self.bridge.cv2_to_imgmsg(fused, encoding='bgr8')
        self.fusion_pub.publish(fuse_msg)
        
        color_proj = self._create_color_projection_map(img, pts[:, :3])
        color_proj = self.bridge.cv2_to_imgmsg(color_proj, encoding='bgr8')
        self.color_proj_msg.publish(color_proj)

        self.index += 1
    def _project_points(self, img_bgr, lidar_xyz):
        h, w = img_bgr.shape[:2]
        Pc = (self.R_l2c @ lidar_xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.0
        if front.sum() == 0:
            return img_bgr

        Pc = Pc[front]
        Z = Z[front]
        uv = (self.K @ (Pc.T / Z)).T
        u, v = uv[:, 0], uv[:, 1]
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if keep.sum() == 0:
            return img_bgr

        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        Z = Z[keep]
        # Color by depth
        Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
        Zc = (Zn * 255).astype(np.uint8)
        colors = cv2.applyColorMap(Zc, cv2.COLORMAP_JET)
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(img_bgr, (x, y), 3, tuple(int(a) for a in c[0]), -1)
        return img_bgr
    def _create_color_projection_map(self, img_bgr, lidar_xyz):
        """
        Create a black-background image showing LiDAR points projected into the camera view,
        colored by their corresponding pixel color from the image.
        """
        h, w = img_bgr.shape[:2]
        black_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Transform LiDAR -> Camera frame
        Pc = (self.R_l2c @ lidar_xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.0
        if front.sum() == 0:
            return black_map

        Pc = Pc[front]
        Z = Z[front]

        # Project to 2D
        uv = (self.K @ (Pc.T / Z)).T
        u, v = uv[:, 0], uv[:, 1]

        # Keep points inside the image boundaries
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if keep.sum() == 0:
            return black_map

        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)

        # Get true RGB color of each projected pixel
        colors = img_bgr[v, u, :]

        # Draw each projected LiDAR point on black canvas
        for (x, y, c) in zip(u, v, colors):
            cv2.circle(black_map, (x, y), 4, tuple(int(a) for a in c), -1)

        return black_map


def main(args=None):
    rclpy.init(args=args)
    node = LiveFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
