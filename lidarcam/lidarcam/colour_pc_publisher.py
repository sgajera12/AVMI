#!/usr/bin/env python3
"""
UGV Colored Point Cloud Publisher for RViz

Publishes colored point clouds where:
- Points visible in camera → RGB from image
- Other points → Gray color
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
import cv2
import math
import sqlite3
from rclpy.serialization import deserialize_message
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge


def create_colored_cloud(header, points):
    """
    Create PointCloud2 with RGB colors.
    points: (N, 6) array [x, y, z, r, g, b]
    """
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    
    # Convert RGB to packed uint32
    rgb_packed = np.zeros(len(points), dtype=np.uint32)
    rgb_packed = (
        (points[:, 3].astype(np.uint32) << 16) |  # R
        (points[:, 4].astype(np.uint32) << 8)  |  # G
        (points[:, 5].astype(np.uint32))          # B
    )
    
    # Combine xyz and rgb
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


class ColoredPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('colored_pointcloud_publisher')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('bag_file', '/home/pinaka/dataset/AVMI/data/rosbag1210.db3'),
                ('fps', 10.0),
            ]
        )
        
        self.bag_file = self.get_parameter('bag_file').value
        self.fps = self.get_parameter('fps').value
        
        # Calibrated parameters
        self._setup_calibration()
        
        # Publishers
        self.colored_pc_pub = self.create_publisher(PointCloud2, '/colored_pointcloud', 10)
        self.raw_image_pub = self.create_publisher(Image, '/camera/left', 10)
        
        # Load bag data
        self._load_bag_data()
        
        # Timer
        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        
        self.get_logger().info(f"Colored Point Cloud Publisher started!")
        self.get_logger().info(f"Loaded {len(self.sync_data)} frames, publishing at {self.fps} Hz")
    
    def _setup_calibration(self):
        """Load calibrated parameters"""
        # Camera intrinsics
        fov_degrees = 90.0
        fov_rad = math.radians(fov_degrees)
        focal_length = (640 / 2.0) / math.tan(fov_rad / 2.0)
        
        self.K = np.array([
            [focal_length, 0, 320.0],
            [0, focal_length, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)
    
        pitch_deg = -74.0
        yaw_deg = -6.0
        roll_deg = 96.0
        self.dist_coeffs = np.array([0.000, 0.150, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Extrinsics
        self.t_l2c = np.array([1.478000, 0.000000, -1.064000], dtype=np.float32)
        
        pitch_rad = math.radians(-74.0)
        yaw_rad = math.radians(-6.0)
        roll_rad = math.radians(96.0)
        
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
        """Load synchronized data from bag"""
        conn = sqlite3.connect(self.bag_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name FROM topics")
        topics = {name: id for id, name in cursor.fetchall()}
        
        camera_id = topics['/camera/left/image_raw']
        lidar_id = topics['/lidar/points2']
        
        # Load all messages
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
        """Publish colored point cloud for current frame"""
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
        points = np.array(points_list, dtype=np.float32)
        
        # Create colored point cloud
        colored_pc = self.create_colored_pointcloud(img, points)
        
        if colored_pc is not None:
            # Create ROS message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'lidar_link'
            
            pc_msg = create_colored_cloud(header, colored_pc)
            self.colored_pc_pub.publish(pc_msg)
            
            # Publish raw image
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            img_msg.header = header
            self.raw_image_pub.publish(img_msg)
            
            if self.frame_idx % 10 == 0:
                self.get_logger().info(f"Published frame {self.frame_idx}/{len(self.sync_data)}")
    
    def create_colored_pointcloud(self, img_bgr, lidar_xyz):
        """
        Color ALL points:
        - Visible in camera → RGB
        - Not visible → Gray
        """
        h, w = img_bgr.shape[:2]
        total_points = len(lidar_xyz)
        
        # All points start gray
        all_colors = np.full((total_points, 3), 128, dtype=np.float32)
        
        # Transform to camera frame
        Pc = (self.R_l2c @ lidar_xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() > 0:
            Pc_front = Pc[front]
            front_indices = np.where(front)[0]
            
            # Project to image
            rvec = np.zeros(3, dtype=np.float32)
            tvec = np.zeros(3, dtype=np.float32)
            image_points, _ = cv2.projectPoints(Pc_front, rvec, tvec, self.K, self.dist_coeffs)
            image_points = image_points.reshape(-1, 2)
            u, v = image_points[:, 0], image_points[:, 1]
            
            # Find inside image
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            
            if inside.sum() > 0:
                u_valid = u[inside].astype(np.int32)
                v_valid = v[inside].astype(np.int32)
                visible_indices = front_indices[inside]
                
                # Sample colors
                colors_bgr = img_bgr[v_valid, u_valid]
                colors_rgb = colors_bgr[:, ::-1]
                
                # Assign RGB to visible points
                all_colors[visible_indices] = colors_rgb.astype(np.float32)
        
        # Combine: (N, 6) [x, y, z, r, g, b]
        colored_points = np.hstack((lidar_xyz, all_colors))
        return colored_points


def main(args=None):
    rclpy.init(args=args)
    node = ColoredPointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()