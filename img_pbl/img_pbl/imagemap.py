#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image, PointField
from cv_bridge import CvBridge
import struct
from scipy.spatial.transform import Rotation as R

class LidarCameraMapper(Node):
    def __init__(self):
        super().__init__('lidar_camera_mapper')
        self.bridge = CvBridge()

        # --- camera intrinsics ---
        self.fx = 2813.643275
        self.fy = 2808.326079
        self.cx = 969.285772
        self.cy = 624.049972
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0,  0,  1]])

        # --- extrinsics (lidar to camera) ---
        q = [0.49757205900281865, -0.5191354403519034,
             -0.49750025670087406, 0.48519473951486275]  # w,x,y,z
        self.R_lidar_to_cam = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        self.t = np.array([-0.362996576109623, 0.060237225768389635, -0.18926884540339461])

        self.latest_image = None

        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(PointCloud2, '/lidar/points', self.lidar_callback, 10)
        self.pub_overlay = self.create_publisher(Image, '/fusion/overlay', 10)

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, dPinholeCameraParametersesired_encoding='bgr8')

    def lidar_callback(self, msg: PointCloud2):
        if self.latest_image is None:
            return

        img = self.latest_image.copy()

        # --- parse lidar points ---
        points = self.read_points(msg)

        if points.shape[0] == 0:
            return

        # --- transform to camera frame ---
        pts_cam = (self.R_lidar_to_cam @ points.T + self.t.reshape(3,1)).T

        # filter in front of camera
        mask = pts_cam[:, 2] > 0
        pts_cam = pts_cam[mask]

        # --- project to image ---
        uv = (self.K @ (pts_cam.T / pts_cam[:, 2])).T
        u, v = uv[:, 0], uv[:, 1]
        h, w, _ = img.shape

        for x, y in zip(u, v):
            if 0 <= int(x) < w and 0 <= int(y) < h:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        # --- publish overlay image ---
        overlay_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

    def read_points(self, cloud_msg):
        fmt = 'fff'  # x y z as float32
        step = cloud_msg.point_step
        buf = cloud_msg.data
        n_points = cloud_msg.width * cloud_msg.height
        if n_points == 0:
            return np.empty((0,3), dtype=np.float32)
        points = np.ndarray((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            base = i * step
            x, y, z = struct.unpack_from(fmt, buf, base)
            points[i] = [x, y, z]
        return points


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
