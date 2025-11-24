#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
import struct
import math


def pointcloud2_to_xyz(cloud_msg):
    """Convert PointCloud2 → Nx3 numpy array (x, y, z)."""
    fmt = 'ffff'   # x, y, z, intensity
    step = struct.calcsize(fmt)
    data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    n = len(data) // step
    pts = np.zeros((n, 3), np.float32)
    for i in range(n):
        x, y, z, intensity = struct.unpack_from(fmt, data, i * step)
        pts[i] = (x, y, z)
    return pts

def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rz(yaw) * Ry(pitch) * Rx(roll)  (ROS standard)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=np.float32)

class LidarCamFusion(Node):
    def __init__(self):
        super().__init__('lidar_cam_fusion_live')
        self.bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, '/camera/image_raw', self.cam_cb, sensor_qos)
        self.create_subscription(PointCloud2, '/lidar/points2', self.lidar_cb, sensor_qos)
        self.pub_overlay = self.create_publisher(Image, '/fusion/overlay_image', 10)

        self.img, self.pts = None, None

        w, h = 640.0, 480.0
        fov_h = np.deg2rad(90.0)  
        fx = (w/2.0)/np.tan(fov_h/2.0)
        fy = fx * (w/h)          
        cx, cy = w/2.0, h/2.0
        self.K = np.array([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,   1]], dtype=np.float32)
        self.cx, self.cy = cx, cy

        # base to lidar
        t_b_l = np.array([-0.3108, 0.0, 1.797], dtype=np.float32)
        R_b_l = rpy_to_R(0.0, 0.0, 0.0)

        # base to camera (now computed)
        t_b_c = np.array([1.567, 0.0, 1.033], dtype=np.float32)
        R_b_c = rpy_to_R(0.0, 0.0, 0.0)
        # Fix orientation mismatch: ROS -> optical frame

        #Computing lidar to camera
        # T_l_c = T_c_b * T_b_l
        R_c_b = R_b_c.T
        t_c_b = -R_c_b @ t_b_c
        self.R_l2c = R_c_b @ R_b_l
        self.t_l2c = R_c_b @ t_b_l + t_c_b

        self.apply_optical = True  # set False if your image header frame_id already ends with '_optical_frame'
        self.R_cam_to_opt = np.array([[0, 0, 1],
                                      [1, 0, 0],
                                      [0,-1, 0]], dtype=np.float32)

        self.get_logger().info(
            f"K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}\n"
            f"R_l2c=\n{self.R_l2c}\n t_l2c={self.t_l2c}"
        )

    def cam_cb(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # decide once: is the frame already optical?
        if hasattr(msg.header, "frame_id"):
            fid = msg.header.frame_id or ""
            if "optical" in fid:
                self.apply_optical = False
        self.try_project()

    def lidar_cb(self, msg):
        self.pts = pointcloud2_to_xyz(msg)
        self.try_project()

    def try_project(self):
        if self.img is None or self.pts is None:
            return
        img = self.img.copy()
        h, w = img.shape[:2]

        R_ros_to_optical = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, -1, 0]
        ], dtype=np.float32)

        Pc = (R_ros_to_optical @ (self.R_l2c @ self.pts.T + self.t_l2c.reshape(3, 1))).T
        Pc[:, 1] *= -1

        Z = Pc[:, 2]
        front = Z > 0
        if not np.any(front):
            return
        Pc = Pc[front]; Z = Z[front]

        uv = (self.K @ (Pc.T / Z)).T
        u, v = uv[:, 0], uv[:, 1]
        m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(m):
            self.get_logger().info(
                f"In-front: {len(Z)}; in-bounds: 0 | "
                f"Z range [{Z.min():.1f}, {Z.max():.1f}]")
            return

        u = u[m].astype(np.int32); 
        v = v[m].astype(np.int32); 
        Z = Z[m]

        Zn = (Z - Z.min())/(Z.max()-Z.min()+1e-6)
        
        cols = cv2.applyColorMap((Zn*255).astype(np.uint8), cv2.COLORMAP_JET)
        for (x, y, c) in zip(u, v, cols.reshape(-1, 3)):
            img[y, x] = c

        # cv2.line(img, (int(self.cx), 0), (int(self.cx), h), (0, 255, 255), 1)
        # cv2.line(img, (0, int(self.cy)), (w, int(self.cy)), (0, 255, 255), 1)

        msg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        self.pub_overlay.publish(msg)

        # quick diagnostics
        self.get_logger().info(
            f"Projected {len(u)} / {len(self.pts)} | "
            f"Zcam mean {Z.mean():.1f} min {Z.min():.1f} max {Z.max():.1f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarCamFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
# colcon build --symlink-install --packages-select lidarcam