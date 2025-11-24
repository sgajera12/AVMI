#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2, math, struct

def pointcloud2_to_xyz(msg):
    """Convert PointCloud2 → Nx3 numpy array (x, y, z)."""
    fmt = 'ffff'   # x, y, z, intensity
    step = struct.calcsize(fmt)
    data = np.frombuffer(msg.data, dtype=np.uint8)
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
    # Rz(yaw) * Ry(pitch) * Rx(roll)  (ROS convention)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr           ]
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

        # topics (unchanged)
        self.create_subscription(Image,      '/camera/image_raw', self.cam_cb,   sensor_qos)
        self.create_subscription(PointCloud2, '/lidar/points2',   self.lidar_cb, sensor_qos)
        self.pub_overlay = self.create_publisher(Image, '/fusion/overlay_image', 10)

        # state
        self.img = None
        self.pts = None
        w, h = 640.0, 480.0
        fov_h = np.deg2rad(90.0)  
        fx = (w/2.0)/np.tan(fov_h/2.0)
        fy = fx * (w/h)          
        cx, cy = w/2.0, h/2.0
        self.K = np.array([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,   1]], dtype=np.float32)
        self.cx, self.cy = cx, cy

        # base_link -> lidar_link (given by your static_transform_publisher)
        t_b_l = np.array([-0.3108, 0.0, 1.797], dtype=np.float32)
        R_b_l = rpy_to_R(0.0, 0.0, 0.0)

        # base_link -> camera_optical_frame : you assumed coincident & aligned
        t_b_c = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        R_b_c = rpy_to_R(0.0, 0.0, 0.0)

        # lidar -> camera (optical) :  T_l_c = T_c_b * T_b_l
        R_c_b = R_b_c.T
        t_c_b = -R_c_b @ t_b_c
        self.R_l2c = R_c_b @ R_b_l
        self.t_l2c = R_c_b @ t_b_l + t_c_b
        self.apply_optical = True  # set False if your image header frame_id already ends with '_optical_frame'
        self.R_cam_to_opt = np.array([[0, 0, 1],
                                        [1, 0, 0],
                                        [0,-1, 0]], dtype=np.float32)
        self.get_logger().info(f"R_l2c=\n{self.R_l2c}\n t_l2c={self.t_l2c}")

    # Callbacks (unchanged structure)
    def cam_cb(self, msg: Image):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if hasattr(msg.header, "frame_id"):
            fid = msg.header.frame_id or ""
            if "optical" in fid:
                self.apply_optical = False
        self.try_project()

    def lidar_cb(self, msg: PointCloud2):
        self.pts = pointcloud2_to_xyz(msg)
        self.try_project()

    # Projection (unchanged publisher/output) 
    def try_project(self):
        if self.img is None or self.pts is None or self.K is None:
            return

        img = self.img.copy()
        h, w = img.shape[:2]

        #LiDAR → camera_optical_frame
        Pc = (self.R_l2c @ self.pts.T + self.t_l2c.reshape(3, 1)).T

        # basic range filter (helps visually)
        Z = Pc[:, 2]
        mask = (Z > 0.1)
        if not np.any(mask):
            return
        Pc = Pc[mask]; Z = Z[mask]

        # project
        uv = (self.K @ (Pc.T / Z)).T
        u = uv[:, 0]; v = uv[:, 1]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            return

        u = u[inside].astype(np.int32)
        v = v[inside].astype(np.int32)
        Z = Z[inside]

        # colorize by depth
        Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
        cols = cv2.applyColorMap((Zn * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # draw (fast)
        img[v, u] = cols.reshape(-1, 3)

        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'camera_link'
        self.pub_overlay.publish(out)

        self.get_logger().info(
            f"Projected {len(u)} / {len(self.pts)} | Zcam mean {Z.mean():.1f} min {Z.min():.1f} max {Z.max():.1f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarCamFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
