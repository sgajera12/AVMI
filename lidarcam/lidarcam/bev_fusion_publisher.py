#!/usr/bin/env python3
"""
Rellis-3D BEV Fusion Publisher for RViz
Publishes a stream of Bird’s-Eye-View (LiDAR+Camera fused) images
to visualize continuously in RViz.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2,PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge

import numpy as np
import cv2
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import time
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

class BevFusionPublisher(Node):
    def __init__(self):
        super().__init__('bev_fusion_publisher')
        self.bridge = CvBridge()

        # Parameters -------------------------------------------------
        self.declare_parameter('base_path', '/home/pinaka/dataset/rellis3d/Rellis-3D')
        self.declare_parameter('sequence', '00000')
        self.declare_parameter('fps', 2.0)       # frames per second
        self.declare_parameter('res', 0.1)       # BEV resolution (m/pixel)
        self.declare_parameter('x_range', [-30.0, 30.0])
        self.declare_parameter('y_range', [-30.0, 30.0])

        base = Path(self.get_parameter('base_path').value)
        seq  = self.get_parameter('sequence').value
        self.rate = self.get_parameter('fps').value
        self.res = self.get_parameter('res').value
        self.x_range = self.get_parameter('x_range').value
        self.y_range = self.get_parameter('y_range').value

        self.calib_file = base / f"calibration/{seq}/transforms.yaml"
        self.intr_file  = base / f"intrinsic/{seq}/camera_info.txt"
        self.lidar_dir  = base / f"lidar/{seq}/os1_cloud_node_kitti_bin"
        self.image_dir  = base / f"image/{seq}/pylon_camera_node"

        # Load calibration once --------------------------------------
        with open(self.calib_file, "r") as f:
            tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]

        q, t = tf["q"], tf["t"]
        R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
        t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)

        self.R_l2c = R_c2l.T
        self.t_l2c = -R_c2l.T @ t_c2l

        fx, fy, cx, cy = np.loadtxt(self.intr_file)
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=np.float32)
        self.get_logger().info(f"Loaded calibration, fx={fx:.1f}, fy={fy:.1f}")

        # Publisher ---------------------------------------------------
        self.pub_bev = self.create_publisher(Image, '/bev_fusion_image', 10)
        self.rawimage = self.create_publisher(Image, '/raw_image', 10)
        self.lidarpoints = self.create_publisher(PointCloud2, '/lidar_points', 10)
        

        # File lists --------------------------------------------------
        self.bins = sorted([p for p in self.lidar_dir.iterdir() if p.suffix == '.bin'])
        self.imgs = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png']])
        self.frame_idx = 0

        self.get_logger().info(f"Found {len(self.bins)} LiDAR frames and {len(self.imgs)} images.")

        # Timer -------------------------------------------------------
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

    # ----------------------------------------------------------------
    def timer_callback(self):
        if self.frame_idx >= len(self.bins):
            self.get_logger().info("Reached end of dataset.")
            self.destroy_timer(self.timer)
            return

        lidar_file = self.bins[self.frame_idx]
        image_file = self.imgs[self.frame_idx]
        self.frame_idx += 1

        # --- Load LiDAR & Image ---
        pts = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        intensity = pts[:, 3]
        img = cv2.imread(str(image_file))
        if img is None:
            self.get_logger().warn(f"Cannot read image {image_file}")
            return

        bev_img = self.generate_bev(xyz, intensity, img)
        msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_bev.publish(msg)
        self.rawimage.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        self.lidarpoints.publish(create_cloud(Header(stamp=self.get_clock().now().to_msg(),
                                                  frame_id='lidar_link'), pts))

        self.get_logger().info(f"Published BEV frame {self.frame_idx}/{len(self.bins)}")

    # ----------------------------------------------------------------
    def generate_bev(self, xyz_lidar, intensity, img):
        """Project LiDAR to camera, colorize, and build BEV"""
        h, w = img.shape[:2]
        Pc = (self.R_l2c @ xyz_lidar.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0
        uv = (self.K @ (Pc.T / np.maximum(Z, 1e-6))).T
        u, v = uv[:, 0], uv[:, 1]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        cam_visible = front & inside

        # Camera colors
        colors_cam = np.zeros((xyz_lidar.shape[0], 3), dtype=np.uint8)
        if np.any(cam_visible):
            u_vis = u[cam_visible].astype(np.int32)
            v_vis = v[cam_visible].astype(np.int32)
            colors_cam[cam_visible] = img[v_vis, u_vis, :]

        # LiDAR intensity colors
        inten = intensity.copy()
        inten -= inten.min()
        inten /= (inten.max() + 1e-6)
        inten_u8 = (inten * 255).astype(np.uint8)
        lidar_col = cv2.applyColorMap(inten_u8, cv2.COLORMAP_BONE)[:, 0, :]

        # Combine
        final_colors = lidar_col.copy()
        final_colors[cam_visible] = colors_cam[cam_visible]

        # Vehicle frame
        xyz_v = np.stack([xyz_lidar[:, 1], -xyz_lidar[:, 0], xyz_lidar[:, 2]], axis=1)
        X, Y = xyz_v[:, 0], xyz_v[:, 1]

        X_MIN, X_MAX = self.x_range
        Y_MIN, Y_MAX = self.y_range
        RES = self.res
        bev_H = int((X_MAX - X_MIN) / RES)
        bev_W = int((Y_MAX - Y_MIN) / RES)

        bev_rgb = np.zeros((bev_H, bev_W, 3), np.float32)
        bev_count = np.zeros((bev_H, bev_W), np.int32)

        mask = (X >= X_MIN) & (X < X_MAX) & (Y >= Y_MIN) & (Y < Y_MAX)
        Xb, Yb, Cb = X[mask], Y[mask], final_colors[mask]
        ix = ((Xb - X_MIN) / RES).astype(np.int32)
        iy = ((Yb - Y_MIN) / RES).astype(np.int32)
        ix = np.clip(ix, 0, bev_H - 1)
        iy = np.clip(iy, 0, bev_W - 1)

        for r, c, col in zip(ix, iy, Cb):
            bev_rgb[r, c] += col.astype(np.float32)
            bev_count[r, c] += 1
        nz = bev_count > 0
        bev_rgb[nz] /= bev_count[nz][..., None]

        bev_img = bev_rgb.astype(np.uint8)
        # bev_img =np.flipud(bev_rgb.astype(np.uint8))

        bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bev_img


def main(args=None):
    rclpy.init(args=args)
    node = BevFusionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()