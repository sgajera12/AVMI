#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from scipy.spatial.transform import Rotation as R
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


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


def unreal_to_ros_transform(location_ue, rotation_ue_degrees, scale_ue):
    """
    Convert Unreal Engine transform to ROS transform
    
    UE5: X=forward, Y=right, Z=up (left-handed)
    ROS: X=forward, Y=left, Z=up (right-handed)
    
    Args:
        location_ue: [x, y, z] in Unreal (cm)
        rotation_ue_degrees: [roll, pitch, yaw] in Unreal (degrees)
        scale_ue: [sx, sy, sz] scale factors
    
    Returns:
        t_ros: translation in meters
        R_ros: rotation matrix
    """
    # Convert location from cm to meters and Unreal to ROS
    # UE to ROS: X->X, Y->-Y, Z->Z (flip Y axis)
    t_ue = np.array(location_ue) / 100.0  # cm to meters
    
    # Account for scale
    scale = np.array(scale_ue) if isinstance(scale_ue, (list, tuple)) else np.array([scale_ue, scale_ue, scale_ue])
    
    # If scale is negative, it means coordinate flip
    # Apply scale to position
    t_ue = t_ue / scale
    
    # Convert Unreal rotation (Roll-Pitch-Yaw) to rotation matrix
    # Unreal uses intrinsic rotations: Roll(X), Pitch(Y), Yaw(Z)
    roll, pitch, yaw = np.deg2rad(rotation_ue_degrees)
    
    # Unreal's rotation order (intrinsic XYZ = extrinsic ZYX)
    # R_ue = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    R_ue = R.from_euler('xyz', [roll, pitch, yaw + np.deg2rad(180)], degrees=False).as_matrix()

    
    # Transform from Unreal coordinate system to ROS
    # UE: X=fwd, Y=right, Z=up (left-handed)
    # ROS: X=fwd, Y=left, Z=up (right-handed)
    # Transformation: [X, Y, Z]_ros = [X, -Y, Z]_ue
    T_ue_to_ros = np.array([
        [1,  0,  0],
        [0, -1,  0],  # Flip Y
        [0,  0,  1]
    ])
    
    t_ros = T_ue_to_ros @ t_ue
    R_ros = T_ue_to_ros @ R_ue @ T_ue_to_ros.T
    return t_ros, R_ros


class LidarCamFusionLive(Node):
    def __init__(self):
        super().__init__('lidar_cam_fusion_live')
        self.bridge = CvBridge()

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('lidar_topic', '/lidar/points2')
        self.declare_parameter('fusion_topic', '/fusion/projected_image')
        self.declare_parameter('color_proj_topic', '/fusion/color_projection_image')
        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)
        self.declare_parameter('fov_deg', 90.0)

        self.image_topic = self.get_parameter('image_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.fusion_topic = self.get_parameter('fusion_topic').value
        self.color_proj_topic = self.get_parameter('color_proj_topic').value

        w, h = self.get_parameter('image_width').value, self.get_parameter('image_height').value
        fov = np.deg2rad(self.get_parameter('fov_deg').value)
        fx = fy = (w / 2) / np.tan(fov / 2)
        cx, cy = w / 2, h / 2
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float32)

        # Unreal Engine data (from your screenshots)
        # Lidar: Location, Rotation (Roll, Pitch, Yaw), Scale
        lidar_location_ue = [3959.0, 16841.0, -4239.0]  # cm
        lidar_rotation_ue = [-5.638522, 4.727098, -60.151668]  # degrees (Roll, Pitch, Yaw)
        lidar_scale_ue = 1.0  # assuming default scale
        
        #Camera: Location, Rotation (Roll, Pitch, Yaw), Scale
        camera_location_ue = [4021.525848, 16915.686234, -4277.101523]  # cm
        camera_rotation_ue = [-1.294432, -7.238753, 180.163131]  # degrees (Roll, Pitch, Yaw)
        camera_scale_ue = -0.25
        
        # --- Convert world poses to local (relative) ---
        # Make the camera origin the reference (0,0,0)
        lidar_location_rel = np.array(lidar_location_ue) - np.array(camera_location_ue)
        camera_location_rel = np.array([0.0, 0.0, 0.0])

        # Now use these relative positions in the conversion
        t_l2w_ros, R_l2w_ros = unreal_to_ros_transform(lidar_location_rel, lidar_rotation_ue, lidar_scale_ue)
        t_c2w_ros, R_c2w_ros = unreal_to_ros_transform(camera_location_rel, camera_rotation_ue, camera_scale_ue)

        #Compute Lidar to Camera transformation in ROS coordinates
        R_w2c = R_c2w_ros.T
        t_w2c = -R_w2c @ t_c2w_ros
        
        self.get_logger().info(f"current valuest_= l2w_ros={t_l2w_ros},\n R_l2w_ros= {R_l2w_ros},\n t_c2w_ros= {t_c2w_ros}, \n R_c2w_ros={R_c2w_ros}")
        self.R_l2c = R_w2c @ R_l2w_ros
        self.t_l2c = R_w2c @ t_l2w_ros + t_w2c
        self.get_logger().info(f"L2C translation before optical and rfix: {self.t_l2c}")
        r = R.from_matrix(self.R_l2c)
        euler = r.as_euler('xyz', degrees=True)
        self.get_logger().info(f"L2C Euler angles (deg): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}")

        #camera coordinate convention
        #considering ros optical frame: X=right, Y=down, Z=forward
        R_ros_to_optical = np.array([
            [0,  0,  1],  # Z_optical = X_ros (forward)
            [-1, 0,  0],  # X_optical = -Y_ros (right) 
            [0, -1,  0]   # Y_optical = -Z_ros (down)
        ])
        R_ros_to_optical = np.diag([1, 1, -1])
        R_fix = np.diag([1, -1, 1])
        self.R_l2c = R_fix @ self.R_l2c
        self.R_l2c = R_ros_to_optical @ self.R_l2c
      
        self.t_l2c = R_ros_to_optical @ self.t_l2c

        self.fusion_pub = self.create_publisher(Image, self.fusion_topic, 10)
        self.color_pub = self.create_publisher(Image, self.color_proj_topic, 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, sensor_qos)
        self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, sensor_qos)
        self.last_image = None
        self.last_lidar = None
        self.frame_count = 0
        # self.get_logger().info("="*70)
        # self.get_logger().info("LiDAR-Camera Fusion (Unreal Engine 5 → ROS)")
        # self.get_logger().info(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        # self.get_logger().info(f"FOV: {self.get_parameter('fov_deg').value}°")
        # self.get_logger().info(f"\nLidar→World (ROS): t={t_l2w_ros}")
        # self.get_logger().info(f"Camera→World (ROS): t={t_c2w_ros}")
        # self.get_logger().info(f"\nLidar→Camera: t={self.t_l2c}")
        # self.get_logger().info(f"R_l2c:\n{self.R_l2c}")
        # self.get_logger().info("="*70)

    def image_callback(self, msg):
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_fuse()

    def lidar_callback(self, msg):
        self.last_lidar = pointcloud2_to_xyz_array(msg)
        self.try_fuse()

    def try_fuse(self):
        if self.last_image is None or self.last_lidar is None:
            return

        img = self.last_image.copy()
        pts_lidar = self.last_lidar[:, :3]
        
        # Transform lidar points to camera frame
        Pc = (self.R_l2c @ pts_lidar.T).T + self.t_l2c
        
        # Debug logging every 30 frames
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            valid = Pc[:, 2] > 0
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(pts_lidar)} total pts, "
                f"{valid.sum()} in front (Z>0)"
            )
            if valid.sum() > 0:
                Pc_valid = Pc[valid]
                self.get_logger().info(
                    f"  Cam frame - X:[{Pc_valid[:, 0].min():.1f},{Pc_valid[:, 0].max():.1f}] "
                    f"Y:[{Pc_valid[:, 1].min():.1f},{Pc_valid[:, 1].max():.1f}] "
                    f"Z:[{Pc_valid[:, 2].min():.1f},{Pc_valid[:, 2].max():.1f}]"
                )

        Pc = (self.R_l2c @ pts_lidar.T).T + self.t_l2c
        self.get_logger().info(f"Mean Z in cam frame: {Pc[:,2].mean():.3f}, " f"min={Pc[:,2].min():.3f}, max={Pc[:,2].max():.3f}")
        Pc_dbg = Pc[:5]
        self.get_logger().info(f"Sample Pc (X,Y,Z):\n{Pc_dbg}")

        fused_depth = self.project_depth_colormap(img.copy(), Pc)
        fused_color = self.project_truecolor_map(img.copy(), Pc)

        self.fusion_pub.publish(self.bridge.cv2_to_imgmsg(fused_depth, encoding='bgr8'))
        self.color_pub.publish(self.bridge.cv2_to_imgmsg(fused_color, encoding='bgr8'))

    def project_depth_colormap(self, img, Pc):
        h, w = img.shape[:2]
        Z = Pc[:, 2]
        valid = Z > 0
        Pc_valid = Pc[valid]
        Z_valid = Z[valid]
        
        if len(Z_valid) == 0:
            cv2.putText(img, "No points in front of camera (Z<=0)", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img
            
        # Project to image
        uv = (self.K @ Pc_valid.T).T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]
        
        # Filter points in image bounds
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_valid = u[mask].astype(np.int32)
        v_valid = v[mask].astype(np.int32)
        Z_final = Z_valid[mask]
        
        if len(u_valid) == 0:
            cv2.putText(img, f"Points in front: {len(Z_valid)}, but none in image bounds", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            return img

        # Color by depth
        Z_norm = (Z_final - Z_final.min()) / (Z_final.max() - Z_final.min() + 1e-6)
        colors = cv2.applyColorMap((Z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        for (x, y, c) in zip(u_valid, v_valid, colors):
            cv2.circle(img, (x, y), 3, tuple(int(val) for val in c[0]), -1)
        
        # Info overlay
        info_text = f"Projected: {len(u_valid)}/{len(Pc)} points"
        cv2.putText(img, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        depth_text = f"Depth: {Z_final.min():.1f}m - {Z_final.max():.1f}m"
        cv2.putText(img, depth_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img

    def project_truecolor_map(self, img, Pc):
        h, w = img.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        Z = Pc[:, 2]
        valid = Z > 0
        Pc_valid = Pc[valid]
        
        if len(Pc_valid) == 0:
            return canvas
            
        uv = (self.K @ Pc_valid.T).T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]
        
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_valid = u[mask].astype(np.int32)
        v_valid = v[mask].astype(np.int32)
        
        if len(u_valid) == 0:
            return canvas
            
        colors = img[v_valid, u_valid, :]
        for (x, y, c) in zip(u_valid, v_valid, colors):
            cv2.circle(canvas, (x, y), 3, tuple(int(val) for val in c), -1)
        
        return canvas


def main(args=None):
    rclpy.init(args=args)
    node = LidarCamFusionLive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()