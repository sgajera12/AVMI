#!/usr/bin/env python3
"""
Create Colored 3D Point Cloud from UGV LiDAR-Camera Fusion

This script does the REVERSE of projection:
1. Takes LiDAR 3D points
2. Projects them onto the camera image
3. Samples RGB color from the image at projected locations
4. Creates a colored 3D point cloud (.ply file)
5. Visualizes in 3D using Open3D

WHY: This lets you see the fused data in 3D space, where each LiDAR point
     has the color from the camera. Perfect for verification and visualization!

WHAT: Input = LiDAR points (x,y,z) + Camera image (RGB)
      Output = Colored point cloud (x,y,z,R,G,B)

HOW: Uses the same calibrated parameters we found via calibration_tuner.py
"""

import os
import numpy as np
import cv2
import math
import sqlite3
from pathlib import Path

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("WARNING: Open3D not installed. Will save PLY but cannot visualize.")
    print("Install with: pip install open3d")
    HAS_OPEN3D = False


class ColoredPointCloudGenerator:
    """
    Generates colored 3D point clouds by fusing LiDAR and camera data.
    
    Think of this like "painting" the LiDAR points with colors from the camera.
    The LiDAR knows where things are (x, y, z), and the camera knows what
    color they are (R, G, B). We combine them!
    """
    
    def __init__(self):
        self.bridge = CvBridge()
        
        # CALIBRATED PARAMETERS (from calibration_tuner.py)
        # These are the values we found that fix the circular arc problem!
        
        # Camera intrinsics
        image_width = 640
        image_height = 480
        fov_degrees = 100.0
        
        # Calculate camera matrix
        fov_rad = math.radians(fov_degrees)
        focal_length = (image_width / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy = focal_length
        cx = image_width / 2.0
        cy = image_height / 2.0
        
        self.K = np.array([
            [fx,  0,  cx],
            [ 0, fy,  cy],
            [ 0,  0,   1]
        ], dtype=np.float32)
        
        # Distortion coefficients
        self.dist_coeffs = np.array([0.000, 0.100, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Extrinsics: LiDAR to Camera transformation
        self.t_l2c = np.array([-0.922000, -0.750000, -1.864000], dtype=np.float32)
        
        # Rotation matrix (pitch=-90°, yaw=0°, roll=90°)
        pitch_deg = -90.0
        yaw_deg = 0.0
        roll_deg = 90.0
        
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        roll_rad = math.radians(roll_deg)
        
        # Build rotation matrices
        R_pitch = np.array([
            [math.cos(pitch_rad),  0, math.sin(pitch_rad)],
            [0,                    1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ], dtype=np.float32)
        
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
            [0,                  0,                  1]
        ], dtype=np.float32)
        
        R_roll = np.array([
            [1, 0,                   0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad),  math.cos(roll_rad)]
        ], dtype=np.float32)
        
        self.R_l2c = R_yaw @ R_pitch @ R_roll
        
        print("="*70)
        print("COLORED POINT CLOUD GENERATOR - Initialized")
        print("="*70)
        print(f"Camera FOV: {fov_degrees}°")
        print(f"Camera Matrix:\n{self.K}")
        print(f"Translation (LiDAR to Camera): {self.t_l2c}")
        print(f"Rotation: pitch={pitch_deg}°, yaw={yaw_deg}°, roll={roll_deg}°")
        print("="*70)
    
    def load_frame_from_bag(self, bag_path, frame_number=0):
        """
        Load a specific frame from the ROS2 bag file.
        
        WHY: We need synchronized LiDAR and camera data
        WHAT: Extracts one frame (LiDAR + Image) from the bag
        HOW: Uses SQLite to query the bag database
        
        Args:
            bag_path: Path to .db3 bag file
            frame_number: Which frame to load (0 = first frame)
        
        Returns:
            img: Camera image (H x W x 3) in BGR format
            points: LiDAR points (N x 3) as numpy array [x, y, z]
        """
        print(f"\nLoading frame {frame_number} from bag...")
        
        if not os.path.exists(bag_path):
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
        
        conn = sqlite3.connect(bag_path)
        cursor = conn.cursor()
        
        # Get topic IDs
        cursor.execute("SELECT id, name FROM topics")
        topics = {name: id for id, name in cursor.fetchall()}
        
        camera_id = topics['/camera/image_raw']
        lidar_id = topics['/lidar/points2']
        
        # Load camera frame
        cursor.execute(
            f"SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET {frame_number}",
            (camera_id,)
        )
        camera_result = cursor.fetchone()
        if not camera_result:
            raise ValueError(f"Frame {frame_number} not found in camera data")
        
        camera_msg = deserialize_message(camera_result[0], Image)
        
        # Load LiDAR frame
        cursor.execute(
            f"SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1 OFFSET {frame_number}",
            (lidar_id,)
        )
        lidar_result = cursor.fetchone()
        if not lidar_result:
            raise ValueError(f"Frame {frame_number} not found in LiDAR data")
        
        lidar_msg = deserialize_message(lidar_result[0], PointCloud2)
        
        conn.close()
        
        # Convert to OpenCV and numpy
        img = self.bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        
        points_list = []
        for point in pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        points = np.array(points_list, dtype=np.float32)
        
        print(f"✅ Loaded frame {frame_number}:")
        print(f"   - Image: {img.shape[1]}x{img.shape[0]} pixels")
        print(f"   - LiDAR: {len(points)} points")
        
        return img, points
    
    def create_colored_pointcloud(self, img_bgr, lidar_xyz):
        """
        Create colored point cloud by projecting LiDAR onto image.
        
        STEP-BY-STEP EXPLANATION:
        -------------------------
        
        Step 1: Transform LiDAR points to Camera frame
        -----------------------------------------------
        LiDAR points are in LiDAR coordinate system. We need to transform
        them to Camera coordinate system using rotation (R) and translation (t).
        
        Formula: P_camera = R × P_lidar + t
        
        Example: If LiDAR sees a point at (5, 0, 0) meters forward,
                 after transformation it might be at (3, 1, 2) in camera frame
        
        Step 2: Filter points in front of camera
        -----------------------------------------
        Cameras can only see things in front of them (positive Z).
        We remove points behind the camera (Z < 0.1m).
        
        Think: Like only keeping things you can see when looking forward,
               ignoring things behind you.
        
        Step 3: Project 3D points to 2D image coordinates
        --------------------------------------------------
        Use camera intrinsics (K matrix) and distortion to map 3D points
        to 2D pixel coordinates (u, v).
        
        Think: Like drawing a perspective view - things farther away
               appear smaller and closer to the image center.
        
        Step 4: Keep only points inside image boundaries
        -------------------------------------------------
        Some points project outside the image (u < 0 or u > width).
        We only keep points that fall inside the image.
        
        Think: Like cropping - only keep what's visible in the frame.
        
        Step 5: Sample RGB colors from image
        -------------------------------------
        For each projected point at (u, v), we read the RGB color
        from the image at that pixel location.
        
        Think: Like using a color picker tool in Photoshop - click a pixel,
               get its color!
        
        Step 6: Create colored point cloud
        -----------------------------------
        Combine original 3D coordinates with sampled RGB colors.
        Result: Each point has (x, y, z, R, G, B).
        
        Args:
            img_bgr: Camera image (H x W x 3) in BGR format
            lidar_xyz: LiDAR points (N x 3) [x, y, z]
        
        Returns:
            colored_points: (M x 6) array with [x, y, z, R, G, B]
            projection_image: Visualization showing which points were used
        """
        h, w = img_bgr.shape[:2]
        
        print("\n" + "="*70)
        print("CREATING COLORED POINT CLOUD")
        print("="*70)
        
        # Step 1: Transform to camera frame
        print("Step 1: Transforming LiDAR points to camera frame...")
        Pc = (self.R_l2c @ lidar_xyz.T + self.t_l2c.reshape(3, 1)).T
        original_count = len(Pc)
        print(f"   - Original points: {original_count}")
        
        # Step 2: Filter points in front
        print("Step 2: Filtering points in front of camera (Z > 0.1m)...")
        Z = Pc[:, 2]
        front = Z > 0.1
        
        if front.sum() == 0:
            print("   ⚠️ WARNING: No points in front of camera!")
            return None, img_bgr
        
        Pc_front = Pc[front]
        lidar_front = lidar_xyz[front]  # Keep original coordinates!
        Z_front = Z[front]
        print(f"   - Points in front: {len(Pc_front)} ({100*len(Pc_front)/original_count:.1f}%)")
        
        # Step 3: Project to image with distortion
        print("Step 3: Projecting 3D points to 2D image coordinates...")
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        image_points, _ = cv2.projectPoints(Pc_front, rvec, tvec, self.K, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)
        u, v = image_points[:, 0], image_points[:, 1]
        
        # Step 4: Keep points inside image
        print("Step 4: Keeping only points inside image boundaries...")
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if inside.sum() == 0:
            print("   ⚠️ WARNING: No points project inside image!")
            return None, img_bgr
        
        u_valid = u[inside].astype(np.int32)
        v_valid = v[inside].astype(np.int32)
        lidar_valid = lidar_front[inside]  # These are the visible points in ORIGINAL LiDAR frame!
        Z_valid = Z_front[inside]
        
        print(f"   - Points inside image: {len(lidar_valid)} ({100*len(lidar_valid)/len(Pc_front):.1f}%)")
        
        # Step 5: Sample RGB colors from image
        print("Step 5: Sampling RGB colors from image pixels...")
        colors_bgr = img_bgr[v_valid, u_valid]  # (N, 3) BGR
        colors_rgb = colors_bgr[:, ::-1]        # Convert BGR -> RGB
        
        # Step 6: Combine into colored point cloud
        print("Step 6: Creating colored point cloud...")
        colored_points = np.hstack((
            lidar_valid,                    # (N, 3) - x, y, z in LiDAR frame
            colors_rgb.astype(np.float32)   # (N, 3) - R, G, B
        ))  # Result: (N, 6)
        
        print(f"✅ Created colored point cloud with {len(colored_points)} points")
        print("="*70)
        
        # Create visualization image showing projected points
        vis_img = img_bgr.copy()
        
        # Color points by depth for visualization
        Z_norm = (Z_valid - Z_valid.min()) / (Z_valid.max() - Z_valid.min() + 1e-6)
        Z_color = (Z_norm * 255).astype(np.uint8)
        depth_colors = cv2.applyColorMap(Z_color, cv2.COLORMAP_JET)
        
        for (x, y, c) in zip(u_valid, v_valid, depth_colors):
            cv2.circle(vis_img, (x, y), 2, tuple(int(a) for a in c[0]), -1)
        
        # Add info text
        cv2.putText(vis_img, f"Visible Points: {len(colored_points)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return colored_points, vis_img
    
    def save_ply(self, colored_points, output_path):
        """
        Save colored point cloud as PLY file.
        
        WHY: PLY is a standard 3D format that stores geometry + color
        WHAT: Creates ASCII PLY file with x, y, z, R, G, B per point
        HOW: Writes header + data in text format
        
        PLY Format Structure:
        ---------------------
        ply
        format ascii 1.0
        element vertex N
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        x1 y1 z1 R1 G1 B1
        x2 y2 z2 R2 G2 B2
        ...
        
        Args:
            colored_points: (N x 6) array [x, y, z, R, G, B]
            output_path: Where to save the .ply file
        """
        print(f"\nSaving PLY file to: {output_path}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract coordinates and colors
        xyz = colored_points[:, :3]
        rgb = colored_points[:, 3:6].astype(np.uint8)
        
        # Write PLY header
        ply_header = f'''ply
format ascii 1.0
element vertex {len(colored_points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
        
        # Write data
        with open(output_path, 'w') as f:
            f.write(ply_header)
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        
        print(f"✅ Saved {len(colored_points)} points to {output_path}")
    
    def visualize_ply(self, ply_path):
        """
        Visualize PLY file using Open3D.
        
        WHY: To see the colored 3D point cloud interactively
        WHAT: Opens interactive 3D viewer
        HOW: Uses Open3D's visualization tools
        
        Controls in Open3D viewer:
        - Mouse drag: Rotate view
        - Mouse wheel: Zoom
        - Shift + mouse drag: Pan
        - Press 'H': Show help
        - Press 'Q': Quit
        
        Args:
            ply_path: Path to .ply file
        """
        if not HAS_OPEN3D:
            print("⚠️ Open3D not available. Cannot visualize.")
            return
        
        print(f"\nLoading PLY file for visualization: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        
        print(f"Point cloud loaded with {len(pcd.points)} points")
        print("\nOpening 3D viewer...")
        print("Controls:")
        print("  - Mouse drag: Rotate")
        print("  - Mouse wheel: Zoom")
        print("  - Shift + drag: Pan")
        print("  - Press 'H': Help")
        print("  - Press 'Q': Quit")
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Colored LiDAR Point Cloud",
            width=1280,
            height=720,
            left=50,
            top=50
        )


def main():
    """
    Main function - processes one frame and creates colored point cloud.
    """
    print("\n" + "="*70)
    print("UGV COLORED POINT CLOUD GENERATOR")
    print("="*70)
    
    # Configuration
    bag_path = '/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3'
    frame_number = 261  # Frame that was used for calibration
    output_dir = Path("colored_pointclouds")
    output_dir.mkdir(exist_ok=True)
    
    output_ply = output_dir / f"ugv_colored_frame_{frame_number:06d}.ply"
    output_vis = output_dir / f"ugv_projection_frame_{frame_number:06d}.jpg"
    
    print(f"\nConfiguration:")
    print(f"  Bag file: {bag_path}")
    print(f"  Frame: {frame_number}")
    print(f"  Output PLY: {output_ply}")
    print(f"  Output visualization: {output_vis}")
    
    # Create generator
    generator = ColoredPointCloudGenerator()
    
    # Load data
    img, points = generator.load_frame_from_bag(bag_path, frame_number)
    
    # Create colored point cloud
    colored_pc, vis_img = generator.create_colored_pointcloud(img, points)
    
    if colored_pc is None:
        print("\n❌ Failed to create colored point cloud!")
        print("   Check calibration parameters and try different frame.")
        return
    
    # Save PLY file
    generator.save_ply(colored_pc, str(output_ply))
    
    # Save visualization image
    cv2.imwrite(str(output_vis), vis_img)
    print(f"✅ Saved projection visualization to {output_vis}")
    
    # Visualize in 3D
    print("\n" + "="*70)
    print("Opening 3D visualization...")
    print("="*70)
    generator.visualize_ply(str(output_ply))
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)
    print(f"Your colored point cloud is saved at: {output_ply}")
    print("You can open it with:")
    print(f"  - Open3D: python -c 'import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud(\"{output_ply}\")])'")
    print(f"  - CloudCompare: cloudcompare {output_ply}")
    print(f"  - MeshLab: meshlab {output_ply}")
    print("="*70)


if __name__ == '__main__':
    main()