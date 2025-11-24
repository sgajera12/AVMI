#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

try:
    import open3d as o3d
except ImportError:
    o3d = None

def create_cloud(header, points):
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    # toconvert Nx4 numpy array (x,y,z,intensity) > raw bytes
    cloud_bytes = points.astype(np.float32).tobytes()

    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=16,
        row_step=16 * len(points),
        data=cloud_bytes,
    )


class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'lidar/points', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)  # 2 Hz
        self.data_dir = '/home/pinaka/dataset/rellis3d/Rellis-3D/lidar/00000/os1_cloud_node_kitti_bin'
        self.files = sorted([os.path.join(self.data_dir, f)
                             for f in os.listdir(self.data_dir)
                             if f.endswith('.bin') or f.endswith('.ply')])
        self.index = 0
        self.get_logger().info(f'Found {len(self.files)} pointcloud files.')

    def timer_callback(self):
        if not self.files:
            self.get_logger().warn('No LiDAR files found.')
            return

        file = self.files[self.index]
        if file.endswith('.bin'):
            points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
        elif file.endswith('.ply') and o3d is not None:
            pc = o3d.io.read_point_cloud(file)
            points = np.asarray(pc.points)
            intensities = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack((points, intensities))
        else:
            self.get_logger().warn(f'Skipping {file}')
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'lidar_link'
        cloud_msg = create_cloud(header, points)
        self.publisher_.publish(cloud_msg)
        self.get_logger().info(f'Published LiDAR frame: {os.path.basename(file)}')

        self.index = (self.index + 1) % len(self.files)


def main(args=None):
    rclpy.init(args=args)
    node = LidarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()