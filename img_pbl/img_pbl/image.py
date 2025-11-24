#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.bridge = CvBridge()
        self.image_dir = '/home/pinaka/dataset/rellis3d/Rellis-3D/image/00000/pylon_camera_node'
        self.image_list = sorted([os.path.join(self.image_dir, f)
                                  for f in os.listdir(self.image_dir)
                                  if f.endswith(('.jpg', '.png'))])
        self.index = 0
        self.get_logger().info('Image publisher initialized.')

    def timer_callback(self):
        if self.index >= len(self.image_list):
            self.index = 0
        img = cv2.imread(self.image_list[self.index])
        if img is not None:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published image {self.image_list[self.index]}')
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
