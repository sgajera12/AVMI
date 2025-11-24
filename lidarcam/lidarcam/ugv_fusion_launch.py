"""
Launch file for UGV LiDAR-Camera Fusion

This launch file makes it easy to start the fusion node with different configurations.

USAGE:
------
Basic:
  ros2 launch ugv_fusion_launch.py

With custom bag file:
  ros2 launch ugv_fusion_launch.py bag_file:=/path/to/your/bag.db3

With custom rate:
  ros2 launch ugv_fusion_launch.py republish_rate_hz:=5.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate the launch description for the UGV fusion node.
    
    This sets up:
    1. Launch arguments (parameters you can override)
    2. The fusion node with these parameters
    """
    
    # Declare launch arguments with default values
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value='/home/pinaka/dataset/AVMI/run1_lidar_camera/run1_lidar_camera_0.db3',
        description='Path to the ROS2 bag file'
    )
    
    rate_arg = DeclareLaunchArgument(
        'republish_rate_hz',
        default_value='10.0',
        description='Rate (Hz) at which to republish messages from the bag'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Topic name for camera images'
    )
    
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/lidar/points2',
        description='Topic name for LiDAR point clouds'
    )
    
    fusion_topic_arg = DeclareLaunchArgument(
        'fusion_topic',
        default_value='/fusion/projected_image',
        description='Topic name for depth-colored fusion output'
    )
    
    color_proj_topic_arg = DeclareLaunchArgument(
        'color_proj_topic',
        default_value='/fusion/color_projection_image',
        description='Topic name for RGB-colored projection output'
    )
    
    # Create the fusion node
    fusion_node = Node(
        package='ugv_fusion',  # You'll need to create this package
        executable='ugv_fusion_node.py',
        name='ugv_fusion_node',
        output='screen',
        parameters=[{
            'bag_file': LaunchConfiguration('bag_file'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'lidar_topic': LaunchConfiguration('lidar_topic'),
            'fusion_topic': LaunchConfiguration('fusion_topic'),
            'color_proj_topic': LaunchConfiguration('color_proj_topic'),
            'republish_rate_hz': LaunchConfiguration('republish_rate_hz'),
        }]
    )
    
    return LaunchDescription([
        bag_file_arg,
        rate_arg,
        camera_topic_arg,
        lidar_topic_arg,
        fusion_topic_arg,
        color_proj_topic_arg,
        fusion_node,
    ])
