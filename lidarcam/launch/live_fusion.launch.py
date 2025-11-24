from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    bag_path = '/home/pinaka/dataset/AVMI/offroad_run_1_0.db3'

    return LaunchDescription([
        # ExecuteProcess(
        #     cmd=['ros2', 'bag', 'play', bag_path, '--clock', '--loop'],
        #     output='screen'
        # ),

        Node(
            package='lidarcam',
            executable='lidar_cam_fusion_live',
            name='lidar_cam_fusion_live',
            output='screen',
            parameters=[{
                'image_topic': '/camera/image_raw',
                'lidar_topic': '/lidar/points',
                'fusion_topic': '/fusion/projected_image',
                'color_proj_topic': '/fusion/color_projection_image',
                'rate_hz': 10.0,
            }],
        ),

        # ExecuteProcess(
        #     cmd=[
        #         'rviz2',
        #         '-d', '/home/pinaka/ros2_ws/src/lidarcam/rviz/fusion_view.rviz'
        #     ],
        #     output='screen'
        # ),
    ])
