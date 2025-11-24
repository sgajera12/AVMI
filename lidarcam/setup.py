from setuptools import find_packages, setup

package_name = 'lidarcam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/live_fusion.launch.py']),

    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'PyYAML', 'scipy'],
    zip_safe=True,
    maintainer='pinaka',
    maintainer_email='pinaka@todo.todo',
    description='Live camera + LiDAR + projection overlay publisher (Rellis-3D)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'live_fusion = lidarcam.live_fusion_node_copy:main',
            'lidar_cam_fusion_live = lidarcam.lidar_cam_fusion_live_copy:main',
            'bev_publish = lidarcam.bev_fusion_publisher:main',
            'lidar_cam_fusion = lidarcam.lidarcam_fusion_new:main',
            'lidar_cam_fusiontest = lidarcam.testlidarimg:main',

        ],
    },
)