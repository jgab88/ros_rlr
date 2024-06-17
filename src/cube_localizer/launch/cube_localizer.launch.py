import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('cube_localizer'), 'config.rviz')

    return LaunchDescription([
        Node(
            package='cube_localizer',
            executable='cube_localizer',
            name='cube_localizer_node'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config]
        ),

        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_scan_publisher_node',
            parameters=[
                {'serial_port': '/dev/ttyUSB0'},
                {'serial_baudrate': 115200},
                {'frame_id': 'laser_frame'},
                {'inverted': False},
                {'angle_compensate': True}
            ]
        ),

    ])
