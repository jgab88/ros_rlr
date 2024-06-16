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
        )
    ])