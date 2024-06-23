from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pt_accumulator',
            executable='pt_accumulator',
            name='pt_accumulator'
        )
    ])