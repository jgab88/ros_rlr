from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ... (other launch configurations)
        Node(
            package='rplidar_ros2',
            executable='rplidar_scan_publisher',
            name='rplidar_scan_publisher_node',
            parameters=[
                {'serial_port': '/dev/ttyUSB0'},
                {'serial_baudrate': 115200},
                {'frame_id': 'laser_frame'},
                {'inverted': False},
                {'angle_compensate': True}
            ]
        )
    ])