import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from tf_transformations import quaternion_from_euler, quaternion_multiply
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty
import math
import tf2_ros
from rclpy.duration import Duration

class CubeLocalizer(Node):
    def __init__(self):
        super().__init__('cube_localizer')
        
        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, qos_profile)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.pose_pub = self.create_publisher(PoseStamped, 'cube_pose', 10)
        self.marker_pub = self.create_publisher(Marker, 'cube_marker', 10)
        self.adjusted_scan_pub = self.create_publisher(LaserScan, 'adjusted_scan', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.reset_service = self.create_service(Empty, 'reset_cube', self.reset_callback)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.height = 0.0
        self.heading = 0.0
        self.last_time = self.get_clock().now()

    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        orientation = msg.orientation
        _, _, yaw = self.euler_from_quaternion(orientation)
        self.heading = yaw

        self.height = 0.5 + 0.5 * math.sin(current_time.nanoseconds / 1e9)
        self.speed = 0.5 + 0.5 * math.cos(current_time.nanoseconds / 1e9)

        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.z = self.height

        pose = PoseStamped()
        pose.header.stamp = current_time.to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = self.z
        pose.pose.orientation = orientation
        self.pose_pub.publish(pose)

        marker = Marker()
        marker.header.stamp = current_time.to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'cube'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

        # Create a rotation to make the laser scan vertical (90 degrees around X-axis)
        vertical_rotation = quaternion_from_euler(0, math.pi/2, 0)
    
        # Combine the vertical rotation with the current orientation
        combined_rotation = quaternion_multiply(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            vertical_rotation
        )

        # Publish transform slightly into the future
        future_time = current_time + Duration(seconds=0.1)
        transform = TransformStamped()
        transform.header.stamp = future_time.to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'laser_frame'
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.translation.z = self.z
        transform.transform.rotation = Quaternion(x=combined_rotation[0], y=combined_rotation[1], 
                                                  z=combined_rotation[2], w=combined_rotation[3])
        self.tf_broadcaster.sendTransform(transform)

    def lidar_callback(self, msg):
        current_time = self.get_clock().now()

        self.get_logger().info(f"Received LaserScan message at {current_time.nanoseconds/1e9:.6f}")
        self.get_logger().info(f"Message timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
        self.get_logger().info(f"Frame ID: {msg.header.frame_id}")
        self.get_logger().info(f"Angle range: [{msg.angle_min:.4f}, {msg.angle_max:.4f}]")
        self.get_logger().info(f"Angle increment: {msg.angle_increment:.6f}")

        actual_min_range = max(0.1, min(r for r in msg.ranges if r > 0))
        actual_max_range = min(16.0, max(msg.ranges))
        
        self.get_logger().info(f"Actual range limits: [{actual_min_range:.2f}, {actual_max_range:.2f}]")
        self.get_logger().info(f"Total points: {len(msg.ranges)}")
        
        valid_ranges = [r for r in msg.ranges if actual_min_range <= r <= actual_max_range]
        self.get_logger().info(f"Valid points: {len(valid_ranges)}")

        if valid_ranges:
            self.get_logger().info(f"Range statistics: min={min(valid_ranges):.2f}, max={max(valid_ranges):.2f}, avg={np.mean(valid_ranges):.2f}")
        else:
            self.get_logger().warn("No valid ranges found")

        self.get_logger().info(f"Sample of range values: {msg.ranges[:10]}")

        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        time_diff = (current_time.nanoseconds - msg_time.nanoseconds) / 1e9
        self.get_logger().info(f"Time difference: {time_diff:.6f} seconds")

        # Create and publish adjusted LaserScan message
        # Use the same timestamp for LaserScan as for the transform
        future_time = current_time + Duration(seconds=0.1)
        adjusted_msg = LaserScan()
        adjusted_msg.header = msg.header
        adjusted_msg.header.stamp = future_time.to_msg()
        adjusted_msg.angle_min = msg.angle_min
        adjusted_msg.angle_max = msg.angle_max
        adjusted_msg.angle_increment = msg.angle_increment
        adjusted_msg.time_increment = msg.time_increment
        adjusted_msg.scan_time = msg.scan_time
        adjusted_msg.range_min = actual_min_range
        adjusted_msg.range_max = actual_max_range
        adjusted_msg.ranges = msg.ranges
        adjusted_msg.intensities = msg.intensities
        self.adjusted_scan_pub.publish(adjusted_msg)

        self.get_logger().info("LaserScan processing completed")
        self.get_logger().info("----------------------------")

    def euler_from_quaternion(self, quaternion):
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def reset_callback(self, request, response):
        self.x = self.y = self.z = self.speed = self.height = self.heading = 0.0
        self.last_time = self.get_clock().now()
        self.get_logger().info("Cube position reset to origin.")
        return response

def main(args=None):
    rclpy.init(args=args)
    localizer = CubeLocalizer()
    rclpy.spin(localizer)
    localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()