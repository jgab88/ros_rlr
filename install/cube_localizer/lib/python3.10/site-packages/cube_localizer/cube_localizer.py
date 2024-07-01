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

        self.pipe_radius = 0.6096  # 48 inches in meters 0.6096*
        self.radius_variation = 0.0075  # 5cm* variation in pipe radius 0.025*
        self.texture_variation = 0.0075  # 2cm* variation for surface texture 0.02*

        # Timer for simulating LaserScan data
        self.create_timer(0.1, self.simulate_laser_scan)  # 10Hz

    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        orientation = msg.orientation
        _, _, yaw = self.euler_from_quaternion(orientation)
        self.heading = yaw

        self.height = 0.1 + 0.1 * math.sin(current_time.nanoseconds / 1e9) #Height oscilation 0.5 + 0.5*
        self.speed = 0.5  # constant speed

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

        # Modify the transform to orient the laser scan correctly
        laser_orientation = quaternion_multiply(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            quaternion_from_euler(0, -math.pi/2, 0)  # Rotate 90 degrees around Y-axis
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
        transform.transform.rotation = Quaternion(x=laser_orientation[0], y=laser_orientation[1],
                                                  z=laser_orientation[2], w=laser_orientation[3])
        self.tf_broadcaster.sendTransform(transform)

    def simulate_laser_scan(self):
        current_time = self.get_clock().now()
        current_radius = self.pipe_radius + np.random.uniform(-self.radius_variation, self.radius_variation)
        
        num_points = 360
        angles = np.linspace(0, 2*np.pi, num_points)
        
        # Generate circular cross-section
        y = current_radius * np.cos(angles)
        z = current_radius * np.sin(angles)
        
        # Add turbulation to simulate surface texture
        texture = np.random.uniform(-self.texture_variation, self.texture_variation, num_points)
        ranges = np.sqrt(y**2 + z**2) + texture
        
        # Replace any invalid values with max_range
        ranges = np.clip(ranges, 0.1, 16.0)

        # Create and publish adjusted LaserScan message
        adjusted_msg = LaserScan()
        adjusted_msg.header.stamp = current_time.to_msg()
        adjusted_msg.header.frame_id = 'laser_frame'
        adjusted_msg.angle_min = 0.0  # Change this to 0.0 (float)
        adjusted_msg.angle_max = 2.0 * np.pi  # Change this to 2.0 * np.pi (float)
        adjusted_msg.angle_increment = (2.0 * np.pi) / num_points  # Ensure this is a float
        adjusted_msg.time_increment = 0.0
        adjusted_msg.scan_time = 0.1
        adjusted_msg.range_min = 0.1
        adjusted_msg.range_max = 16.0
        adjusted_msg.ranges = ranges.tolist()

        self.adjusted_scan_pub.publish(adjusted_msg)

        self.get_logger().info(f"Published simulated pipe scan. Radius: {current_radius:.3f}m")
        self.get_logger().info(f"Number of points: {num_points}")
        self.get_logger().info(f"Range statistics: min={np.min(ranges):.2f}, max={np.max(ranges):.2f}, avg={np.mean(ranges):.2f}")

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