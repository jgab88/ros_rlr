import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import math

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

        # Get the quaternion orientation from the IMU data
        orientation = msg.orientation

        # Convert quaternion to Euler angles
        _, _, yaw = self.euler_from_quaternion(orientation)
        self.heading = yaw

        # Simulate height sensor data (0.5 to 1.5 meters)
        self.height = 0.5 + 0.5 * math.sin(current_time.nanoseconds / 1e9)

        # Simulate speed sensor data (0 to 1 m/s)
        self.speed = 0.5 + 0.5 * math.cos(current_time.nanoseconds / 1e9)

        # Update cube position based on speed and heading
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.z = self.height

        # Publish cube pose
        pose = PoseStamped()
        pose.header.stamp = current_time.to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = self.z
        pose.pose.orientation = orientation
        self.pose_pub.publish(pose)

        # Publish cube marker
        marker = Marker()
        marker.header.stamp = current_time.to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'cube'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = self.z
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.orientation = orientation
        self.marker_pub.publish(marker)

    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    localizer = CubeLocalizer()
    rclpy.spin(localizer)
    localizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()