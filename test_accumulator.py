import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
import open3d as o3d
from rclpy.time import Time

class PointCloudAccumulator(Node):
    def __init__(self):
        super().__init__('pointcloud_accumulator')
        self.scan_sub = self.create_subscription(LaserScan, '/adjusted_scan', self.scan_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.accumulated_cloud = o3d.geometry.PointCloud()
        self.last_accumulation_time = self.get_clock().now()
        self.accumulation_interval = .5  # Accumulate points every 1 second

    def scan_callback(self, msg):
        current_time = self.get_clock().now()
        if (current_time - self.last_accumulation_time).nanoseconds / 1e9 < self.accumulation_interval:
            return

        self.last_accumulation_time = current_time
        self.get_logger().info(f"Processing scan at time {current_time.nanoseconds / 1e9:.2f}")

        try:
            transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time())
            self.get_logger().info(f"Transform: translation={transform.transform.translation}, rotation={transform.transform.rotation}")
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform: {ex}")
            return

        # Convert LaserScan to pointcloud
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        
        # Filter out invalid measurements
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]
        
        if len(angles) == 0 or len(ranges) == 0:
            self.get_logger().warn("No valid points in this scan")
            return
        
        points = np.column_stack((
            np.cos(angles) * ranges,
            np.sin(angles) * ranges,
            np.zeros_like(ranges)
        ))

        # Create Open3D pointcloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # Apply transform
        transform_matrix = np.array([
            [1-2*(transform.transform.rotation.y**2+transform.transform.rotation.z**2), 
             2*(transform.transform.rotation.x*transform.transform.rotation.y-transform.transform.rotation.w*transform.transform.rotation.z), 
             2*(transform.transform.rotation.x*transform.transform.rotation.z+transform.transform.rotation.w*transform.transform.rotation.y), 
             transform.transform.translation.x],
            [2*(transform.transform.rotation.x*transform.transform.rotation.y+transform.transform.rotation.w*transform.transform.rotation.z), 
             1-2*(transform.transform.rotation.x**2+transform.transform.rotation.z**2), 
             2*(transform.transform.rotation.y*transform.transform.rotation.z-transform.transform.rotation.w*transform.transform.rotation.x), 
             transform.transform.translation.y],
            [2*(transform.transform.rotation.x*transform.transform.rotation.z-transform.transform.rotation.w*transform.transform.rotation.y), 
             2*(transform.transform.rotation.y*transform.transform.rotation.z+transform.transform.rotation.w*transform.transform.rotation.x), 
             1-2*(transform.transform.rotation.x**2+transform.transform.rotation.y**2), 
             transform.transform.translation.z],
            [0, 0, 0, 1]
        ])
        cloud.transform(transform_matrix)

        # Accumulate
        self.accumulated_cloud += cloud

        self.get_logger().info(f"Accumulated {len(points)} points. Total: {len(self.accumulated_cloud.points)}")

    def save_pointcloud(self):
        o3d.io.write_point_cloud("accumulated_cloud.ply", self.accumulated_cloud)
        self.get_logger().info(f"Saved point cloud with {len(self.accumulated_cloud.points)} points.")

def main(args=None):
    rclpy.init(args=args)
    accumulator = PointCloudAccumulator()
    try:
        rclpy.spin(accumulator)
    except KeyboardInterrupt:
        pass
    finally:
        accumulator.save_pointcloud()
        accumulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()