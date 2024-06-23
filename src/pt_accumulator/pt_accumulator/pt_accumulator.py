import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
import numpy as np
import open3d as o3d

class PointCloudAccumulator(Node):
    def __init__(self):
        super().__init__('pointcloud_accumulator')
        self.scan_sub = self.create_subscription(LaserScan, '/adjusted_scan', self.scan_callback, 10)
        self.tf_sub = self.create_subscription(TransformStamped, '/tf', self.tf_callback, 10)
        self.accumulated_cloud = o3d.geometry.PointCloud()
        self.latest_transform = np.eye(4)

    def scan_callback(self, msg):
        self.get_logger().info(f"Received LaserScan with {len(msg.ranges)} ranges")
        
        # Convert LaserScan to pointcloud
        num_points = len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, num_points)
        ranges = np.array(msg.ranges)
        
        self.get_logger().info(f"Angles shape: {angles.shape}, Ranges shape: {ranges.shape}")
        
        # Filter out invalid measurements
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]
        
        self.get_logger().info(f"Valid points: {np.sum(valid)}")
        
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

        # Transform cloud based on latest TF
        cloud.transform(self.latest_transform)

        # Accumulate
        self.accumulated_cloud += cloud

        self.get_logger().info(f"Accumulated {len(points)} points. Total: {len(self.accumulated_cloud.points)}")

    def tf_callback(self, msg):
        # Update latest transform
        translation = msg.transform.translation
        rotation = msg.transform.rotation
        self.latest_transform = np.array([
            [1-2*(rotation.y**2+rotation.z**2), 2*(rotation.x*rotation.y-rotation.w*rotation.z), 2*(rotation.x*rotation.z+rotation.w*rotation.y), translation.x],
            [2*(rotation.x*rotation.y+rotation.w*rotation.z), 1-2*(rotation.x**2+rotation.z**2), 2*(rotation.y*rotation.z-rotation.w*rotation.x), translation.y],
            [2*(rotation.x*rotation.z-rotation.w*rotation.y), 2*(rotation.y*rotation.z+rotation.w*rotation.x), 1-2*(rotation.x**2+rotation.y**2), translation.z],
            [0, 0, 0, 1]
        ])

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