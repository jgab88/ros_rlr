import rclpy
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import TransformStamped
import numpy as np
import open3d as o3d
from rclpy.node import Node

class PointCloudAccumulator(Node):
    def __init__(self):
        super().__init__('pointcloud_accumulator')
        self.scan_sub = self.create_subscription(LaserScan, '/adjusted_scan', self.scan_callback, 10)
        self.tf_sub = self.create_subscription(TransformStamped, '/tf', self.tf_callback, 10)
        self.accumulated_cloud = o3d.geometry.PointCloud()

    def scan_callback(self, msg):
        # Convert LaserScan to pointcloud
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        points = np.array([
            np.cos(angles) * msg.ranges,
            np.sin(angles) * msg.ranges,
            np.zeros_like(msg.ranges)
        ]).T

        # Create Open3D pointcloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # Transform cloud based on latest TF
        cloud.transform(self.latest_transform)

        # Accumulate
        self.accumulated_cloud += cloud

    def tf_callback(self, msg):
        # Update latest transform
        self.latest_transform = np.array([
            [msg.transform.rotation.w, -msg.transform.rotation.z, msg.transform.rotation.y, msg.transform.translation.x],
            [msg.transform.rotation.z, msg.transform.rotation.w, -msg.transform.rotation.x, msg.transform.translation.y],
            [-msg.transform.rotation.y, msg.transform.rotation.x, msg.transform.rotation.w, msg.transform.translation.z],
            [0, 0, 0, 1]
        ])

    def save_pointcloud(self):
        o3d.io.write_point_cloud("accumulated_cloud.ply", self.accumulated_cloud)

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