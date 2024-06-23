import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import open3d as o3d
from rclpy.time import Time
import threading
import os

class PointCloudAccumulator(Node):
    def __init__(self):
        super().__init__('pointcloud_accumulator')
        self.scan_sub = self.create_subscription(LaserScan, '/adjusted_scan', self.scan_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.accumulated_cloud = o3d.geometry.PointCloud()
        self.last_accumulation_time = self.get_clock().now()
        self.accumulation_interval = 0.1  # Accumulate points every 0.1 seconds
        self.lock = threading.Lock()

        # Check if DISPLAY is set
        display_var = os.environ.get('DISPLAY')
        if not display_var:
            self.get_logger().error("DISPLAY environment variable is not set. Cannot open GUI window.")
            return

        print(f"DISPLAY variable is set to {display_var}")
        
        # Set up Open3D visualization
        self.vis = o3d.visualization.Visualizer()
        if self.vis.create_window():
            print("Open3D visualization window created successfully.")
        else:
            print("Failed to create Open3D visualization window.")
        self.vis.add_geometry(self.accumulated_cloud)
        self.vis_thread = threading.Thread(target=self.visualization_loop)
        self.vis_thread.start()

    def scan_callback(self, msg):
        current_time = self.get_clock().now()
        if (current_time - self.last_accumulation_time).nanoseconds / 1e9 < self.accumulation_interval:
            return

        self.last_accumulation_time = current_time
        print(f"Processing scan at time {current_time.nanoseconds / 1e9:.2f}")

        try:
            transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time())
        except TransformException as ex:
            print(f"Could not transform: {ex}")
            return

        # Convert LaserScan to pointcloud
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        
        # Filter out invalid measurements
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        angles = angles[valid]
        ranges = ranges[valid]
        
        if len(angles) == 0 or len(ranges) == 0:
            print("No valid points in this scan")
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
        with self.lock:
            self.accumulated_cloud += cloud

        print(f"Accumulated {len(points)} points. Total: {len(self.accumulated_cloud.points)}")

    def visualization_loop(self):
        while rclpy.ok():
            with self.lock:
                self.vis.update_geometry(self.accumulated_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def create_mesh(self):
        with self.lock:
            if len(self.accumulated_cloud.points) < 100:
                self.get_logger().warn("Not enough points to create a mesh")
                return None

        self.get_logger().info(f"Starting mesh creation with {len(self.accumulated_cloud.points)} points")

        # Downsample the point cloud
        downsampled_cloud = self.accumulated_cloud.voxel_down_sample(voxel_size=0.05)
        self.get_logger().info(f"Downsampled to {len(downsampled_cloud.points)} points")

        # Estimate normals
        downsampled_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.get_logger().info("Normals estimated")

        # Create mesh
        distances = downsampled_cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        self.get_logger().info(f"Creating mesh with radius {radius}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            downsampled_cloud, o3d.utility.DoubleVector([radius, radius * 2]))

        if len(mesh.triangles) == 0:
            self.get_logger().warn("No triangles created in the mesh")
            return None

        self.get_logger().info(f"Mesh created with {len(mesh.triangles)} triangles")

        # Simplify mesh
        original_triangle_count = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)
        self.get_logger().info(f"Mesh simplified from {original_triangle_count} to {len(mesh.triangles)} triangles")

        return mesh
    
    def save_pointcloud_and_mesh(self):
        with self.lock:
            point_cloud_path = os.path.join(os.getcwd(), "accumulated_cloud.ply")
            o3d.io.write_point_cloud(point_cloud_path, self.accumulated_cloud)
            self.get_logger().info(f"Saved point cloud with {len(self.accumulated_cloud.points)} points to {point_cloud_path}")

        self.get_logger().info("Starting mesh creation...")
        mesh = self.create_mesh()
        if mesh:
            mesh_path = os.path.join(os.getcwd(), "reconstructed_mesh.ply")
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            self.get_logger().info(f"Saved reconstructed mesh to {mesh_path}")
        else:
            self.get_logger().warn("Failed to create mesh")

def main(args=None):
    rclpy.init(args=args)
    accumulator = PointCloudAccumulator()
    if accumulator.vis:
        try:
            rclpy.spin(accumulator)
        except KeyboardInterrupt:
            pass
        finally:
            accumulator.save_pointcloud_and_mesh()
            accumulator.vis.destroy_window()
            accumulator.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
