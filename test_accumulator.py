import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import open3d as o3d
from rclpy.time import Time
import threading
import copy
import time

class MeshGenerator(Node):
    def __init__(self):
        super().__init__('mesh_generator')
        self.scan_sub = self.create_subscription(LaserScan, '/adjusted_scan', self.scan_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.accumulated_cloud = o3d.geometry.PointCloud()
        self.last_accumulation_time = self.get_clock().now()
        self.accumulation_interval = 0.1  # Accumulate points every 0.1 seconds
        self.mesh_update_interval = 5.0  # Update mesh every 5 seconds
        self.last_mesh_update_time = self.get_clock().now()
        self.lock = threading.Lock()
        self.max_points = 1000000  # Maximum number of points to keep
        self.marker_id = 0  # Unique ID for each marker

        # Create a publisher for the mesh as a marker array
        self.mesh_pub = self.create_publisher(MarkerArray, '/cube_marker_array', 10)

        # Create a timer for periodic mesh updates
        self.create_timer(1.0, self.mesh_update_timer_callback)

    def scan_callback(self, msg):
        current_time = self.get_clock().now()
        if (current_time - self.last_accumulation_time).nanoseconds / 1e9 < self.accumulation_interval:
            return

        self.last_accumulation_time = current_time
        self.get_logger().info(f"Processing scan at time {current_time.nanoseconds / 1e9:.2f}")

        try:
            transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform: {ex}")
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        
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

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

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

        with self.lock:
            self.accumulated_cloud += cloud
            # Implement sliding window
            if len(self.accumulated_cloud.points) > self.max_points:
                self.accumulated_cloud = self.accumulated_cloud.select_by_index(range(len(self.accumulated_cloud.points) - self.max_points, len(self.accumulated_cloud.points)))

        self.get_logger().info(f"Accumulated {len(points)} points. Total: {len(self.accumulated_cloud.points)}")

    def mesh_update_timer_callback(self):
        current_time = self.get_clock().now()
        if (current_time - self.last_mesh_update_time).nanoseconds / 1e9 >= self.mesh_update_interval:
            self.update_and_publish_mesh()
            self.last_mesh_update_time = current_time

    def update_and_publish_mesh(self):
        start_time = time.time()
        with self.lock:
            if len(self.accumulated_cloud.points) < 100:
                self.get_logger().warn("Not enough points to create a mesh")
                return

            # Create a copy of the point cloud for mesh generation
            cloud_copy = copy.deepcopy(self.accumulated_cloud)

        self.get_logger().info(f"Generating mesh from {len(cloud_copy.points)} points")

        try:
            # Downsample the point cloud
            downsampled_cloud = cloud_copy.voxel_down_sample(voxel_size=0.02)

            # Estimate normals
            downsampled_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

            # Create mesh
            distances = downsampled_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                downsampled_cloud, o3d.utility.DoubleVector([radius, radius * 2]))

            # Simplify mesh
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

            # Mesh cleaning
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Convert mesh to marker array
            marker_array = self.mesh_to_marker_array(mesh)

            # Publish marker array
            self.mesh_pub.publish(marker_array)
            end_time = time.time()
            self.get_logger().info(f"Mesh published with {len(mesh.triangles)} triangles. Processing time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            self.get_logger().error(f"Error in mesh generation: {str(e)}")

    def mesh_to_marker_array(self, mesh):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mesh"
        marker.id = self.marker_id
        self.marker_id += 1  # Increment the ID for the next marker
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        for triangle in triangles:
            for vertex_id in triangle:
                point = vertices[vertex_id]
                marker.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))

        marker_array.markers.append(marker)

        # Add a deletion marker for the previous mesh
        deletion_marker = Marker()
        deletion_marker.header.frame_id = "map"
        deletion_marker.header.stamp = self.get_clock().now().to_msg()
        deletion_marker.ns = "mesh"
        deletion_marker.id = self.marker_id - 2  # ID of the previous mesh
        deletion_marker.action = Marker.DELETE
        marker_array.markers.append(deletion_marker)

        return marker_array

def main(args=None):
    rclpy.init(args=args)
    mesh_generator = MeshGenerator()
    
    try:
        rclpy.spin(mesh_generator)
    except KeyboardInterrupt:
        pass
    finally:
        mesh_generator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()