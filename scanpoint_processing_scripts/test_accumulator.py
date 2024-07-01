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
import concurrent.futures
from queue import Queue, PriorityQueue
import os

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
        self.max_points = 100000  # Maximum number of points to keep

        # Mesh generation variables
        self.mesh_id = 0
        self.processing_meshes = set()
        self.completed_meshes = PriorityQueue()
        self.mesh_queue = Queue()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # Adjust as needed
        self.mesh_generation_thread = threading.Thread(target=self.mesh_generation_worker)
        self.mesh_generation_thread.start()

        # Create a publisher for the mesh as a marker array
        self.mesh_pub = self.create_publisher(MarkerArray, '/cube_marker_array', 10)

        # Create a timer for periodic mesh updates
        self.create_timer(1.0, self.mesh_update_timer_callback)

        # Create a timer for publishing completed meshes
        self.create_timer(0.1, self.publish_completed_meshes)

        self.next_mesh_to_publish = 1

        # Complete mesh saving variables
        self.total_scans = 0
        self.save_interval = 100  # Save mesh every 100 scans
        self.output_directory = "output_meshes"  # Directory to save mesh files
        os.makedirs(self.output_directory, exist_ok=True)

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
            if len(self.accumulated_cloud.points) > self.max_points:
                excess_points = len(self.accumulated_cloud.points) - self.max_points
                self.accumulated_cloud = self.accumulated_cloud.select_by_index(range(excess_points, len(self.accumulated_cloud.points)))

        self.get_logger().info(f"Accumulated {len(points)} points. Total: {len(self.accumulated_cloud.points)}")

        self.total_scans += 1
        if self.total_scans % self.save_interval == 0:
            self.save_complete_mesh()

    def mesh_update_timer_callback(self):
        current_time = self.get_clock().now()
        if (current_time - self.last_mesh_update_time).nanoseconds / 1e9 >= self.mesh_update_interval:
            self.mesh_id += 1
            self.mesh_queue.put(self.mesh_id)
            self.last_mesh_update_time = current_time

    def mesh_generation_worker(self):
        while rclpy.ok():
            mesh_id = self.mesh_queue.get()
            if mesh_id is None:
                break
            if mesh_id not in self.processing_meshes:
                self.processing_meshes.add(mesh_id)
                self.thread_pool.submit(self.generate_and_publish_mesh, mesh_id)

    def generate_and_publish_mesh(self, mesh_id):
        start_time = time.time()
        try:
            with self.lock:
                cloud_copy = copy.deepcopy(self.accumulated_cloud)

            self.get_logger().info(f"Generating mesh {mesh_id} from {len(cloud_copy.points)} points")

            # Downsample the point cloud
            voxel_size = 0.02 if len(cloud_copy.points) < 20000 else 0.05
            downsampled_cloud = cloud_copy.voxel_down_sample(voxel_size=voxel_size)

            # Estimate normals
            downsampled_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

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

            marker_array = self.mesh_to_marker_array(mesh, mesh_id)
            self.completed_meshes.put((mesh_id, marker_array))
            
            self.get_logger().info(f"Mesh {mesh_id} generated with {len(mesh.triangles)} triangles. Processing time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            self.get_logger().error(f"Error generating mesh {mesh_id}: {str(e)}")
        finally:
            self.processing_meshes.remove(mesh_id)

    def publish_completed_meshes(self):
        while not self.completed_meshes.empty():
            mesh_id, marker_array = self.completed_meshes.queue[0]  # Peek at the top item
            if mesh_id == self.next_mesh_to_publish:
                self.completed_meshes.get()  # Remove the item from the queue
                self.mesh_pub.publish(marker_array)
                self.get_logger().info(f"Published mesh {mesh_id}")
                self.next_mesh_to_publish += 1
            else:
                break  # Wait for the next mesh in sequence

    def mesh_to_marker_array(self, mesh, mesh_id):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mesh"
        marker.id = mesh_id
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

        # Add a deletion marker for the previous mesh
        if mesh_id > 1:
            deletion_marker = Marker()
            deletion_marker.header.frame_id = "map"
            deletion_marker.header.stamp = self.get_clock().now().to_msg()
            deletion_marker.ns = "mesh"
            deletion_marker.id = mesh_id - 1
            deletion_marker.action = Marker.DELETE
            marker_array.markers.append(deletion_marker)

        marker_array.markers.append(marker)
        return marker_array

    def save_complete_mesh(self):
        start_time = time.time()
        try:
            with self.lock:
                cloud_copy = copy.deepcopy(self.accumulated_cloud)

            self.get_logger().info(f"Generating complete mesh from {len(cloud_copy.points)} points")

            # Downsample the point cloud
            voxel_size = 0.02 if len(cloud_copy.points) < 20000 else 0.05
            downsampled_cloud = cloud_copy.voxel_down_sample(voxel_size=voxel_size)

            # Estimate normals
            downsampled_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # Create mesh
            distances = downsampled_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                downsampled_cloud, o3d.utility.DoubleVector([radius, radius * 2]))

            # Simplify mesh
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)  # Increased for final mesh

            # Mesh cleaning
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            # Save mesh
            filename = os.path.join(self.output_directory, f"complete_mesh_{self.total_scans}.ply")
            o3d.io.write_triangle_mesh(filename, mesh)
            
            self.get_logger().info(f"Complete mesh saved to {filename}. Processing time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            self.get_logger().error(f"Error saving complete mesh: {str(e)}")

    def destroy_node(self):
        # Save final mesh before shutting down
        self.save_complete_mesh()
        self.mesh_queue.put(None)  # Signal the worker thread to exit
        self.mesh_generation_thread.join()
        self.thread_pool.shutdown()
        super().destroy_node()

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