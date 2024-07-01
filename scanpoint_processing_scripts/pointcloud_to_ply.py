import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import numpy as np
import open3d as o3d
from sensor_msgs_py import point_cloud2
import tf2_ros
from tf_transformations import quaternion_matrix

def align_points_to_principal_axes(points):
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Compute principal axes
    covariance_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Rotate points
    aligned_points = np.dot(centered_points, eigenvectors)

    return aligned_points, centroid, eigenvectors

def transform_points(points, transform):
    # Create a 4x4 transformation matrix
    t = transform.transform
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    
    rot = t.rotation
    quat = [rot.x, rot.y, rot.z, rot.w]
    rot_matrix = quaternion_matrix(quat)
    trans_matrix[:3, :3] = rot_matrix[:3, :3]

    # Add homogeneous coordinate
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply transformation
    transformed_points = np.dot(trans_matrix, points_homogeneous.T).T
    
    # Remove homogeneous coordinate
    return transformed_points[:, :3]

def main():
    rclpy.init()
    
    node = Node("pointcloud_processor")

    storage_options = StorageOptions(uri='/home/jg/ros_rlr/scanpoint_processing_scripts/rosbag2_2024_06_30-20_47_37/rosbag2_2024_06_30-20_47_37_0.db3', storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    all_points = []
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic != '/accumulated_point_cloud':
            continue

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_data))
        points = points.view((np.float32, 3)).reshape(-1, 3)

        # Try to get the transform from the point cloud frame to the map frame
        try:
            transform = tf_buffer.lookup_transform('map', msg.header.frame_id, msg.header.stamp)
            points = transform_points(points, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"Failed to lookup transform for point cloud. Using untransformed points.")

        all_points.append(points)

    combined_points = np.vstack(all_points)

    # Align points to principal axes
    aligned_points, centroid, eigenvectors = align_points_to_principal_axes(combined_points)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(aligned_points)

    # Filtering: Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Downsampling
    pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Adjust voxel size as needed

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd])

    # Create meshes using different methods
    print("Creating Poisson mesh...")
    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=False)
    
    print("Creating BPA mesh...")
    radii = [0.05, 0.1, 0.2, 0.4]
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # Transform meshes back to original coordinate system
    poisson_mesh.rotate(eigenvectors.T, center=(0, 0, 0))
    poisson_mesh.translate(centroid)
    bpa_mesh.rotate(eigenvectors.T, center=(0, 0, 0))
    bpa_mesh.translate(centroid)

    # Visualize the results
    o3d.visualization.draw_geometries([poisson_mesh])
    o3d.visualization.draw_geometries([bpa_mesh])

    # Save the meshes
    o3d.io.write_triangle_mesh("output/pipe_mesh_poisson.ply", poisson_mesh)
    o3d.io.write_triangle_mesh("output/pipe_mesh_bpa.ply", bpa_mesh)

    print("Meshes saved as pipe_mesh_poisson.ply and pipe_mesh_bpa.ply")

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()