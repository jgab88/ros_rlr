import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs_py import point_cloud2
import tf2_ros

def main():
    rclpy.init()
    
    node = Node("pointcloud_processor")

    storage_options = StorageOptions(uri='/home/jg/ros_rlr/scanpoint_processing_scripts/rosbag2_2024_06_30-20_32_38/rosbag2_2024_06_30-20_32_38_0.db3', storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    all_points = []
    scanner_path = []
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        if topic == '/accumulated_point_cloud':
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array([(p['x'], p['y'], p['z']) for p in pc_data], dtype=np.float64)
            all_points.append(points)

        elif topic == '/tf' or topic == '/tf_static':
            msg_type = get_message(type_map[topic])
            tf_msg = deserialize_message(data, msg_type)
            for transform in tf_msg.transforms:
                if transform.child_frame_id == 'laser_frame':  # or whatever frame your scanner uses
                    scanner_path.append([
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ])

    if not all_points:
        print("No point cloud data found in the bag file.")
        return

    combined_points = np.vstack(all_points)
    scanner_path = np.array(scanner_path)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)

    # Process point cloud (filtering, normal estimation, etc.)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create meshes
    print("Creating Poisson mesh...")
    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=False)
    
    print("Creating BPA mesh...")
    radii = [0.05, 0.1, 0.2, 0.4]
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # Save point cloud with original positioning
    o3d.io.write_point_cloud("output/pipe_scan.pcd", pcd)

    # Save meshes
    o3d.io.write_triangle_mesh("output/pipe_mesh_poisson.ply", poisson_mesh)
    o3d.io.write_triangle_mesh("output/pipe_mesh_bpa.ply", bpa_mesh)

    # Save scanner path
    np.savetxt("output/scanner_path.txt", scanner_path)

    # Create and save a simple spline from the scanner path
    path_pcd = o3d.geometry.PointCloud()
    path_pcd.points = o3d.utility.Vector3dVector(scanner_path)
    o3d.io.write_point_cloud("output/scanner_path_spline.pcd", path_pcd)

    print("Point cloud, meshes, and scanner path saved.")

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()