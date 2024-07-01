import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import numpy as np
import open3d as o3d
from sensor_msgs_py import point_cloud2

def main():
    rclpy.init()

    storage_options = StorageOptions(uri='/home/jg/ros_rlr/rosbag2_2024_06_29-22_57_20/rosbag2_2024_06_29-22_57_20_0.db3', storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    counter = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        # Only process messages from the /accumulated_point_cloud topic
        if topic != '/accumulated_point_cloud':
            continue

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        # Read the point cloud data
        pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Convert to numpy array and reshape
        points = np.array(list(pc_data))
        points = points.view((np.float32, 3)).reshape(-1, 3)

        print(f"Points shape: {points.shape}")
        print(f"Points data type: {points.dtype}")
        print(f"First few points: {points[:5]}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(f"pointcloud_{counter}.pcd", pcd)
        
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} point clouds")

    print("Finished processing bag file")

if __name__ == '__main__':
    main()