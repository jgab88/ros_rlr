import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import TransformStamped, PointStamped
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from laser_geometry import LaserProjection
import numpy as np
from sensor_msgs_py import point_cloud2

class ScanToPointCloud(Node):
    def __init__(self):
        super().__init__('scan_to_pointcloud')
        self.scan_sub = self.create_subscription(LaserScan, '/adjusted_scan', self.scan_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.laser_projector = LaserProjection()
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/accumulated_point_cloud', 10)
        self.max_points = 10000  # Limit the number of points

    def scan_callback(self, msg):
        try:
            # Project the laser scan into a point cloud
            cloud = self.laser_projector.projectLaser(msg)
            
            # Look up the transform from the laser frame to the map frame
            transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rclpy.time.Time())
            
            # Apply the transform to the point cloud
            cloud_transformed = self.transform_point_cloud(cloud, transform)
            
            # Publish the transformed point cloud
            self.point_cloud_pub.publish(cloud_transformed)

        except TransformException as ex:
            self.get_logger().error(f'Could not transform scan: {ex}')
        except Exception as e:
            self.get_logger().error(f'Error processing scan: {str(e)}')
        
    def transform_point_cloud(self, cloud, transform):
        points = point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
        transformed_points = []
        
        for point in points:
            # Apply the transform to each point
            point_stamped = PointStamped()
            point_stamped.point.x = float(point[0])
            point_stamped.point.y = float(point[1])
            point_stamped.point.z = float(point[2])
            point_transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            transformed_points.append([float(point_transformed.point.x), 
                                       float(point_transformed.point.y), 
                                       float(point_transformed.point.z)])

        # Limit the number of points
        if len(transformed_points) > self.max_points:
            transformed_points = transformed_points[:self.max_points]

        fields = [point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                  point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                  point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1)]

        header = cloud.header
        header.frame_id = 'map'  # Set the frame to 'map'
        
        self.get_logger().info(f"Sample point: {transformed_points[0] if transformed_points else 'No points'}")

        return point_cloud2.create_cloud(header, fields, transformed_points)

def main(args=None):
    rclpy.init(args=args)
    node = ScanToPointCloud()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()