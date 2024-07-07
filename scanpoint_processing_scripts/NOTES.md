**Launch cube_localizer package:**
    ros2 launch cube_localizer cube_localizer.launch.py

**Run PointCloud2 node script on a seperate terminal window:**
    python3 scan_to_pintcloud.py 
    #This should be implemented into cube_localizer so that it runs all in one launch cmd

**To reset the position of the virtual cube which represents the physical lidar:**
    ros2 service call /reset_cube std_srvs/srv/Empty

**To start recording the associated topics (adjusted_scan, tf, tf_static, accumulated_point_cloud):**
    ros2 bag record /adjusted_scan /tf /tf_static /accumulated_point_cloud

**One should start recording when the cube is at it's origin. To do this, we can call the reset and rosbag command together:**
    ros2 service call /reset_cube std_srvs/srv/Empty && ros2 bag record /adjusted_scan /tf /tf_static /accumulated_point_cloud

    **To stop recording:**
    press Ctrl + C on the terminal window which rosbag was initiated.

**To process the rosbag data:**
    *Edit pointcloud_to_ply.py (line: storage_options = StorageOptions(uri=YourRosbagFile))