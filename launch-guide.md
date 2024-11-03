# ROS2 Cube Localizer Launch Guide

## Prerequisites

Before launching the cube_localizer package after a system reboot, ensure all dependencies and configurations are properly set up.

### Required Packages
```bash
# ROS2 TF Transformations
sudo apt-get install ros-humble-tf-transformations

# RViz2 (ensure non-snap version)
sudo apt-get install ros-humble-rviz2

# SLLidar ROS2 package (if not already installed)
sudo apt-get install ros-humble-sllidar-ros2
```

### Hardware Setup
1. Connect your BNO055 IMU to USB port
2. Connect your SLLidar to USB port
3. Verify device permissions:
```bash
ls -l /dev/ttyUSB*
```
4. Set up udev rules if needed:
```bash
sudo nano /etc/udev/rules.d/99-usb-serial.rules
```
Add these lines:
```
KERNEL=="ttyUSB*", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0666", SYMLINK+="lidar"
KERNEL=="ttyUSB*", ATTRS{manufacturer}=="Bosch Sensortec GmbH", MODE:="0666", SYMLINK+="imu"
```
Then reload udev rules:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Launch Sequence

### 1. Source ROS2 Environment
```bash
source /opt/ros/humble/setup.bash
```

### 2. Navigate to Workspace and Source
```bash
cd ~/ros_rlr
source install/setup.bash
```

### 3. Launch the Package
```bash
ros2 launch cube_localizer cube_localizer.launch.py
```

## Verification Steps

1. Check RViz2 visualization:
   - Grid should be visible
   - Cube marker should appear
   - LiDAR scan data should be visible
   - TF frames should be properly connected

2. Verify topic publications:
```bash
ros2 topic list
```
Expected topics:
- `/imu/data`
- `/scan`
- `/cube_pose`
- `/cube_marker`
- `/adjusted_scan`

3. Check TF tree:
```bash
ros2 run tf2_tools view_frames
```

## Troubleshooting

### Common Issues and Solutions

1. **TF Transformation Error**
   - If you see "cannot transform scan to map", try relaunching the file
   - Error usually resolves after system fully initializes

2. **Device Permission Issues**
```bash
# Check device permissions
ls -l /dev/ttyUSB*

# If needed, add user to dialout group
sudo usermod -a -G dialout $USER
```

3. **IMU Not Found**
```bash
# List USB devices
ls /dev/ttyUSB*

# Check IMU connection
ros2 topic echo /imu/data
```

4. **LiDAR Not Found**
```bash
# Verify LiDAR connection
ros2 topic echo /scan

# Check LiDAR node status
ros2 node list | grep sllidar
```

## Optional: Create Launch Script

Create a convenient launch script:

1. Create script file:
```bash
nano ~/launch_cube_localizer.sh
```

2. Add content:
```bash
#!/bin/bash
source /opt/ros/humble/setup.bash
cd ~/ros_rlr
source install/setup.bash
ros2 launch cube_localizer cube_localizer.launch.py
```

3. Make executable:
```bash
chmod +x ~/launch_cube_localizer.sh
```

4. Launch using:
```bash
~/launch_cube_localizer.sh
```

## Optional: Create Systemd Service

For automatic startup:

1. Create service file:
```bash
sudo nano /etc/systemd/system/cube-localizer.service
```

2. Add content:
```ini
[Unit]
Description=Cube Localizer ROS2 Launch
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/bin/bash -c 'source /opt/ros/humble/setup.bash && cd /home/YOUR_USERNAME/ros_rlr && source install/setup.bash && ros2 launch cube_localizer cube_localizer.launch.py'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

3. Enable and start service:
```bash
sudo systemctl enable cube-localizer.service
sudo systemctl start cube-localizer.service
```

## Notes

- Always ensure proper hardware connections before launching
- Run from native terminal rather than VS Code terminal for RViz
- Allow a few seconds for all nodes to initialize properly
- Monitor system resources if running automatically at startup

