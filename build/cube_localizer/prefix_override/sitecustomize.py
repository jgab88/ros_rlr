import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jg/pipe_scanning/ros_rlr/install/cube_localizer'
