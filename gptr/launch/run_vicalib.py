# my_node_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import numpy as np

# Save path
result_save_path = '/media/tmn/mySataSSD1/Experiments/gptr_v2/logs/vicalib_/'

# Data path
vi_data_path = '/home/tmn/ros2_ws/src/gptr/gptr/dataVICalib/'

def generate_launch_description():
    
    # GPTR LO node
    gptr_vicalib_node = Node(
        package     = 'gptr',
        executable  = 'gptr_vicalib',  # Name of the executable built by your package
        name        = 'gptr_vicalib',  # Optional: gives the node instance a name
        output      = 'screen',   # Print the node output to the screen
        # prefix      = 'gdb -ex=r --args',
        parameters  =
        [
            {"auto_exit"         : 1},
            {"if_save_traj"      : 1},
            {"traj_save_path"    : result_save_path},
            {"data_path"         : vi_data_path},

            # Parameters for the Gaussian Process
            {"gpDt"              : 0.01},
            {"gpQr"              : 6.00},
            {"gpQc"              : 6.00},
            {"pose_type"         : "SO3xR3"}, # Choose 'SE3' or 'SO3xR3'
            {"lie_epsilon"       : 1e-2},
            {"use_approx_drv"    : 0},

            # Parameters for the solve
            {"SLIDE_SIZE"        : 2},
            {"WINDOW_SIZE"       : 20},
            {"w_corner"          : 8.0},
            {"GYR_N"             : 2000.0},
            {"GYR_W"             : 0.0},
            {"ACC_N"             : 100.0},
            {"ACC_W"             : 0.0},
            {"corner_loss_thres" :-1.0},
            {"mp_loss_thres"     :-1.0},
            
            # Time skewing factor
            {"tskew0"            : 1.0},
            {"tskewmax"          : 4.5},
            {"tskewstep"         : 0.1},
            {"Dtstep"            : [0.1, 0.2, 0.05]},
            
        ]  # Optional: pass parameters if needed
    )

    # Rviz node
    # rviz_node = Node(
    #     package     = 'rviz2',
    #     executable  = 'rviz2',
    #     name        = 'rviz2',
    #     output      = 'screen',
    #     arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_vicalib.rviz']
    # )

    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([gptr_vicalib_node,
                            #   rviz_node
                             ])