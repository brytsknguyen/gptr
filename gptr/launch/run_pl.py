# my_node_launch.py
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, LogInfo, EmitEvent, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from math import pi as M_PI
from math import sqrt as sqrt

import numpy as np
import shutil, os

# # Direction to log the exp
log_dir_ = f'/media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lidar_mle_noise_300x6Hz_/'

# Type of pose
# pose_type_ = 'SE3'
pose_type_ = 'SO3xR3'

# Type of pose
use_approx_drv_ = '0'

# Knot length
deltaT_ = '0.05'

# Initial pose in each sequence
xyzypr_W_L0 =[ 0,   0,  0,   0, 0,  0,
               0.2, 0, -0.2, 0, 90, 0 ]

# Copy the current file to the log file
os.makedirs(log_dir_, exist_ok=True)
launch_file_name = os.path.realpath(__file__)
shutil.copy(launch_file_name, log_dir_)
print(launch_file_name)

def generate_launch_description():
    
    # lidar_bag_file  = DeclareLaunchArgument('lidar_bag_file', default_value=lidar_bag_file_, description='')   # Bag file
    log_dir         = DeclareLaunchArgument('log_dir', default_value=log_dir_, description='')                 # Direction to log the exp
    pose_type       = DeclareLaunchArgument('pose_type', default_value=pose_type_, description='')             # Variant of kinematics
    use_approx_drv  = DeclareLaunchArgument('use_approx_drv', default_value=use_approx_drv_, description='')   # Variant of approximation
    deltaT          = DeclareLaunchArgument('deltaT', default_value=deltaT_, description='')                   # Variant of approximation

    # GPTR LO node
    gptr_pl_node = Node(
        package     = 'gptr',
        executable  = 'gptr_pl',  # Name of the executable built by your package
        name        = 'gptr_pl',  # Optional: gives the node instance a name
        output      = 'screen',   # Print the node output to the screen
        # prefix      = 'gdb -ex=r --args',
        # prefix      = 'sleep 3;',
        parameters  =
        [
            {"log_dir"         : LaunchConfiguration('log_dir')}, 
            # Gtr traj params
            
            # SO3xR3 trajectory
            {"wqx1"            : 3*0.05},
            {"wqy1"            : 3*0.05},
            {"wqz1"            : 1*0.05},
            {"rqx1"            : M_PI*0.5},
            {"rqy1"            : M_PI*0.5},
            {"rqz1"            : M_PI*sqrt(3)/2},
            
            {"wpx1"            : 3*0.45},
            {"wpy1"            : 3*0.45},
            {"wpz1"            : 1*0.45},
            {"rpx1"            : 5.0},
            {"rpy1"            : 5.0},
            {"rpz1"            : 5.0},

            # SE3 trajectory
            {"wpx2"            : 3*0.15},
            {"wpy2"            : 3*0.15},
            {"wpz2"            : 1*0.15},
            {"rpx2"            : 5.0},
            {"rpy2"            : 5.0},
            {"rpz2"            : 5.0},
            
            
            # GN MAP optimization params
            {"maxTime"         : 20.0},
            {'deltaT'          : LaunchConfiguration('deltaT')},
            {'mpCovROSJerk'    : 10.0},
            {'mpCovPVAJerk'    : 10.0},
            {"pose_type"       : LaunchConfiguration('pose_type')}, # Choose 'SE3' or 'SO3xR3'
            {"use_approx_drv"  : LaunchConfiguration('use_approx_drv')},
            {"lie_epsilon"     : 1e-2},
            {"max_ceres_iter"  : 50},
            {"random_start"    : 0},
            
            
            # UWB param config
            {"lidar_rate"      : 200.0},
            {"lidar_noise"     : 0.05},
            
            # UWB param config
            {"Dtstep"          : [0.1, 0.3, 0.05]},
            {"Wstep"           : [1, 30]},

        ]  # Optional: pass parameters if needed
    )

    # Rviz node
    # rviz_node = Node(
    #     package     = 'rviz2',
    #     executable  = 'rviz2',
    #     name        = 'rviz2',
    #     output      = 'screen',
    #     arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_pl.rviz']
    # )

    on_exit_action = RegisterEventHandler(event_handler=OnProcessExit(
                                          target_action=gptr_pl_node,
                                          on_exit=[Shutdown()])
                                         )
    
    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([log_dir, deltaT, pose_type, use_approx_drv,
                              gptr_pl_node,
                            #   rviz_node,
                              on_exit_action])