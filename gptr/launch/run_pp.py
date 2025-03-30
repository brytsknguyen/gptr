# my_node_launch.py
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, LogInfo, EmitEvent, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from math import pi as M_PI
from math import sqrt as sqrt

# # Sequence
# sequence_ = 'cathhs_07'

# # Bag file
# lidar_bag_file_ = f'/media/tmn/mySataSSD1/Experiments/gptr/{sequence_}'

# # Direction to log the exp
# log_dir_ = f'/media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lio/sim_exp/cathhs_{sequence_}_gptr_two_lidar/'

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

def generate_launch_description():
    
    # lidar_bag_file  = DeclareLaunchArgument('lidar_bag_file', default_value=lidar_bag_file_, description='')   # Bag file
    # log_dir         = DeclareLaunchArgument('log_dir', default_value=log_dir_, description='')                 # Direction to log the exp
    pose_type       = DeclareLaunchArgument('pose_type', default_value=pose_type_, description='')             # Variant of kinematics
    use_approx_drv  = DeclareLaunchArgument('use_approx_drv', default_value=use_approx_drv_, description='')   # Variant of approximation
    deltaT          = DeclareLaunchArgument('deltaT', default_value=deltaT_, description='')                   # Variant of approximation

    # GPTR LO node
    gptr_lo_node = Node(
        package     = 'gptr',
        executable  = 'gptr_pp',  # Name of the executable built by your package
        name        = 'gptr_pp',  # Optional: gives the node instance a name
        output      = 'screen',   # Print the node output to the screen
        # prefix      = 'gdb -ex=r --args',
        # prefix      = 'sleep 3;',
        parameters  =
        [
            # Gtr traj params
            {"wq"              : 5.0},
            {"wp"              : 0.15},
            {"rq1"             : M_PI*0.5},
            {"rq2"             : M_PI*sqrt(3)*0.5},
            {"rp"              : 5.0},
            
            
            # GN MAP optimization params
            {"maxTime"         : 69.0/3},
            {'deltaT'          : LaunchConfiguration('deltaT')},
            {'mpCovROSJerk'    : 1.0},
            {'mpCovPVAJerk'    : 1.0},
            {"pose_type"       : LaunchConfiguration('pose_type')}, # Choose 'SE3' or 'SO3xR3'
            {"use_approx_drv"  : LaunchConfiguration('use_approx_drv')},
            {"lie_epsilon"     : 1e-2},
            {"max_ceres_iter"  : 50},
            
            
            # UWB param config
            {"uwb_rate"        : 50.0},

        ]  # Optional: pass parameters if needed
    )

    # Rviz node
    rviz_node = Node(
        package     = 'rviz2',
        executable  = 'rviz2',
        name        = 'rviz2',
        output      = 'screen',
        arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_pp.rviz']
    )

    on_exit_action = RegisterEventHandler(event_handler=OnProcessExit(
                                          target_action=gptr_lo_node,
                                          on_exit=[Shutdown()])
                                         )
    
    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([deltaT, pose_type, use_approx_drv,
                              gptr_lo_node, rviz_node,
                              on_exit_action])