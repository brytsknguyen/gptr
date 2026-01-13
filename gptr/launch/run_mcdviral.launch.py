# my_node_launch.py
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, LogInfo, EmitEvent, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Sequence
sequence_ = 'ntu_day_01'

# Bag file
lidar_bag_file_ = f'/home/tmn/DATASETS/gptr/mcd_{sequence_}_lidars/'

# Direction to log the exp
log_dir_ = f'/media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lio/mcd_exp/mcd_{sequence_}_gptr_two_lidar/'

# Type of pose
# pose_type_ = 'SE3'
pose_type_ = 'SO3xR3'

# Type of pose
use_approx_drv_ = '0'

# Knot length
deltaT_ = '0.02204'

# Initial pose in each sequence
xyzypr_W_L0 = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]

if 'ntu_day_01' in sequence_:
    xyzypr_W_L0 = [ 49.27, 107.37, 7.62, -41,  0, 0,
                    49.27, 107.37, 7.62, -41,  0, 0 ]
elif 'ntu_day_02' in sequence_:
    xyzypr_W_L0 = [ 62.03, 119.61, 7.69, -134, 3, 1,
                    62.03, 119.57, 7.69, -134, 3, 1 ]
elif 'ntu_day_10' in sequence_:
    xyzypr_W_L0 = [ 39.49, 23.48,  6.54,  36,  3, 1,
                    39.49, 23.48,  6.54,  36,  3, 1 ]

skipped_time = 0.0
if 'ntu_day_01' in sequence_:
    skipped_time = 0.0
elif 'ntu_day_02' in sequence_:
    skipped_time = 14.5


def generate_launch_description():
    
    lidar_bag_file  = DeclareLaunchArgument('lidar_bag_file', default_value=lidar_bag_file_, description='')   # Bag file
    log_dir         = DeclareLaunchArgument('log_dir', default_value=log_dir_, description='')                 # Direction to log the exp
    pose_type       = DeclareLaunchArgument('pose_type', default_value=pose_type_, description='')             # Variant of kinematics
    use_approx_drv  = DeclareLaunchArgument('use_approx_drv', default_value=use_approx_drv_, description='')   # Variant of approximation
    deltaT          = DeclareLaunchArgument('deltaT', default_value=deltaT_, description='')                   # Variant of approximation

    # GPTR LO node
    gptr_lo_node = Node(
        package     = 'gptr',
        executable  = 'gptr_lo',  # Name of the executable built by your package
        name        = 'gptr_lo',  # Optional: gives the node instance a name
        output      = 'screen',   # Print the node output to the screen
        # prefix      = 'gdb -ex=r --args',
        # prefix      = 'sleep 3;',
        parameters  =
        [
            # Location of the prior map
            {"priormap_file"   : "/media/tmn/mySataSSD1/Experiments/gptr/mcd_ntu_priormap_downsampled.pcd"},
            
            # Location of bag file
            {"lidar_bag_file"  : LaunchConfiguration('lidar_bag_file')},
            
            # Total number of clouds loaded
            {'MAX_CLOUDS'      : -1},

            # Time since first pointcloud to skip MAP Opt
            {'SKIPPED_TIME'    : skipped_time},
            {'RECURRENT_SKIP'  : 0},

            # Set to '1' to skip optimization
            {'VIZ_ONLY'        : 0},

            # Lidar topics and their settings
            {'lidar_topic'     : ['/os_cloud_node/points', '/livox/lidar']},
            {'lidar_type'      : ['ouster', 'livox']},
            {'stamp_time'      : ['end', 'start']},

            # Imu topics and their settings
            # {imu_topic       : ['/os_cloud_node/imu']},

            # Run kf for comparison
            {'runkf'           : 0},

            # Set to 1 to load the search for previous logs and redo
            {'resume_from_log' : [0, 0]},

            # Initial pose of each lidars
            {'xyzypr_W_L0'     : xyzypr_W_L0},

            # Groundtruth for evaluation
            {'xtrz_gndtr'      : [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]},

            # Leaf size to downsample priormap
            {'pmap_leaf_size'  : 0.15},
            {'cloud_ds'        : [1.6, 0.2]},

            # GN MAP optimization params
            {'deltaT'          : LaunchConfiguration('deltaT')},
            # Motion prior factors
            {'mpCovROSJerk'    : 1.0},
            {'mpCovPVAJerk'    : 1.0},
            {"pose_type"       : LaunchConfiguration('pose_type')}, # Choose 'SE3' or 'SO3xR3'
            {"use_approx_drv"  : LaunchConfiguration('use_approx_drv')},
            {"lie_epsilon"     : 5e-2},

            {'lidar_ds_rate'   : 8},
            {'lidar_weight'    : 50.0},

            # Extrinsic factors
            {'xtCovROSJerk'    : 0.2},
            {'xtCovPVAJerk'    : 0.5},

            # Loss function threshold
            {'ld_loss_thres'   : -1.0},
            {'pp_loss_thres'   : -1.0},
            {'mp_loss_thres'   : -10.0},
            {'xt_loss_thres'   : -10.0},

            {'max_omg'         : -20.0},
            {'max_alp'         : -40.0},
            {'max_vel'         : -10.0},
            {'max_acc'         : -5.0},

            # Extrinsic estimation
            {'SW_CLOUDNUM'     : PythonExpression(['int(0.8/', LaunchConfiguration('deltaT'), ' + 0.5)'])},
            {'SW_CLOUDSTEP'    : 1},
            {'max_lidarcoefs'  : 2000},
            {'XTRZ_DENSITY'    : 1},
            {'min_planarity'   : 0.75},
            {'max_plane_dis'   : 0.5},
            {'knnsize'         : 6},
            
            {'use_ceres'       : 1},
            {'max_ceres_iter'  : 50},
            {'max_outer_iter'  : 1},
            {'max_inner_iter'  : 2},
            {'min_inner_iter'  : 2},
            {'conv_thres'      : 2},
            {'dJ_conv_thres'   : 10.0},
            {'conv_dX_thres'   : [-0.05, -0.5, -1.0, -0.05, -0.5, -1.0 ]},
            {'change_thres'    : [-15.0, -0.5, -1.0, -15.0, -8.0, -2.0 ]},
            {'fix_time_begin'  : 0.025},
            {'fix_time_end'    : 0.0},
            {'fuse_marg'       : 1},
            {'compute_cost'    : 0},
            {'lambda'          : 1.0},
            {'dXM'             : 0.02},

            # Log dir
            {'log_period'      : 5.0},
            {'log_dir'         : LaunchConfiguration('log_dir')},
        ]  # Optional: pass parameters if needed
    )
    
    # Visualization node for gptr
    cartinbot_viz = Node(
        package     = 'gptr',
        executable  = 'cartinbot_viz.py',  # Name of the executable built by your package
        name        = 'cartinbot_viz',     # Optional: gives the node instance a name
        output      = 'screen',            # Print the node output to the screen
        parameters  = []
    )

    # Rviz node
    rviz_node = Node(
        package     = 'rviz2',
        executable  = 'rviz2',
        name        = 'rviz2',
        output      = 'screen',
        arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_mcdviral.rviz']
    )

    on_exit_action = RegisterEventHandler(event_handler=OnProcessExit(
                                          target_action=gptr_lo_node,
                                          on_exit=[Shutdown()])
                                         )
    
    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([lidar_bag_file, log_dir, deltaT, pose_type, use_approx_drv,
                              gptr_lo_node, cartinbot_viz, rviz_node,
                              on_exit_action])