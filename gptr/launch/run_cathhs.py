# my_node_launch.py
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, LogInfo, EmitEvent, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Sequence
sequence_ = 'cathhs_07'

# Bag file
lidar_bag_file_ = f'/media/tmn/mySataSSD1/Experiments/gptr/{sequence_}'

# Direction to log the exp
log_dir_ = f'/media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lio/cathhs_exp/cathhs_{sequence_}_gptr_two_lidar/'

# Type of pose
# pose_type_ = 'SE3'
pose_type_ = 'SO3xR3'

# Type of pose
use_approx_drv_ = '0'

# Knot length
deltaT_ = '0.02204'

# Initial pose in each sequence
xyzypr_W_L0 =[ 0,   0,  0,   0, 0,  0,
               0.2, 0, -0.2, 0, 90, 0 ]

def generate_launch_description():
    
    launcharg_lidar_bag_file  = DeclareLaunchArgument('lidar_bag_file', default_value=lidar_bag_file_, description='')   # Bag file
    launcharg_log_dir         = DeclareLaunchArgument('log_dir', default_value=log_dir_, description='')                 # Direction to log the exp
    launcharg_pose_type       = DeclareLaunchArgument('pose_type', default_value=pose_type_, description='')             # Variant of kinematics
    launcharg_use_approx_drv  = DeclareLaunchArgument('use_approx_drv', default_value=use_approx_drv_, description='')   # Variant of approximation
    launcharg_deltaT          = DeclareLaunchArgument('deltaT', default_value=deltaT_, description='')                   # Variant of approximation

    lidar_bag_file  = LaunchConfiguration('lidar_bag_file')
    log_dir         = LaunchConfiguration('log_dir')       
    pose_type       = LaunchConfiguration('pose_type')     
    use_approx_drv  = LaunchConfiguration('use_approx_drv')
    deltaT          = LaunchConfiguration('deltaT')        

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
            {"priormap_file"   : "/media/tmn/mySataSSD1/Experiments/gptr/cathhs_iot_prior_2cm.pcd"},
            
            # Location of bag file
            {"lidar_bag_file"  : lidar_bag_file},
            
            # Total number of clouds loaded
            {'MAX_CLOUDS'      : -1},

            # Time since first pointcloud to skip MAP Opt
            {'SKIPPED_TIME'    : 2.0},
            {'RECURRENT_SKIP'  : 0},

            # Set to '1' to skip optimization
            {'VIZ_ONLY'        : 0},

            # Lidar topics and their settings
            {'lidar_topic'     : ['/livox/lidar_0/points', '/livox/lidar_1/points']},
            {'lidar_type'      : ['ouster', 'ouster']},
            {'stamp_time'      : ['start', 'start']},

            # Imu topics and their settings
            # {imu_topic       : ['/os_cloud_node/imu']},

            # Run kf for comparison
            {'runkf'           : 1},

            # Set to 1 to load the search for previous logs and redo
            {'resume_from_log' : [0, 0]},

            # Initial pose of each lidars
            {'xyzypr_W_L0'     : xyzypr_W_L0},

            # Groundtruth for evaluation
            {'xtrz_gndtr'      : [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]},
            
            {'T_E_G'           : [ -0.00037886633842519943, -0.011824967458688653, -0.6445255757936823, 177.41697100832081, -0.8380643201341046, -0.4324815049625096,
                                    0.018040371221108277,   -0.005781886649251375, -0.6659410499572929, 177.8095334058065,  -0.6988913250237251, -0.4278818562418 ]},

            # Leaf size to downsample priormap
            {'pmap_leaf_size'  : 0.1},
            {'cloud_ds'        : [0.2, 0.2]},

            # GN MAP optimization params
            {'deltaT'          : deltaT},
            # Motion prior factors
            {'mpCovROSJerk'    : 10.0},
            {'mpCovPVAJerk'    : 10.0},
            {"pose_type"       : pose_type}, # Choose 'SE3' or 'SO3xR3'
            {"use_approx_drv"  : use_approx_drv},
            {"lie_epsilon"     : 1.0e-2},

            {'lidar_ds_rate'   : 1},
            {'lidar_weight'    : 10.0},

            # Extrinsic factors
            {'xtCovROSJerk'    : 1.0},
            {'xtCovPVAJerk'    : 1.0},

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
            {'SW_CLOUDNUM'     : PythonExpression(['int(0.6/', LaunchConfiguration('deltaT'), ' + 0.5)'])},
            {'SW_CLOUDSTEP'    : 1},
            {'max_lidarcoefs'  : 2000},
            {'XTRZ_DENSITY'    : 1},
            {'min_planarity'   : 0.5},
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
            {'fix_time_begin'  : -0.0125},
            {'fix_time_end'    : -0.0},
            {'fuse_marg'       : 1},
            {'compute_cost'    : 0},
            {'lambda'          : 1.0},
            {'dXM'             : 0.02},

            # Log dir
            {'log_period'      : 5.0},
            {'log_dir'         : log_dir},
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
        arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_cathhs.rviz']
    )

    on_exit_action = RegisterEventHandler(event_handler=OnProcessExit(
                                          target_action=gptr_lo_node,
                                          on_exit=[Shutdown()])
                                         )
    
    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([launcharg_lidar_bag_file,
                              launcharg_log_dir,
                              launcharg_deltaT,
                              launcharg_pose_type,
                              launcharg_use_approx_drv,
                              gptr_lo_node, cartinbot_viz, rviz_node,
                              on_exit_action])