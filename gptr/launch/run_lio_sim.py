# my_node_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Sequence
sequence = 'cloud_avia_mid_dynamic_extrinsics'

# Bag file
lidar_bag_file = f'/media/tmn/mySataSSD1/Experiments/gptr/{sequence}'

# Initial pose in each sequence
xyzypr_W_L0 =[ 0,    0,   0.70,  43,  48, 0,
              -0.3, -0.3, 0.55, -134, 0,  0 ]

# Direction to log the exp
log_dir = f'/media/tmn/mySataSSD1/Experiments/gptr/logs/lio/sim_exp/sim_{sequence}_gptr_two_lidar/'

def generate_launch_description():
    
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
            {"priormap_file"  : "/media/tmn/mySataSSD1/Experiments/gptr/sim_priormap.pcd"},
            
            # Location of bag file
            {"lidar_bag_file" : lidar_bag_file},
            
            # Total number of clouds loaded
            {'MAX_CLOUDS'      : 300},

            # Time since first pointcloud to skip MAP Opt
            {'SKIPPED_TIME'    : 4.9},
            {'RECURRENT_SKIP'  : 0},

            # Set to '1' to skip optimization
            {'VIZ_ONLY'        : 0},

            # Lidar topics and their settings
            {'lidar_topic'     : ['/lidar_0/points']},
            {'lidar_type'      : ['livox', 'livox']},
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
            {'xtrz_gndtr'      : [ 0, 0, 0, 0, 0, 0,
                                  -0.1767767, 0, -0.53033009, 180, 45, 0 ]},

            # Leaf size to downsample priormap
            {'pmap_leaf_size'  : 0.15},
            {'cloud_ds'        : [0.1, 0.1]},

            # GN MAP optimization params
            {'deltaT'          : 0.05743},
            # Motion prior factors
            {'mpCovROSJerk'    : 1.0},
            {'mpCovPVAJerk'    : 1.0},
            {"pose_type"       : "SE3"}, # Choose 'SE3' or 'SO3xR3'
            {"lie_epsilon"     : 1e-2},
            {"use_closed_form" : 1},

            {'lidar_ds_rate'   : 1},
            {'lidar_weight'    : 10.0},

            # Extrinsic factors
            {'xtCovROSJerk'         : 200.0},
            {'xtCovPVAJerk'         : 200.0},

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
            {'SW_CLOUDNUM'     : 10},
            {'SW_CLOUDSTEP'    : 1},
            {'max_lidarcoefs'  : 4000},
            {'XTRZ_DENSITY'    : 1},
            {'min_planarity'   : 0.5},
            {'max_plane_dis'   : 0.5},
            {'knnsize'         : 6},
            
            {'use_ceres'       : 1},
            {'max_ceres_iter'  : 50},
            {'max_outer_iter'  : 1},
            {'max_inner_iter'  : 40},
            {'min_inner_iter'  : 3},
            {'conv_thres'      : 3},
            {'dJ_conv_thres'   : 10.0},
            {'conv_dX_thres'   : [-0.05, -0.5, -1.0, -0.05, -0.5, -1.0 ]},
            {'change_thres'    : [-15.0, -0.5, -1.0, -15.0, -8.0, -2.0 ]},
            {'fix_time_begin'  : 0.025},
            {'fix_time_end'    : 0.0},
            {'fuse_marg'       : 1},
            {'compute_cost'    : 1},
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
        arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_lo_sim.rviz']
    )

    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([gptr_lo_node, cartinbot_viz, rviz_node])