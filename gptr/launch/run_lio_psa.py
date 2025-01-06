# my_node_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Sequence
sequence = 'psa_sample'

# Bag file
lidar_bag_file = f'/media/tmn/mySataSSD1/Experiments/gptr/{sequence}'

# Initial pose in each sequence
xyzypr_W_L0 =[ 0,    0,   0.70,  43,  48, 0,
              -0.3, -0.3, 0.55, -134, 0,  0 ]

# Direction to log the exp
log_dir = f'/media/tmn/mySataSSD1/Experiments/gptr/logs/lio/psa_exp/psa_{sequence}_gptr/'

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
            {"kf_min_dis"     : 0.5},
            {"kf_min_angle"   : 10},
            
            # Location of the prior map
            {"priormap_file"  : "none"},
            
            # Location of bag file
            {"lidar_bag_file" : lidar_bag_file},
            
            # Total number of clouds loaded
            {'MAX_CLOUDS'      : -1},

            # Time since first pointcloud to skip MAP Opt
            {'SKIPPED_TIME'    : 0.0},
            {'RECURRENT_SKIP'  : 0},

            # Set to '1' to skip optimization
            {'VIZ_ONLY'        : 0},

            # Lidar topics and their settings
            {'lidar_topic'     : ['/ouster/points']},
            {'lidar_type'      : ['ouster']},
            {'stamp_time'      : ['start' ]},

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
            {'deltaT'          : 0.02204},

            {'lidar_ds_rate'   : 1},
            {'lidar_weight'    : 10.0},

            # Motion prior factors
            {'mpSigGa'         : 1.0},
            {'mpSigNu'         : 1.0},

            # Extrinsic factors
            {'xtSigGa'         : 200.0},
            {'xtSigNu'         : 200.0},

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
            {'SW_CLOUDNUM'     : 4},
            {'SW_CLOUDSTEP'    : 1},
            {'max_lidarcoefs'  : 8000},
            {'XTRZ_DENSITY'    : 1},
            {'min_planarity'   : 0.5},
            {'max_plane_dis'   : 0.5},
            {'knnsize'         : 6},
            {'max_ceres_iter'  : 50},
            {'max_outer_iter'  : 1},
            {'max_inner_iter'  : 40},
            {'min_inner_iter'  : 3},
            {'conv_thres'      : 3},
            {'dJ_conv_thres'   : 10.0},
            {'conv_dX_thres'   : [-0.05, -0.5, -1.0, -0.05, -0.5, -1.0 ]},
            {'change_thres'    : [-15.0, -0.5, -1.0, -15.0, -8.0, -2.0 ]},

            {'fix_time_begin'  :  0.001},
            {'fix_time_end'    : -0.025},
            {'fuse_marg'       : 1},
            {'compute_cost'    : 0},

            # Log dir
            {'log_period'      : 5.0},
            {'log_dir'         : log_dir},
        ]  # Optional: pass parameters if needed
    )
    
    # # Visualization node for gptr
    # cartinbot_viz = Node(
    #     package     = 'gptr',
    #     executable  = 'cartinbot_viz.py',  # Name of the executable built by your package
    #     name        = 'cartinbot_viz',     # Optional: gives the node instance a name
    #     output      = 'screen',            # Print the node output to the screen
    #     parameters  = []
    # )

    # Rviz node
    rviz_node = Node(
        package     = 'rviz2',
        executable  = 'rviz2',
        name        = 'rviz2',
        output      = 'screen',
        arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_lo_sim.rviz']
    )

    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([gptr_lo_node, rviz_node])