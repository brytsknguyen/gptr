# my_node_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

bag_file         = "/media/tmn/mySataSSD1/DATASETS/UTIL_DATASETS/dataset/flight-dataset/ros2bag-data/const1/const1-trial1-tdoa2"
anchor_path      = "/media/tmn/mySataSSD1/DATASETS/UTIL_DATASETS/dataset/flight-dataset/survey-results/anchor_const1_survey.txt"
result_save_path = "/home/tmn/Documents/results/gptr/c1/"

def generate_launch_description():
    
    # GPTR UI node
    gptr_ui_node = Node(
        package     = 'gptr',
        executable  = 'gptr_ui',  # Name of the executable built by your package
        name        = 'gptr_ui',  # Optional: gives the node instance a name
        # prefix      = ['gdb -ex run --args'],
        output      = 'screen',   # Print the node output to the screen
        parameters  =
        [
            {"auto_exit"        : 1},
            {"if_save_traj"     : 1},
            {"traj_save_path"   : result_save_path},

            # Parameters for the Gaussian Process
            {"gpDt"             : 0.04357},
            {"gpQr"             : 1.00},
            {"gpQc"             : 1.00},
            {"pose_type"        : "SE3"}, # Choose 'SE3' or 'SO3xR3'
            {"lie_epsilon"      : 1e-2},
            {"use_closed_form"  : 0},

            # UWB anchor position
            {"anchor_pose_path" : anchor_path},
            {"tdoa_topic"       : '/tdoa_data'},
            {"tof_topic"        : '/tof_data'},
            {"imu_topic"        : '/imu_data'},
            {"gt_topic"         : '/pose_data'},
            {"fuse_tdoa"        : 1},
            {"fuse_tof"         : 1},
            {"fuse_imu"         : 1},

            # Parameters for the solver
            {"SLIDE_SIZE"       : 2},      # How many knots to slide
            {"WINDOW_SIZE"      : 20},     # How many knot length does the sliding window
            {"w_tdoa"           : 100.0},  # Coefficients for TDOA residuals
            {"GYR_N"            : 2000.0}, # Coefficients for IMU residuals
            {"GYR_W"            : 100.0},  # Coefficients for IMU residuals
            {"ACC_N"            : 100.0},  # Coefficients for IMU residuals
            {"ACC_W"            : 100.0},  # Coefficients for IMU residuals
            {"tdoa_loss_thres"  : 30.0},   # Loss function for IMU residuals
            {"mp_loss_thres"    : 100.0},  # Loss function for motion prior residuals
        ]  # Optional: pass parameters if needed
    )

    # ros2 bag 
    bag_node = ExecuteProcess( cmd=["ros2", "bag", "play", bag_file, "-r", "0.5"], output="screen", )     

    # Rviz node
    rviz_node = Node(
        package     = 'rviz2',
        executable  = 'rviz2',
        name        = 'rviz2',
        output      = 'screen',
        arguments   = ['-d', get_package_share_directory('gptr') + '/launch/gptr_ui.rviz']
    )

    # launch_arg = DeclareLaunchArgument('cartinbot_viz', required=True, description='Testing')

    return LaunchDescription([gptr_ui_node, rviz_node, bag_node])