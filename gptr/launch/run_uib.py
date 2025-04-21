# my_node_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

bag_file_         = "/home/kailai/Documents/100_dataset/util_ros2/const4/const4-trial5-tdoa2-traj2/"
anchor_path_      = "/home/kailai/Documents/100_dataset/util_dataset/flight-dataset/survey-results/anchor_const4_survey.txt"
result_save_path_ = "/home/kailai/Documents/results/cmp4/"

def generate_launch_description():

    launcharg_bag_file  = DeclareLaunchArgument('bag_file', default_value=bag_file_, description='')
    launcharg_anchor_path  = DeclareLaunchArgument('anchor_path', default_value=anchor_path_, description='')
    launcharg_result_save_path  = DeclareLaunchArgument('result_save_path', default_value=result_save_path_, description='')

    bag_file  = LaunchConfiguration('bag_file')
    anchor_path  = LaunchConfiguration('anchor_path')
    result_save_path  = LaunchConfiguration('result_save_path')   
    
    # GPTR UI node
    gptr_ui_node = Node(
        package     = 'gptr',
        executable  = 'gptr_uib',  # Name of the executable built by your package
        name        = 'gptr_uib',  # Optional: gives the node instance a name
        # prefix      = ['gdb -ex run --args'],
        output      = 'screen',   # Print the node output to the screen
        parameters  =
        [
            {"auto_exit"        : 1},
            {"if_save_traj"     : 1},
            {"traj_save_path"   : result_save_path},
            {"bag_file"         : bag_file},

            # Parameters for the Gaussian Process
            {"gpQr"             : 1.00},
            {"gpQc"             : 1.00},
            {"lie_epsilon"      : 1e-2},

            # UWB anchor position
            {"anchor_pose_path" : anchor_path},

            # Parameters for the solver
            {"w_tdoa"           : 100.0},  # Coefficients for TDOA residuals
            {"GYR_N"            : 2000.0}, # Coefficients for IMU residuals
            {"ACC_N"            : 100.0},  # Coefficients for IMU residuals

            # Time skewing factor
            {"tskew0"            : 1.0},
            {"tskewmax"          : 2.2},
            {"tskewstep"         : 0.3},
            {"Dtstep"            : [1.0, 0.8, 0.5, 0.3, 0.1, 0.05]},        
        ]  # Optional: pass parameters if needed
    )

    return LaunchDescription([launcharg_bag_file, 
                              launcharg_anchor_path, 
                              launcharg_result_save_path, 
                              gptr_ui_node])
