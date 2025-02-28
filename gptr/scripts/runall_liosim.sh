#!bin/bash

# Function to run exps
runexp()
{
    sequence=$1
    logroot=$2

    # Bag file
    lidar_bag_file=/media/tmn/mySataSSD1/Experiments/gptr_v2/sequences/${sequence}

    # Copy the config
    mkdir -p ${logroot}/${sequence}/
    cp -r /home/tmn/ros2_ws/src/gptr/gptr/launch ${logroot}/${sequence}/

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=0
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # # Kinematic type and use/unuse of closed form
    # pose_type=SE3
    # use_approx_drv=0
    # # Run the experiement
    # for n in {1..1}; do
    #     # Directory to log the exp
    #     log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

    #     # Run the exp
    #     ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    # done

    # # Kinematic type and use/unuse of closed form
    # pose_type=SE3
    # use_approx_drv=1
    # # Run the experiement
    # for n in {1..1}; do
    #     # Directory to log the exp
    #     log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

    #     # Run the exp
    #     ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    # done
}

runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_37022025/lio/
runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_37022025/lio/
runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_37022025/lio/
runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_37022025/lio/
runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_37022025/lio/

# runexp cloud_avia_mid_w55_e8 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_32022025/lio/
# runexp cloud_avia_mid_w45_e8 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_32022025/lio/
# runexp cloud_avia_mid_w35_e8 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_32022025/lio/
# runexp cloud_avia_mid_w25_e8 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_32022025/lio/
# runexp cloud_avia_mid_w15_e8 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_32022025/lio/

