#!bin/bash

# Function to run exps
runexp()
{
    sequence=$1
    logroot=$2

    # Bag file
    lidar_bag_file=/media/tmn/mySataSSD1/Experiments/gptr_v2/sequences/${sequence}

    # Copy the config
    mkdir -p ${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar
    cp -r /home/tmn/ros2_ws/src/gptr/gptr/launch ${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=0
    # Run the experiement
    for n in {1..5}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=1
    # Run the experiement
    for n in {1..5}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_approx_drv=0
    # Run the experiement
    for n in {1..5}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_approx_drv=1
    # Run the experiement
    for n in {1..5}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done
}

runexp cloud_avia_mid_w25 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_27022025/lio/
runexp cloud_avia_mid_w35 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_27022025/lio/
runexp cloud_avia_mid_w45 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_27022025/lio/
runexp cloud_avia_mid_w55 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_27022025/lio/
# runexp cloud_avia_mid_w65 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_27022025/lio/
# runexp cloud_avia_mid_w75 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_27022025/lio/
# runexp cloud_avia_mid_w85 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_26022025/lio/
# runexp cloud_avia_mid_dynamic_extrinsics /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_25022025/lio/
