#!bin/bash

# Function to run exps
runexp()
{
    sequence=$1
    logroot=$2
    deltaT=$3
    method=$4

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
        ros2 launch gptr run_cathhs.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_cathhs.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_approx_drv=0
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_cathhs.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_approx_drv=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_cathhs.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done
}

runexp cathhs_07 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lio/cathhs_exp/gptr/ 0.02204
runexp cathhs_08 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lio/cathhs_exp/gptr/ 0.02204
runexp cathhs_09 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs/lio/cathhs_exp/gptr/ 0.02204
