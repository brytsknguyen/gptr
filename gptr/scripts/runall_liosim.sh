#!bin/bash

# Function to run exps
runexp()
{
    sequence=$1
    logroot=$2

    # Bag file
    lidar_bag_file=/media/tmn/mySataSSD1/Experiments/gptr/${sequence}

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_closed_form=0
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_closed_form}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_closed_form:=$use_closed_form
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_closed_form=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_closed_form}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_closed_form:=$use_closed_form
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_closed_form=0
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_closed_form}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_closed_form:=$use_closed_form
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_closed_form=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_closed_form}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir pose_type:=$pose_type use_closed_form:=$use_closed_form
    done
}

runexp cloud_avia_mid_w75 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_23022025/lio/
# runexp cloud_avia_mid_w50
# runexp cloud_avia_mid_w25
# runexp cloud_avia_mid_dynamic_extrinsics
