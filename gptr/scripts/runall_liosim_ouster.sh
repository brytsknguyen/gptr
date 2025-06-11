#!bin/bash

# Function to run exps
runexp()
{
    sequence=$1
    logroot=$2
    deltaT=$3

    # Bag file
    lidar_bag_file=/media/XXX/mySataSSD1/Experiments/gptr_v2/sequences/${sequence}

    # Copy the config
    mkdir -p ${logroot}/${sequence}/
    cp -r /home/XXX/ros2_ws/src/gptr/gptr/launch ${logroot}/${sequence}/

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=0
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim_ouster.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim_ouster.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_approx_drv=0
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim_ouster.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SE3
    use_approx_drv=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim_ouster.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done
}


# runexp cloud_mid_ouster_w75_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05
# runexp cloud_mid_ouster_w65_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05
# runexp cloud_mid_ouster_w55_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05
# # runexp cloud_mid_ouster_w45_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05
# # runexp cloud_mid_ouster_w35_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05
# # runexp cloud_mid_ouster_w25_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05
# # runexp cloud_mid_ouster_w15_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_050/lio/ 0.05

# runexp cloud_mid_ouster_w75_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10
# runexp cloud_mid_ouster_w65_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10
# runexp cloud_mid_ouster_w55_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10
# # runexp cloud_mid_ouster_w45_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10
# # runexp cloud_mid_ouster_w35_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10
# # runexp cloud_mid_ouster_w25_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10
# # runexp cloud_mid_ouster_w15_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_100/lio/ 0.10

# runexp cloud_mid_ouster_w75_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15
# runexp cloud_mid_ouster_w65_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15
# runexp cloud_mid_ouster_w55_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15
# # runexp cloud_mid_ouster_w45_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15
# # runexp cloud_mid_ouster_w35_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15
# # runexp cloud_mid_ouster_w25_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15
# # runexp cloud_mid_ouster_w15_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_150/lio/ 0.15

runexp cloud_mid_ouster_w75_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20
runexp cloud_mid_ouster_w65_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20
runexp cloud_mid_ouster_w55_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20
# runexp cloud_mid_ouster_w45_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20
# runexp cloud_mid_ouster_w35_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20
# runexp cloud_mid_ouster_w25_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20
# runexp cloud_mid_ouster_w15_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_200/lio/ 0.20

runexp cloud_mid_ouster_w75_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25
runexp cloud_mid_ouster_w65_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25
runexp cloud_mid_ouster_w55_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25
# runexp cloud_mid_ouster_w45_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25
# runexp cloud_mid_ouster_w35_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25
# runexp cloud_mid_ouster_w25_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25
# runexp cloud_mid_ouster_w15_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_250/lio/ 0.25

runexp cloud_mid_ouster_w75_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30
runexp cloud_mid_ouster_w65_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30
runexp cloud_mid_ouster_w55_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30
# runexp cloud_mid_ouster_w45_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30
# runexp cloud_mid_ouster_w35_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30
# runexp cloud_mid_ouster_w25_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30
# runexp cloud_mid_ouster_w15_e5 /media/XXX/mySataSSD1/Experiments/gptr_v2/logs_13032025_ouster/Dt_300/lio/ 0.30


