#!bin/bash

# Function to run exps
runexp()
{
    sequence=$1
    logroot=$2
    deltaT=$3

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
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # Kinematic type and use/unuse of closed form
    pose_type=SO3xR3
    use_approx_drv=1
    # Run the experiement
    for n in {1..1}; do
        # Directory to log the exp
        log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

        # Run the exp
        ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    done

    # # Kinematic type and use/unuse of closed form
    # pose_type=SE3
    # use_approx_drv=0
    # # Run the experiement
    # for n in {1..1}; do
    #     # Directory to log the exp
    #     log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

    #     # Run the exp
    #     ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    # done

    # # Kinematic type and use/unuse of closed form
    # pose_type=SE3
    # use_approx_drv=1
    # # Run the experiement
    # for n in {1..1}; do
    #     # Directory to log the exp
    #     log_dir=${logroot}/${sequence}/exp_${pose_type}_${use_approx_drv}/gptr_two_lidar/try_$n

    #     # Run the exp
    #     ros2 launch gptr run_liosim.py lidar_bag_file:=$lidar_bag_file log_dir:=$log_dir deltaT:=$deltaT pose_type:=$pose_type use_approx_drv:=$use_approx_drv
    # done
}


# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_050/lio/ 0.05

# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_100/lio/ 0.10

# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_150/lio/ 0.15

# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_200/lio/ 0.20

# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_250/lio/ 0.25

# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_300/lio/ 0.30

# runexp cloud_avia_mid_w75_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w65_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w55_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w50_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w45_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w35_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w25_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40
# runexp cloud_avia_mid_w15_e5 /media/tmn/mySataSSD1/Experiments/gptr_v2/logs_16032025/Dt_400/lio/ 0.40




