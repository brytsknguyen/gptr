# A Gaussian Process Trajectory Representation with Closed-Form Kinematics for Continuous-Time Motion Estimation

### ROS 2 User
* Install Ubuntu 22.04 and ROS HUMBLE
* Please use colcon build [SFUISE2](https://github.com/ASIG-X/SFUISE2) in your workspace to have the cf_msg.
* Install sophus and ceres 2.0
  ```
  sudo apt install libfmt-dev # may be required as a dependency of sophus
  sudo apt install ros-humble-sophus
  sudo apt install libceres-dev
  ```
* Git clone and colcon build the repo.

## Testing the lidar pipeline:

### With synthetic data

You can download and unzip the file `cloud_avia_mid_dynamic_extrinsics` from [here](https://drive.google.com/file/d/1Q5fTn5OvWd_I2RvVfiUKir90q5HshzQM/view?usp=sharing). It contains the pointclouds and the prior map for the experiment.

After that, modify the path to the data and prior map in `gptr/launch/run_liosim.py` and launch it. You should see the following visualization from rviz.

<img src="docs/sim.gif" alt="synthetic_exp" width="600"/>

### With handheld setup

Similar to the synthetic dataset, please download the data and the prior map from [here](https://drive.google.com/file/d/1QId8X4LFxYdYewHSBXiDEAvpIFD8w-ei/view?usp=sharing).

Then specify the paths to the data and prior map in `gptr/launch/run_lio_cathhs.launch` before roslaunch. You should see the following illustration.

<img src="docs/cathhs.gif" alt="cathhs_exp" width="600"/>

<br/>

## UWB Exp

Please launch `gptr/launch/run_pp.py`

## Testing on visual-inertial batch optimization
<!-- <img src="docs/vicalib.gif" width="600"/> -->

Please launch `gptr/launch/run_vicalib.py`


## Importing GPTR in your work:

The heart of our toolkit is the [GaussianProcess.hpp](gptr/include/GaussianProcess.hpp) header file which contains the abstraction of the continuous-time trajectory represented by a third-order `GaussianProcess`.

The `GaussianProcess` object provides methods to create, initialize, extend, and query information from the trajectory.

The toolkit contains three main examples:

* Visual-Inertial Calibration: a batch optimization problem where visual-inertial factors are combined to estimate the trajectory and extrinsics of a camera-imu pair, encapsulated in the `GPVICalib.cpp` file.
* UWB-Inertial Localization: a sliding-window Maximum A Posteriori (MAP) optimization problem featuring TDOA UWB measurements and IMU, presented in the `GPUI.cpp` file.
* Multi-lidar Coupled-Motion Estimation: a sliding-window MAP optimization problem with lidar-only observation, featuring multiple trajectories with extrinsic factors providing a connection between these trajectories, implemented in the `GPLO.cpp` trajectory.
