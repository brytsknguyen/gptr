# GPTR: Gaussian Process Trajectory Representation for Continuous-Time Motion Estimation

## Preresiquite

### ROS 1 User
* Install Ubuntu 20.04 and ROS NOETIC
* Checkout the master branch
* Please catkin build [SFUISE](https://github.com/ASIG-X/SFUISE) in your workspace to have the cf_msg, which is required in gptr.
* Please install Ceres 2.0 and sophus
  ```
  sudo apt install libfmt-dev # may be required as a dependency of sophus
  sudo apt install ros-noetic-sophus
  sudo apt install libceres-dev
  ```
* Git clone and catkin build the repo.

### ROS 2 User
* Install Ubuntu 22.04 and ROS HUMBLE
* Checkout ros2 branch
* Please colcon build [SFUISE2](https://github.com/ASIG-X/SFUISE2) in your workspace to have the cf_msg.
* Install sophus and ceres 2.0
  ```
  sudo apt install libfmt-dev # may be required as a dependency of sophus
  sudo apt install ros-humble-sophus
  sudo apt install libceres-dev
  ```
* Git clone and colcon build the repo.

Please raise an issue should you encounter any issue with the compilation of the package.

## Testing the lidar pipeline:

### With synthetic data

You can download and unzip the file `cloud_avia_mid_dynamic_extrinsics` from [here](https://drive.google.com/file/d/1Q5fTn5OvWd_I2RvVfiUKir90q5HshzQM/view?usp=sharing). It contains the pointclouds and the prior map for the experiment.

After that, modify the path to the data and prior map in `run_sim.launch` and launch it. You should see the following visualization from rviz.

<img src="docs/sim.gif" alt="synthetic_exp" width="600"/>

### With handheld setup

Similar to the synthetic dataset, please download the data and the prior map from [here](https://drive.google.com/file/d/1QId8X4LFxYdYewHSBXiDEAvpIFD8w-ei/view?usp=sharing).

Then specify the paths to the data and prior map in `gptr/launch/run_lio_cathhs_iot.launch` before roslaunch. You should see the following illustration.

<img src="docs/cathhs.gif" alt="cathhs_exp" width="600"/>

### Evaluation

Please use the scripts `analysis_cathhs.ipynb` and `analysis_sim.ipynb` to evaluate the result.

<br/>

## Testing on UWB-inertial fusion

Please download the [UTIL](https://utiasdsl.github.io/util-uwb-dataset/) (TDoA-inertial) dataset.

Change `bag_file` and `anchor_path` in `gptr/launch/run_util.launch` according to your own path.

For ROS1 users, please run
```
roslaunch gptr run_util.launch
```
For ROS2 users, please first convert the UTIL dataset to ROS2 bag using `ros2bag_convert_util.sh` from [SFUISE2](https://github.com/ASIG-X/SFUISE2) and run
```
ros2 launch gptr run_util.launch.py
```
Below is an exemplary run on sequence `const2-trial4-tdoa2`
<img src="/docs/ui_video.gif" width="600"/>

### Evaluation
Please set `if_save_traj` in `gptr/launch/run_util.launch` to `1` and change `result_save_path` accordingly. 

```
evo_ape tum /traj_save_path/gt.txt /traj_save_path/traj.txt -a --plot
```
For comparison, a baseline approach based on ESKF is available in the paper of UTIL dataset.

## Testing on visual-inertial estimation and calibration
<img src="/docs/vicalib.gif" width="600"/>
Run the following command from terminal

```
roslaunch gptr run_vicalib.launch
```
This dataset is converted from the original one in [here](https://gitlab.com/tum-vision/lie-spline-experiments).


## Importing GPTR in your work:

The heart of our toolkit is the [GaussianProcess.hpp](gptr/include/GaussianProcess.hpp) header file which contains the abstraction of the continuous-time trajectory represented by a third-order `GaussianProcess`.

The `GaussianProcess` object provides methods to create, initialize, extend, and query information from the trajectory.

The toolkit contains three main examples:

* Visual-Inertial Calibration: a batch optimization problem where visual-inertial factors are combined to estimate the trajectory and extrinsics of a camera-imu pair, encapsulated in the `GPVICalib.cpp` file.
* UWB-Inertial Localization: a sliding-window Maximum A Posteriori (MAP) optimization problem featuring TDOA UWB measurements and IMU, presented in the `GPUI.cpp` file.
* Multi-lidar Coupled-Motion Estimation: a sliding-window MAP optimization problem with lidar-only observation, featuring multiple trajectories with extrinsic factors providing a connection between these trajectories, implemented in the `GPLO.cpp` trajectory.

## Publication

For the theorectical foundation, please find our paper at [arxiv](https://arxiv.org/pdf/2410.22931)

If you use the source code of our work, please cite us as follows:

```
@article{nguyen2024gptr,
  title     = {GPTR: Gaussian Process Trajectory Representation for Continuous-Time Motion Estimation},
  author    = {Nguyen, Thien-Minh, and Cao, Ziyu, and Li, Kailai, and Yuan, Shenghai and Xie, Lihua},
  journal   = {arXiv preprint arXiv:2410.22931},
  year      = {2024}
}
```
