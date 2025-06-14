# A Gaussian Process Trajectory Representation with Closed-Form Kinematics for Continuous-Time Motion Estimation
## Preresiquite

### ROS 1 User
* Install Ubuntu 20.04 and ROS NOETIC
* Please install depencies, including Ceres Solver (1.14) and sophus
  ```
  sudo apt install libfmt-dev # may be required as a dependency of sophus
  sudo apt install ros-noetic-sophus
  sudo apt install libceres-dev
  ```
* Create a catkin workspace with [SFUISE](https://github.com/ASIG-X/SFUISE) (main branch) for the `cf_msgs` dependency, and [GPTR](https://github.com/brytsknguyen/gptr) (master branch)
  ```
  mkdir -p ~/gptr_ws/src && cd ~/gptr_ws
  catkin init
  cd src
  git clone git@github.com:ASIG-X/SFUISE.git
  git clone git@github.com:brytsknguyen/gptr.git
  ```
* Build the repo:
  ```
  cd ~/gptr_ws
  catkin build gptr
  ```

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

### Ceres 2.2

Ceres 2.2 replaces LocalParameterization class with Manifold. We have also did some tests on ceres.2.2 on Ubuntu 22.04 and ROS HUMBLE. 
* Install ceres from source.
* Install sophus from source. Do `git checkout 1.24.6`. If there is a complaint about cmake version, you can manual change the cmake version in CMakeLists.txt file, for example `cmake_minimum_required(VERSION 3.22)` instead of 3.24.
* Make sure `libfmt-dev` is installed.
* Checkout the ceres.2.2 branch of this repo and colcon build the repo.

Please raise an issue should you encounter any issue with the compilation of the package.

## Testing the lidar pipeline:

### With synthetic data

You can download and unzip the file `cloud_avia_mid_dynamic_extrinsics` from [here](https://drive.google.com/file/d/1Q5fTn5OvWd_I2RvVfiUKir90q5HshzQM/view?usp=sharing). It contains the pointclouds and the prior map for the experiment.

Launch the experiment with `run_lio_sim.launch`, ensuring that you set the path to the extracted folder correctly.
```
source ~/gptr_ws/devel/setup.bash
roslaunch gptr run_lio_sim.launch path:=/path/to/cloud_avia_mid_dynamic_extrinsics
```

You should see the following visualization in rviz.

<img src="docs/sim.gif" alt="synthetic_exp" width="600"/>

### With handheld setup

Similar to the synthetic dataset, please download the data and the prior map from [here](https://drive.google.com/file/d/1QId8X4LFxYdYewHSBXiDEAvpIFD8w-ei/view?usp=sharing).

Launch the experiment with `run_lio_cathhs_iot.launch`, ensuring that you set the path to the extracted folder correctly.
```
source ~/gptr_ws/devel/setup.bash
roslaunch gptr run_lio_cathhs_iot.launch path:=/path/to/cathhs_07
```

You should see the following visualization in rviz.

<img src="docs/cathhs.gif" alt="cathhs_exp" width="600"/>

### Evaluation

Please use the scripts `analysis_cathhs.ipynb` and `analysis_sim.ipynb` to evaluate the result.

<br/>

## Testing on UWB-inertial fusion

Please download the [UTIL](https://utiasdsl.github.io/util-uwb-dataset/) (TDoA-inertial) dataset.

For ROS1 users, please run with the correct path to the directory containing the extracted dataset:
```
roslaunch gptr run_util.launch path:=/path/to/UTIL
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

Run the following command from terminal, changing the path as desired:
```
roslaunch gptr run_vicalib.launch result_save_path:=/path/to/results
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
