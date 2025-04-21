cmake_minimum_required(VERSION 3.5)
project(splineui)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cf_msgs REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
)

add_executable(SplineUI src/SplineUI.cpp)
target_include_directories(SplineUI PRIVATE /opt/ros/humble/include) 
ament_target_dependencies(SplineUI
  rclcpp
  Sophus
  tf2_ros
  rosbag2_cpp
  cf_msgs
  sensor_msgs 
  std_msgs 
  tf2_msgs
  nav_msgs
  geometry_msgs
)
target_link_libraries(SplineUI ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt::fmt)
install(TARGETS
  SplineUI
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
ament_package()

