cmake_minimum_required(VERSION 3.8)
project(gptr)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Ros packages
find_package(tf2_ros REQUIRED)

# Message packages
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(cf_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# ROS wrapped packges
find_package(pcl_conversions REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)

# Other packages
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# Add your custom include path here
include_directories(
  include  # Typically your package's own include directory
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
)

message("This is a warning message.")
message("PCL dir ${PCL_INCLUDE_DIRS}")
message("Eigen dir ${EIGEN3_INCLUDE_DIR}")
# message("Ceres dir ${CERES_INCLUDE_DIR}")

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()



# region Visual-Inertial Fusion ---------------------------------------------------------------------------------------

# VIcalib
add_executable(${PROJECT_NAME}_vicalib src/GPVICalib.cpp)
target_include_directories(${PROJECT_NAME}_vicalib PRIVATE /opt/ros/humble/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_vicalib
  rclcpp
  pcl_conversions
  Sophus
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
)
target_compile_options(${PROJECT_NAME}_vicalib PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_vicalib ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt::fmt)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_vicalib
  DESTINATION lib/${PROJECT_NAME}
)

# endregion Visual-Inertial Fusion ------------------------------------------------------------------------------------



# region UWB Inertial Fusion ------------------------------------------------------------------------------------------

# UI Fusion
add_executable(${PROJECT_NAME}_ui src/GPUI.cpp)
target_include_directories(${PROJECT_NAME}_ui PRIVATE /opt/ros/humble/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_ui
  rclcpp
  pcl_conversions
  Sophus
  tf2_ros
  cf_msgs
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
)
target_compile_options(${PROJECT_NAME}_ui PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_ui ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt::fmt)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_ui
  DESTINATION lib/${PROJECT_NAME}
)


# Test
add_executable(${PROJECT_NAME}_ui_test src/testUI.cpp)
target_include_directories(${PROJECT_NAME}_ui_test PRIVATE /opt/ros/humble/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_ui_test
  rclcpp
  pcl_conversions
  Sophus
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
)
target_compile_options(${PROJECT_NAME}_ui_test PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_ui_test ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt::fmt)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_ui_test
  DESTINATION lib/${PROJECT_NAME}
)

# endregion UWB Inertial Fusion ---------------------------------------------------------------------------------------



# region Lidar Fusion -------------------------------------------------------------------------------------------------

# Lidar Odometry
add_executable(${PROJECT_NAME}_lo src/GPLO.cpp include/ikdTree/ikd_Tree.cpp)
target_include_directories(${PROJECT_NAME}_lo PRIVATE /opt/ros/humble/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_lo
  rclcpp
  pcl_conversions
  Sophus
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
)
target_compile_options(${PROJECT_NAME}_lo PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_lo ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt::fmt)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_lo
  DESTINATION lib/${PROJECT_NAME}
)

# Test
add_executable(${PROJECT_NAME}_lo_test src/testLIO.cpp)
target_include_directories(${PROJECT_NAME}_lo_test PRIVATE /opt/ros/humble/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_lo_test
  rclcpp
  pcl_conversions
  Sophus
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
)
target_compile_options(${PROJECT_NAME}_lo_test PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_lo_test ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt::fmt)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_lo_test
  DESTINATION lib/${PROJECT_NAME}
)

# endregion Lidar Fusion ----------------------------------------------------------------------------------------------



# Link the launch folder between the shared and the code
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)


# Install Python executables
install(PROGRAMS
  scripts/cartinbot_viz.py
  DESTINATION lib/${PROJECT_NAME}
)

# Share the resource place
install(DIRECTORY
  scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
