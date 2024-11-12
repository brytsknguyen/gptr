cmake_minimum_required(VERSION 3.0.2)
project(gptr)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  rosbag
  pcl_conversions
  pcl_ros
  cf_msgs
  nlink_parser
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenMP REQUIRED)
# find_package(Sophus REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gptr
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  include/ikdTree
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  # ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/gptr.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collidelibraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} fmt)

# Visual-inertial calibration
add_executable(${PROJECT_NAME}_vicalib src/GPVICalib.cpp)
add_dependencies(${PROJECT_NAME}_vicalib ${catkin_EXPORTED_TARGETS})
target_compile_options(${PROJECT_NAME}_vicalib PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_vicalib ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

# UWB fusion
add_executable(${PROJECT_NAME}_nl src/GPNL.cpp)
add_dependencies(${PROJECT_NAME}_nl ${catkin_EXPORTED_TARGETS})
target_compile_options(${PROJECT_NAME}_nl PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_nl ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

# UWB fusion
add_executable(${PROJECT_NAME}_ui src/GPUI.cpp)
add_dependencies(${PROJECT_NAME}_ui ${catkin_EXPORTED_TARGETS})
target_compile_options(${PROJECT_NAME}_ui PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_ui ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(${PROJECT_NAME}_ui_test src/testUI.cpp)
add_dependencies(${PROJECT_NAME}_ui_test ${catkin_EXPORTED_TARGETS})
target_compile_options(${PROJECT_NAME}_ui_test PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_ui_test ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

# Lidar fusion
add_executable(${PROJECT_NAME}_lo src/GPLO.cpp include/ikdTree/ikd_Tree.cpp)
add_dependencies(${PROJECT_NAME}_lo ${catkin_EXPORTED_TARGETS})
target_compile_options(${PROJECT_NAME}_lo PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_lo ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(${PROJECT_NAME}_lo_test src/testLIO.cpp include/ikdTree/ikd_Tree.cpp)
add_dependencies(${PROJECT_NAME}_lo_test ${catkin_EXPORTED_TARGETS})
target_compile_options(${PROJECT_NAME}_lo_test PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_lo_test ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS})

# A vizualization node
catkin_install_python(PROGRAMS
  scripts/cartinbot_viz.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})