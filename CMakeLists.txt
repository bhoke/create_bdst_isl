cmake_minimum_required(VERSION 2.4.6)
#include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
project(create_bdst_isl)
# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

#set(ROS_BUILD_TYPE Release)

#rosbuild_init()
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)


find_package(Qt4 REQUIRED)
SET(QT_USE_QTSQL TRUE)

set(main_directory ~/catkin_ws/src/place_detection_isl/src)

include_directories(~/catkin_ws/src/place_detection_isl/src)
include_directories(~/catkin_ws/src/place_detection_isl/src/bubble)

set(bubble_space_SOURCES ${main_directory}/bubble/bubbleprocess.cpp ${main_directory}/database/databasemanager.cpp src/bdst.cpp )# src/pcprocessing.cpp)
set(bubble_space_NSOURCES ${main_directory}/imageprocess/imageprocess.cpp ${main_directory}/Utility.cpp src/cluster.c)
set(bubble_space_HEADERS ${main_directory}/bubble/bubbleprocess.h ${main_directory}/database/databasemanager.h src/bdst.h)# include/bubble_space/pcprocessing.h)
set(bubble_space_NHEADERS ${main_directory}/imageprocess/imageprocess.h ${main_directory}/Utility.h src/cluster.h)# include/bubble_space/pcprocessing.h)

QT4_WRAP_CPP(bubble_space_HEADERS_MOC ${bubble_space_HEADERS})
qt4_automoc(${bubble_space_SOURCES})

FIND_PACKAGE( OpenCV REQUIRED )

catkin_package(

)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

add_library(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} opencv_highgui opencv_imgproc opencv_core opencv_ml ${catkin_LIBRARIES})
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
add_executable(create_bdst_isl_node src/main.cpp ${bubble_space_SOURCES} ${bubble_space_NSOURCES} ${bubble_space_NHEADERS} ${bubble_space_HEADERS_MOC})
target_link_libraries(create_bdst_isl_node ${PROJECT_NAME} ${QT_LIBRARIES})
