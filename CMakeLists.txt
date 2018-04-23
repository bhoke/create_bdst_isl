CMAKE_MINIMUM_REQUIRED(VERSION 2.4.6)

PROJECT(create_bdst_isl)
# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#SET(ROS_BUILD_TYPE RelWithDebInfo)

FIND_PACKAGE(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)


SET(CMAKE_AUTOMOC ON)

FIND_PACKAGE(Qt5Widgets)
FIND_PACKAGE(Qt5Sql)
FIND_PACKAGE(Qt5Core)
FIND_PACKAGE(OpenCV REQUIRED)

SET(main_directory ~/catkin_ws/src/place_detection_isl/src)

INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS} ${Qt5Core_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(~/catkin_ws/src/place_detection_isl/src)
INCLUDE_DIRECTORIES(~/catkin_ws/src/place_detection_isl/src/bubble)

SET(bubble_space_SOURCES
  ${main_directory}/bubble/bubbleprocess.cpp
  ${main_directory}/database/databasemanager.cpp
  #src/bdst.cpp
)
SET(bubble_space_NSOURCES
  ${main_directory}/imageprocess/imageprocess.cpp
  ${main_directory}/Utility.cpp
  src/cluster.c
)
SET(bubble_space_HEADERS
  ${main_directory}/bubble/bubbleprocess.h
  ${main_directory}/database/databasemanager.h
  #src/bdst.h
)
SET(bubble_space_NHEADERS
  ${main_directory}/imageprocess/imageprocess.h
  ${main_directory}/Utility.h
  src/cluster.h
)

CATKIN_PACKAGE()

INCLUDE_DIRECTORIES(${catkin_INCLUDE_DIRS})

#INCLUDE(${QT_USE_FILE})
#ADD_DEFINITIONS(${QT_DEFINITIONS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

ADD_LIBRARY(${PROJECT_NAME} src/main.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
  opencv_highgui
  opencv_imgproc
  opencv_core
  opencv_ml
  ${catkin_LIBRARIES}
)
ADD_EXECUTABLE(create_bdst_isl_node
 src/main.cpp
 ${bubble_space_SOURCES}
 ${bubble_space_NSOURCES}
 ${bubble_space_NHEADERS}
 ${bubble_space_HEADERS_MOC}
)
TARGET_LINK_LIBRARIES(create_bdst_isl_node ${PROJECT_NAME} Qt5::Core Qt5::Widgets Qt5::Sql)
