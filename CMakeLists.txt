CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
SET(CMAKE_CXX_STANDARD 11)

PROJECT(create_bdst_isl)

FIND_PACKAGE(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

FIND_PACKAGE(Qt5 COMPONENTS Core Sql)
IF(Qt5_FOUND)
INCLUDE_DIRECTORIES(${Qt5Core_INCLUDE_DIRS})
SET(QT_LIBS Qt5::Core Qt5::Sql)
ELSE()
FIND_PACKAGE(Qt4 REQUIRED QtCore QtSql)
INCLUDE(${QT_USE_FILE})
add_definitions( ${QT_DEFINITIONS} )
SET(QT_LIBS Qt4::QtCore Qt4::QtSql)
ENDIF()

FIND_PACKAGE( OpenCV REQUIRED )
SET(main_directory ~/catkin_ws/src/place_detection_isl/src)

INCLUDE_DIRECTORIES(~/catkin_ws/src/place_detection_isl/src)
INCLUDE_DIRECTORIES(~/catkin_ws/src/place_detection_isl/src/bubble)

set(bubble_space_SOURCES
	${main_directory}/bubble/bubbleprocess.cpp
 	${main_directory}/database/databasemanager.cpp
	${main_directory}/imageprocess/imageprocess.cpp
  	${main_directory}/Utility/Place.cpp
  	${main_directory}/Utility/BasePoint.cpp
  	${main_directory}/Utility/PlaceDetector.cpp
  	${main_directory}/Utility/TemporalWindow.cpp
	src/cluster.c
)

SET(bubble_space_HEADERS
 	${main_directory}/bubble/bubbleprocess.h
 	${main_directory}/database/databasemanager.h
 	${main_directory}/imageprocess/imageprocess.h
  	${main_directory}/Utility/Place.h
  	${main_directory}/Utility/BasePoint.h
  	${main_directory}/Utility/PlaceDetector.h
  	${main_directory}/Utility/TemporalWindow.h
	src/cluster.h
)

CATKIN_PACKAGE()

INCLUDE_DIRECTORIES(
  ${catkin_INCLUDE_DIRS}
)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

ADD_LIBRARY(${PROJECT_NAME} src/main.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
 opencv_highgui
 opencv_imgproc
 opencv_core
 opencv_ml
 ${catkin_LIBRARIES})

ADD_EXECUTABLE(create_bdst_isl_node
src/main.cpp
${bubble_space_SOURCES}
 ${bubble_space_HEADERS}
)
TARGET_LINK_LIBRARIES(create_bdst_isl_node
 ${PROJECT_NAME}
 ${QT_LIBS})
