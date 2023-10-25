cmake_minimum_required(VERSION 3.16)
project(MySFM)
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (PCL REQUIRED)
find_package(cvsba REQUIRED)
link_directories(${PCL_LIBRARY_DIRS} ${cvsba_LIB_DIR})
add_definitions(${PCL_DEFINITIONS})

include_directories( ${PCL_INCLUDE_DIRS} ${cvsba_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)
file(GLOB_RECURSE SOURCE_FILES 
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable( ${PROJECT_NAME} ${SOURCE_FILES} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${cvsba_LIBS} ${PCL_LIBRARIES} -lstdc++fs)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17 )