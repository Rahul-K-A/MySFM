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

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND) 
	SET(REQUIRED_LIBRARIES  ${REQUIRED_LIBRARIES} -lgomp)
	add_definitions(-DUSE_OMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(LINKER_LIBS ${LINKER_LIBS} gomp) 
else()
endif()

# GCC Versions less than 8 dont have support for c++17
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
	message("GCC version is greater than 8. Using new filesystem includes!")
	set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
	add_compile_definitions(USE_NEW_FILESYSTEM_HEADER)
else()
	message("GCC version is less than 8. Using old filesystem includes!")
	target_link_libraries(${PROJECT_NAME} -lstdc++fs)
endif()

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${cvsba_LIBS} ${PCL_LIBRARIES} )

