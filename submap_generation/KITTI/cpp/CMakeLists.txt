cmake_minimum_required(VERSION 3.5.0)
project(pointcloud_preprocess)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_BUILD_TYPE Release)#Debug
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} -Wall -O3 -march=native")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -pg -march=native")

set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pg -march=native")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

#find_package(OpenCV 3.1.0 REQUIRED)
#find_package(Eigen3 REQUIRED)
#find_package(G2O REQUIRED)
#find_package(CSparse REQUIRED)
#find_package(OpenNI2 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Boost  REQUIRED)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
#        ${OPENCV_INCLUDE_DIRS}
#        ${OPENNI2_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(segment_submap segment_submap.cpp)

target_link_libraries(segment_submap
        #        ${OpenCV_LIBS}
        #        ${OpenNI2_LIBRARIES}
        ${PCL_LIBRARIES}
        )


add_executable(segment_submap_xyzi segment_submap_xyzi.cpp)

target_link_libraries(segment_submap_xyzi
        #        ${OpenCV_LIBS}
        #        ${OpenNI2_LIBRARIES}
        ${PCL_LIBRARIES}
        )

add_executable(segment_submap_xyzl segment_submap_xyzl.cpp)

target_link_libraries(segment_submap_xyzl
        #        ${OpenCV_LIBS}
        #        ${OpenNI2_LIBRARIES}
        ${PCL_LIBRARIES}
        )