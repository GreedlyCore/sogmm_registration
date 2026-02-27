cmake_minimum_required(VERSION 2.9)
project(open3d_colcon)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE specified, default to Release.")
    set(CMAKE_BUILD_TYPE "Release")
endif()

option(BUILD_SHARED_LIBS          "Build shared libraries"                   ON)
set(BUILD_SHARED_LIBS ON)

option(BUILD_PYTHON_MODULE        "Build the python module"                  ON)
set(BUILD_PYTHON_MODULE ON)

option(BUILD_EXAMPLES             "Build Open3D examples programs"           OFF)
set(BUILD_EXAMPLES OFF)

if (UNIX AND NOT APPLE)
option(BUILD_CUDA_MODULE          "Build the CUDA module"                    ON)
set(BUILD_CUDA_MODULE ON)
elseif(APPLE)
option(BUILD_CUDA_MODULE          "Build the CUDA module"                    OFF)
set(BUILD_CUDA_MODULE OFF)
else()
option(BUILD_CUDA_MODULE          "Build the CUDA module"                    OFF)
set(BUILD_CUDA_MODULE OFF)
endif()

option(BUILD_LIBREALSENSE         "Build support for Intel RealSense camera" OFF)
set(BUILD_LIBREALSENSE OFF)

option(ENABLE_CACHED_CUDA_MANAGER "Enable cached CUDA memory manager"        OFF)
set(ENABLE_CACHED_CUDA_MANAGER OFF)

if (NOT DEFINED OPEN3D_PARFOR_BLOCK)
   set(OPEN3D_PARFOR_BLOCK 128)
endif()
add_definitions(-DOPEN3D_PARFOR_BLOCK=${OPEN3D_PARFOR_BLOCK})

if (NOT DEFINED OPEN3D_PARFOR_THREAD)
   set(OPEN3D_PARFOR_THREAD 4)
endif()
add_definitions(-DOPEN3D_PARFOR_THREAD=${OPEN3D_PARFOR_THREAD})

add_subdirectory(Open3D)
