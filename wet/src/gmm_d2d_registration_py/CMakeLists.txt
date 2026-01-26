cmake_minimum_required(VERSION 3.5...3.28)
project(gmm_d2d_registration_py)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# Enable optimizations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP found and enabled for gmm_d2d_registration_py")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(gmm_d2d_registration REQUIRED)
find_package(gmm REQUIRED)
find_package(dlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  ${gmm_d2d_registration_INCLUDE_DIR}
  ${gmm_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${dlib_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
)

# nanobind modules
nanobind_add_module(${PROJECT_NAME}
  src/${PROJECT_NAME}/${PROJECT_NAME}.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  ${PYTHON_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  dlib::dlib
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION .)
