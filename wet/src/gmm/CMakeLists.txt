cmake_minimum_required(VERSION 3.5...3.28)
project(gmm)

# Silence "unused variable" warning for CMAKE_EXPORT_COMPILE_COMMANDS passed by colcon
set(ignoreMe "${CMAKE_EXPORT_COMPILE_COMMANDS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Enable optimizations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP found and enabled for gmm")
else()
  message(WARNING "OpenMP not found - GMM will run single-threaded")
endif()

# Add optimization flags with BLAS support
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -fno-math-errno -mfma -funroll-loops")
add_compile_definitions(NDEBUG EIGEN_NO_DEBUG)

# Link to BLAS and LAPACK
link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} gomp pthread m dl)

find_package(Boost COMPONENTS filesystem)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)
# Force config mode to use dry/install Eigen3Config.cmake (skip local FindEigen3.cmake)
find_package(Eigen3 3.4 REQUIRED CONFIG)

include_directories(include
  ${EIGEN3_INCLUDE_DIR}
)

nanobind_add_module(gmm_classic_py src/nanobind/gmm_classic_py.cpp)
target_include_directories(gmm_classic_py PRIVATE include)
target_link_libraries(gmm_classic_py PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gmm_classic_py PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(gmm_classic_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION "include/${PROJECT_NAME}"
)

install(DIRECTORY cmake/
  DESTINATION "share/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.cmake"
)
