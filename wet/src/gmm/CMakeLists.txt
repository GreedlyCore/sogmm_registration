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

include_directories(include
  ${EIGEN3_INCLUDE_DIR}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION "include/${PROJECT_NAME}"
)

install(DIRECTORY cmake/
  DESTINATION "share/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.cmake"
)
