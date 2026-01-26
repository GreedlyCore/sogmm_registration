cmake_minimum_required(VERSION 3.13)

project(self_organizing_gmm
	VERSION 0.0.0
	DESCRIPTION
	"PRI Mapping"
	LANGUAGES CXX C)

# configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-std=c++17 -stdlib=libc++ -march=native -O3 -ffast-math -fno-math-errno -mfma -funroll-loops -DNDEBUG -DEIGEN_NO_DEBUG")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp /usr/local/lib/libomp.dylib -I/usr/local/include")
  set(OpenMP_CXX_LIB_NAMES "libomp")
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp /usr/local/lib/libomp.dylib -I/usr/local/include")
  set(OpenMP_C_LIB_NAMES "libomp")
  set(OpenMP_libomp_LIBRARY "/usr/local/lib/libomp.dylib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
  # -Wno-enum-compare: suppress EigenRand + Eigen 3.4 enum mismatch warning in RandUtils.h
  set(CMAKE_CXX_FLAGS "-std=c++17 -march=native -O3 -fopenmp -Wno-enum-compare -ffast-math -fno-math-errno -mfma -funroll-loops -DNDEBUG -DEIGEN_NO_DEBUG")
endif()
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# find dependencies
find_package(OpenMP REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(nanoflann REQUIRED)

include(FetchContent)
FetchContent_Declare(
  EigenRand
  GIT_REPOSITORY https://github.com/bab2min/EigenRand.git
  GIT_TAG        v0.5.1
)
# Disable EigenRand tests and benchmarks
set(EIGENRAND_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(EIGENRAND_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(EigenRand)

# Setup nanobind modules
nanobind_add_module(time_profiler_py src/nanobind/time_profiler_py.cpp)
target_include_directories(time_profiler_py
  PRIVATE
  include
)
set_target_properties(time_profiler_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

nanobind_add_module(mean_shift_py src/nanobind/mean_shift_py.cpp)
target_include_directories(mean_shift_py PRIVATE include)
target_link_libraries(
  mean_shift_py
  PRIVATE Eigen3::Eigen
  PRIVATE nanoflann::nanoflann
  PRIVATE OpenMP::OpenMP_CXX
)
set_target_properties(mean_shift_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

nanobind_add_module(camera_model_py src/nanobind/camera_model_py.cpp)
target_include_directories(camera_model_py PRIVATE include)
target_link_libraries(
  camera_model_py
  PRIVATE Eigen3::Eigen
)
set_target_properties(camera_model_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

nanobind_add_module(gmm_py src/nanobind/gmm_py.cpp)
target_include_directories(gmm_py PRIVATE include ${EigenRand_SOURCE_DIR})
target_link_libraries(
  gmm_py
  PRIVATE Eigen3::Eigen
  PRIVATE OpenMP::OpenMP_CXX
)
set_target_properties(gmm_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

nanobind_add_module(kinit_py src/nanobind/kinit_py.cpp)
target_include_directories(kinit_py PRIVATE include)
target_link_libraries(
  kinit_py
  PRIVATE Eigen3::Eigen
  PRIVATE OpenMP::OpenMP_CXX
)
set_target_properties(kinit_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

nanobind_add_module(sogmm_cpu src/nanobind/sogmm_cpu.cpp)
target_include_directories(sogmm_cpu PRIVATE include ${EigenRand_SOURCE_DIR})
target_link_libraries(
  sogmm_cpu
  PRIVATE Eigen3::Eigen
  PRIVATE OpenMP::OpenMP_CXX
  PRIVATE nanoflann::nanoflann
)
set_target_properties(sogmm_cpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

# Install
install(TARGETS DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION "include/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY cmake/
  DESTINATION "share/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.cmake"
)
