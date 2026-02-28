cmake_minimum_required(VERSION 3.13)

project(gmm_d2d_registration
	VERSION 0.0.0
	DESCRIPTION
	"GMM Registration"
	LANGUAGES CXX C)

find_package(dlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(self_organizing_gmm REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Intel MKL paths -- disabled: MKL hurts performance for small matrices (3x3 GMM covariances)
# set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
# set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
# set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
# include_directories(${MKL_INCLUDE_DIR})
# link_directories(${MKL_LIB_DIR})

set(CMAKE_CXX_FLAGS "-O3 -march=native -ffast-math -fno-math-errno -mfma -fopenmp -funroll-loops")
add_compile_definitions(NDEBUG EIGEN_NO_DEBUG)
# add_compile_definitions(EIGEN_USE_MKL_ALL MKL_LP64)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Link OpenMP for all targets in this project (MKL libs commented out)
# link_libraries(OpenMP::OpenMP_CXX gomp pthread m dl mkl_rt)
link_libraries(OpenMP::OpenMP_CXX gomp pthread m dl)

find_package(Eigen3 REQUIRED)

include_directories(include
	${dlib_INCLUDE_DIR}
	${sogmm_INCLUDE_DIR}
)

install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION "include/${PROJECT_NAME}"
)

install(DIRECTORY cmake/
	DESTINATION "share/${PROJECT_NAME}"
	FILES_MATCHING PATTERN "*.cmake"
)

add_executable(test_partial_wrt_u test/test_partial_wrt_u.cpp)
target_link_libraries(test_partial_wrt_u Eigen3::Eigen)
