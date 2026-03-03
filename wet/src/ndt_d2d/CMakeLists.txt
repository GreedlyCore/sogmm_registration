cmake_minimum_required(VERSION 3.13)

project(ndt_registration
	VERSION 1.0.30
	DESCRIPTION "3D NDT D2D/P2D point cloud registration"
	LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system)
find_package(ndt_map REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")

include_directories(
	include
	${PCL_INCLUDE_DIRS}
	${ndt_map_INCLUDE_DIR}
)
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

set(${PROJECT_NAME}_LIB_SRCS
	src/ndt_matcher_d2d.cpp
	src/ndt_matcher_d2d_2d.cpp
	src/ndt_matcher_d2d_feature.cpp
	src/ndt_matcher_p2d.cpp
)
add_library(${PROJECT_NAME} ${${PROJECT_NAME}_LIB_SRCS})
target_link_libraries(${PROJECT_NAME}
	Eigen3::Eigen
	${PCL_LIBRARIES}
	OpenMP::OpenMP_CXX
	${ndt_map_LIBRARIES}
)

add_executable(simple_test test/simple.cc)
target_link_libraries(simple_test ${PROJECT_NAME})

add_executable(d2d_deathmatch test/deathmatch.cc)
target_link_libraries(d2d_deathmatch ${PROJECT_NAME})

add_executable(p2d_deathmatch test/p2d_deathmatch.cc)
target_link_libraries(p2d_deathmatch ${PROJECT_NAME})

add_executable(check_reg test/check_reg.cc)
target_link_libraries(check_reg ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} simple_test d2d_deathmatch p2d_deathmatch check_reg
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)

install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION include/${PROJECT_NAME}
)
