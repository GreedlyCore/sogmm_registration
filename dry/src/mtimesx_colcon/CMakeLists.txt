cmake_minimum_required(VERSION 2.8)
project(mtimesx_colcon)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(MATLAB QUIET)

if (MATLAB_FOUND)

  find_package(BLAS REQUIRED)
  if (BLAS_FOUND)
    message(STATUS "BLAS FOUND")
  else()
    message(FATAL_ERROR "BLAS NOT FOUND")
   endif()

   if (UNIX AND NOT APPLE)
     set(MEX_EXT "mexa64")
   endif()

   if (APPLE)
     set(MEX_EXT "mexmaci64")
   endif()

   if (WIN32)
     message(FATAL_ERROR "WINDOWS UNTESTED")
   endif()

   set(COPY_FROM ${CMAKE_CURRENT_BINARY_DIR}/mtimesx_src-prefix/src/mtimesx_src/src/mtimesx.${MEX_EXT})
   set(COPY_TO ${CMAKE_INSTALL_PREFIX}/mtimesx.${MEX_EXT})

   include(ExternalProject)
   externalproject_add(mtimesx_src
     GIT_REPOSITORY git@github.com:gira3d/mtimesx.git
     GIT_TAG master
     CONFIGURE_COMMAND ""
     BUILD_COMMAND cd ../mtimesx_src/src && mex -DDEFINEUNIX -largeArrayDims mtimesx.c ${BLAS_LIBRARIES}
     INSTALL_COMMAND cp ${COPY_FROM} ${COPY_TO}
    )

    install(DIRECTORY cmake/
      DESTINATION "share/${PROJECT_NAME}"
      FILES_MATCHING PATTERN "*.cmake"
    )

    if (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/matlab")
      file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/matlab")
    endif()

    install(DIRECTORY "${CMAKE_BINARY_DIR}/mtimesx_src-prefix/src/mtimesx_src/matlab/"
      DESTINATION "matlab"
      FILES_MATCHING PATTERN "*.m"
    )
endif()
