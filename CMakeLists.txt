cmake_minimum_required(VERSION 2.8.6)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

## We need Eigen
find_package(EIGEN REQUIRED)
include_directories( "${EIGEN_INCLUDE_DIR}" )

## We need libigl
find_package(LIBIGL REQUIRED)
include_directories( "${LIBIGL_INCLUDE_DIR}" )

## We have/want MOSEK
find_package(MOSEK REQUIRED)
include_directories( "${MOSEK_INCLUDE_DIR}" )

## Tetgen
find_package(TETGEN REQUIRED)
include_directories( "${TETGEN_INCLUDE_DIR}" )

link_directories(
	/usr/local/lib
	/opt/local/lib
	${EIGEN_DIRS}
	${TETGEN_DIRS}
)

set(LIBS 
	${COCOA_LIBRARY}
	${LIBIGL_LIBRARIES}
#	${MOSEK_LIBRARIES}
	) 

## We need C++11. Put this directive after CGAL's include.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g " )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  add_compile_options(-stdlib=libc++)
endif()

## A program for rendering static scene
add_executable(estimate_weights
	general_barycentric_coordinates.cpp
	)
target_link_libraries(estimate_weights
	${LIBS}
)
