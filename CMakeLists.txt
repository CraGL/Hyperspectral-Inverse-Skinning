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

FIND_LIBRARY(COCOA_LIBRARY Cocoa)
if(NOT APPLE)
	set(COCOA_LIBRARY "")
endif()

## OpenGL/GLFW
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3>=3.1)
# find_package(GLFW REQUIRED glfw3>=3.1)
include_directories(${GLFW_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})
if(NOT OPENGL_FOUND)
	message(Couldn't find OpenGL Libraries.)
endif()

## GLEW on some platforms for advanced OpenGL stuff
## It's installed on our linux machine: ldconfig -p | grep -i glew
# find_package(GLEW)
# if (GLEW_FOUND)
#    include_directories(${GLEW_INCLUDE_DIRS})
    ## Only link it for certain targets (see GUI_LIBS below)
    # link_libraries(${GLEW_LIBRARIES})
# endif()
## find_package() didn't work, but we only have a linker error so let's do it this way.
find_library(GLEW_LIBRARIES GLEW glew)

## AntTweakBar
find_package(ANTTWEAKBAR REQUIRED)
include_directories( ${ANT_TWEAK_BAR_INCLUDE_DIR} )

## OpenImageIO
find_package(OpenImageIO REQUIRED)
include_directories( "${OpenImageIO_INCLUDE_DIR}" )

## PNGLIB
find_package(PNG REQUIRED)

## We need Zlib
find_package(ZLIB REQUIRED)
include_directories( "${ZLIB_INCLUDE_DIRS}" )

link_directories(
	/usr/local/lib
	/opt/local/lib
	${EIGEN_DIRS}
)

set(NON_GUI_LIBS
	${LIBIGL_LIBRARIES}
	${MOSEK_LIBRARIES}
	)
	
set(GUI_LIBS 
	${COCOA_LIBRARY}
	${LIBIGL_LIBRARIES}
	${OPENGL_LIBRARIES} 
	${GLEW_LIBRARIES}
	${GLFW_LIBRARIES} 
	${ANT_TWEAK_BAR_LIBRARY}
	${PNG_LIBRARIES}
	${OpenImageIO_LIBRARY}
	) 

## We need C++11. Put this directive after CGAL's include.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g " )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  add_compile_options(-stdlib=libc++)
endif()

## A program for rendering static scene
add_executable(estimate_weights
	estimate_weights.cpp
	)
target_link_libraries(estimate_weights
	${NON_GUI_LIBS}
)

include_directories( "." )
## A simple viewer
add_executable(viewer
	viewer.cpp
	jsoncpp.cpp
	viewer/save_screenshot.cpp
	viewer/stb_image_write.cpp
	viewer/shaderHelper.cpp
	viewer/blend_scene.cpp
	viewer/vertex_array_object.cpp
	viewer/flattener.cpp
	viewer/controls.cpp
	viewer/draw_handles.cpp
	)
target_link_libraries(viewer
	${GUI_LIBS}
)

## Another simple viewer
add_executable(viewer2
	viewer2.cpp
	jsoncpp.cpp
	viewer/save_screenshot.cpp
	viewer/stb_image_write.cpp
	viewer/shaderHelper.cpp
	viewer/blend_scene.cpp
	viewer/vertex_array_object.cpp
	viewer/flattener.cpp
	viewer/controls.cpp
	viewer/draw_handles.cpp
	)
target_link_libraries(viewer2
	${GUI_LIBS}
)

## Weight manipulator
add_executable(weight_manipulator
	weight_manipulator.cpp
	)
target_link_libraries(weight_manipulator
	${LIBIGL_LIBRARIES}
)