cmake_minimum_required(VERSION 3.16)

# Name of the project
include(CheckLanguage)

PROJECT(CUDAPOINTCOLLISIONS LANGUAGES CXX CUDA)

# CUDA
# https://stackoverflow.com/questions/13167598/error-cuda-runtime-h-no-such-file-or-directory/75559127#75559127
# find_package(CUDA REQUIRED)
enable_language(CUDA)
# add_definitions(-DUSE_CUDA)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Use glob to get the list of all source files.
# We don't really need to include header and resource files to build, but it's
# nice to have them also show up in IDEs.
IF(${SOL})
    FILE(GLOB_RECURSE CPP_SOURCES "src0/*.cpp")
    FILE(GLOB_RECURSE CUDA_SOURCES "src0/*.cu")
    FILE(GLOB_RECURSE HEADERS "src0/*.h")
    FILE(GLOB_RECURSE GLSL "resources0/*.glsl")
ELSE()
    FILE(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
    FILE(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
    FILE(GLOB_RECURSE HEADERS "src/*.h")
    FILE(GLOB_RECURSE GLSL "resources/*.glsl")
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G --generate-line-info --expt-relaxed-constexpr")

# Set the executable.
ADD_EXECUTABLE(${CMAKE_PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES} ${HEADERS} ${GLSL})

# Get the GLM environment variable. Since GLM is a header-only library, we
# just need to add it to the include directory.
SET(GLM_INCLUDE_DIR "$ENV{GLM_INCLUDE_DIR}")
IF(NOT GLM_INCLUDE_DIR)
    # The environment variable was not set
    SET(ERR_MSG "Please point the environment variable GLM_INCLUDE_DIR to the root directory of your GLM installation.")
    IF(WIN32)
        # On Windows, try the default location
        MESSAGE(STATUS "Looking for GLM in ${DEF_DIR_GLM}")
        IF(IS_DIRECTORY ${DEF_DIR_GLM})
            MESSAGE(STATUS "Found!")
            SET(GLM_INCLUDE_DIR ${DEF_DIR_GLM})
        ELSE()
            MESSAGE(FATAL_ERROR ${ERR_MSG})
        ENDIF()
    ELSE()
        MESSAGE(FATAL_ERROR ${ERR_MSG})
    ENDIF()
ENDIF()
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})

# Use c++17
SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# OS specific options and libraries
IF(WIN32)
    # -Wall produces way too many warnings.
    # -pedantic is not supported.
    # Disable warning 4996.
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 -g")
    TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} opengl32.lib)
    SET_PROPERTY(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})
ELSE()
    # Enable all pedantic warnings.
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
    IF(APPLE)
        # Add required frameworks for GLFW.
        # TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} "-framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo")
    ELSE()
        #Link the Linux OpenGL library
        TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} "GL")
    ENDIF()
ENDIF()

