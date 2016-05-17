cmake_minimum_required(VERSION 3.5)
project(MachineLearningToolbox)

# C++11 and force 64bit
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(SOURCE_FILES src/plugin.cpp include/plugin.h include/perceptron.h src/perceptron.cpp include/multiLayerPerceptron.h
        src/multiLayerPerceptron.cpp include/utils.h src/utils.cpp src/test.cpp)

# Link to Eigen3
SET(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "No environment variable EIGEN3_INCLUDE_DIR found")
endif()

# Unity project should be here
set(UNITY_PLUGIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../MachineLearningUnity/Assets/Plugins)

# Includes
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("include")

# Lib
add_executable(MachineLearningToolboxTest ${SOURCE_FILES})
add_library(MachineLearningToolbox SHARED ${SOURCE_FILES})

# Copy to the unity directory
add_custom_command(TARGET MachineLearningToolbox
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:MachineLearningToolbox> ${UNITY_PLUGIN_PATH})

# On OSX .so should be renammed to .bundle
if(APPLE)
    set_target_properties(MachineLearningToolbox PROPERTIES SUFFIX ".bundle")
endif()
