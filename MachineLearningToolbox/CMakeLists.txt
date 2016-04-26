cmake_minimum_required(VERSION 3.5)
project(MachineLearningToolbox)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(SOURCE_FILES src/plugin.cpp include/plugin.h)
SET(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
IF(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "No environment variable EIGEN3_INCLUDE_DIR found")
ENDIF()
set(UNITY_PLUGIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../MachineLearningUnity/Assets/Plugins)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("include")

#add_executable(MachineLearningToolbox ${SOURCE_FILES})
add_library(MachineLearningToolbox SHARED ${SOURCE_FILES})

add_custom_command(TARGET MachineLearningToolbox
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:MachineLearningToolbox> ${UNITY_PLUGIN_PATH})

#set_target_properties(MachineLearningToolbox PROPERTIES SUFFIX ".bundle")
