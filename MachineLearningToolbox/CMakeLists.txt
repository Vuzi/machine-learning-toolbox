cmake_minimum_required(VERSION 3.5)
project(MachineLearningToolbox)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(SOURCE_FILES src/plugin.cpp include/plugin.h)
set(UNITY_PLUGIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../MachineLearningUnity/Assets/Plugins)

#add_executable(MachineLearningToolbox ${SOURCE_FILES})
add_library(MachineLearningToolbox SHARED ${SOURCE_FILES})

add_custom_command(TARGET MachineLearningToolbox
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:MachineLearningToolbox> ${CMAKE_CURRENT_SOURCE_DIR}/../MachineLearningUnity/Assets/Plugins)
