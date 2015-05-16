cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(unified)

find_package(PCL 1.7.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (unified unified.cpp)
target_link_libraries (unified ${PCL_LIBRARIES})
