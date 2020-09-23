cmake_minimum_required(VERSION 3.13)
project(MultiVector)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(${CMAKE_CXX_FLAGS})
endif()

set(SDK_HOME ${PROJECT_SOURCE_DIR}/sdk)
set(SDK_INCLUDE ${SDK_HOME}/include)
message("sdk home: ${SDK_HOME}")

include_directories(${SDK_INCLUDE})
#include_directories("/usr/include/hdf5/serial")
include_directories(include)
link_directories(${SDK_HOME})
#link_directories("/usr/lib/x86_64-linux-gnu")

add_subdirectory(src)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(MultiVector main.cpp)

target_link_libraries(MultiVector
        multi_vector
        milvus_sdk
        pthread)

add_subdirectory(test)
