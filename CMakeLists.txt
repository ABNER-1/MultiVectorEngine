cmake_minimum_required(VERSION 3.13)
project(MultiVector)

set(CMAKE_CXX_STANDARD 17)


set(SDK_HOME ${PROJECT_SOURCE_DIR}/sdk)
set(SDK_INCLUDE ${SDK_HOME}/include)
message("sdk home: ${SDK_HOME}")

include_directories(${SDK_INCLUDE})
include_directories(include)
link_directories(${SDK_HOME})

add_subdirectory(src)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(MultiVector main.cpp)

target_link_libraries(MultiVector
        multi_vector
        milvus_sdk
        pthread)

add_subdirectory(test)
