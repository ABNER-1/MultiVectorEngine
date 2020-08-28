cmake_minimum_required(VERSION 3.13)
project(MultiVector)

set(CMAKE_CXX_STANDARD 17)


set(SDK_HOME ${PROJECT_SOURCE_DIR}/sdk)
set(SDK_INCLUDE ${SDK_HOME}/include)
message("sdk home: ${SDK_HOME}")

include_directories(${SDK_INCLUDE})
include_directories(include)
add_subdirectory(src)
link_directories(${SDK_HOME})


include_directories(${PROJECT_SOURCE_DIR}/include)

set(multi_vector_srcs
        src/MultiVectorEngine.cpp
        src/MultiVectorCollectionIP.cpp
        src/MultiVectorCollectionL2.cpp
        src/Utils.cpp
        )


add_executable(MultiVector main.cpp
        ${multi_vector_srcs})

target_link_libraries(MultiVector
        milvus_sdk
        pthread)