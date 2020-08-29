

add_executable(test_ip testIP.cpp)

target_link_libraries(test_ip
        multi_vector
        milvus_sdk
        pthread)