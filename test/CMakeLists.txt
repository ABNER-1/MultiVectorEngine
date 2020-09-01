

add_executable(test_ip testIP.cpp)

target_link_libraries(test_ip
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_nra test_nra.cpp)

target_link_libraries(test_nra
        multi_vector
        milvus_sdk
        pthread)
