

add_executable(test_ip test_ip.cpp utils.cpp)

target_link_libraries(test_ip
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_nra test_nra.cpp)

target_link_libraries(test_nra
        multi_vector
        milvus_sdk
        pthread)


add_executable(test_l2 test_l2.cpp utils.cpp)

target_link_libraries(test_l2
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_nra2 test_nra2.cpp)

target_link_libraries(test_nra2
        multi_vector
        milvus_sdk
        pthread)

