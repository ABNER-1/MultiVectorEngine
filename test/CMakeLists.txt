

add_executable(test_ip test_ip.cpp utils.cpp)

target_link_libraries(test_ip
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_ipnra_baseline test_ipnra_baseline.cpp utils.cpp)
target_link_libraries(test_ipnra_baseline
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_nra test_nra.cpp)
target_link_libraries(test_nra
        multi_vector
        milvus_sdk
        pthread)

add_executable(ip_nra_performance ip_nra_performance.cpp)

target_link_libraries(ip_nra_performance
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_l2 test_l2.cpp utils.cpp)

target_link_libraries(test_l2
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_gist_l2 test_gist_l2.cpp utils.cpp)

target_link_libraries(test_gist_l2
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_nra2 test_nra2.cpp)

target_link_libraries(test_nra2
        multi_vector
        milvus_sdk
        pthread)

add_executable(test_nra3 test_nra3.cpp)

target_link_libraries(test_nra3
        multi_vector
        milvus_sdk
        pthread)

add_executable(brute_force brute_force.cpp utils.cpp)

target_link_libraries(brute_force
        multi_vector
        milvus_sdk
        gomp
        pthread)
