set(multi_vector_srcs
        MultiVectorEngine.cpp
        MultiVectorCollectionIP.cpp
        MultiVectorCollectionL2.cpp
        MultiVectorCollectionIPNra.cpp
        Utils.cpp)

add_library(multi_vector SHARED ${multi_vector_srcs})
set_target_properties(multi_vector PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(multi_vector
        milvus_sdk
        pthread)
