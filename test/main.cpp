#include <vector>
#include <algorithm>
#include <iostream>
#include "utils.h"

int
main() {
    std::string file_name = "/home/abner/vector/glove-200-angular.hdf5";
    std::vector<std::vector<float>> vector_data;
//    unsigned dim, num;
    std::vector<milvus::multivector::RowEntity> row_entity;
    readArraysFromHdf5(file_name, {64, 64, 72}, row_entity,
                       10000, 2, "train");
    std::cout << "endl" << std::endl;
}