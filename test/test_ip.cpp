#include "MultiVectorEngine.h"
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include "utils.h"
using namespace milvus::multivector;


void
load_data(const std::string& filename,
          std::vector<std::vector<float>>& vector_data,
          unsigned& num, unsigned& dim) {
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    vector_data.resize(num, std::vector<float>(dim));

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        auto data = vector_data[i].data();
        in.seekg(4, std::ios::cur);
        in.read((char*)(data), dim * 4);
    }

    for (auto i = 0; i < num; ++i) {
        std::cout << "i: " << i << "    : ";
        for (auto j = 0; j < dim; ++j) {
            std::cout << vector_data[i][j];
            if (i != dim - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
    in.close();
}

void
split_data(std::vector<std::vector<float>>& raw_data,
           std::vector<std::vector<milvus::Entity>>& splited_data,
           std::vector<int64_t>& dims) {
    // vertify dims
    auto total_dims = std::accumulate(dims.begin(), dims.end(), 0ll);
    if (total_dims > raw_data.size()) {
        std::cerr << "input total dims lager than raw data dims";
    }
    splited_data.resize(raw_data.size());
    for (int i = 0; i < raw_data.size(); ++i) {
        int idx = 0;
        for (int j = 0; j < dims.size(); ++j) {
            for (int k = 0; k < dims[j]; ++k) {
                splited_data[i][j].float_data[k] = raw_data[i][idx + k];
            }
            idx += dims[j];
        }
    }
}

int
main() {
    using namespace milvus::multivector;
    std::string ip = "127.0.0.1";
    std::string port = "19530";
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);
    std::string file_name = "/data/gist/gist_query.fvecs";
    std::vector<std::vector<float>> data;
    unsigned num, dim;
    load_data(file_name, data, num, dim);
//    split_data(data)
    testIndexType(engine, milvus::IndexType::FLAT, {{"nlist", 1024}}, {{"nprobe", 20}});
    testIndexType(engine, milvus::IndexType::IVFFLAT, {{"nlist", 1024}}, {{"nprobe", 20}});
    testIndexType(engine, milvus::IndexType::IVFSQ8, {{"nlist", 1024}}, {{"nprobe", 20}});
    testIndexType(engine, milvus::IndexType::IVFPQ, {{"nlist", 1024}, {"m", 20}}, {{"nprobe", 20}});
    testIndexType(engine, milvus::IndexType::IVFPQ, {{"nlist", 1024}, {"m", 20}}, {{"nprobe", 20}});
    testIndexType(engine, milvus::IndexType::RNSG,
                  {{"search_length", 45}, {"out_degree", 50}, {"candidate_pool_size", 300}, {"knng", 100}},
                  {{"search_length", 100}});
    testIndexType(engine, milvus::IndexType::HNSW,
                  {{"M", 16}, {"efConstruction", 500}},
                  {{"ef", 64}});
    testIndexType(engine, milvus::IndexType::ANNOY,
                  {{"n_trees", 8}},
                  {{"search_k", -1}});
}


