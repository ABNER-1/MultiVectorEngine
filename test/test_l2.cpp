#include <iostream>
#include <fstream>
#include "MultiVectorEngine.h"
#include <random>
#include "utils.h"
#include "nlohmann/json.hpp"

using namespace milvus::multivector;

int
main(int argc, char **argv) {
    std::string config_file;
    if (argc != 2) {
        std::cout << "there must be one parameter 4 this program which tells the location of config file!" << std::endl;
        return EXIT_FAILURE;
    }
    config_file = std::string(argv[1]);
    nlohmann::json config;
    std::ifstream f_conf(config_file);
    f_conf >> config;
    f_conf.close();
    using namespace milvus::multivector;
    std::string ip = "127.0.0.1";
    std::string port = "19530";
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);

    testIndexType(engine, milvus::IndexType::FLAT, {{"nlist", 1024}}, {{"nprobe", 20}}, config, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFFLAT, {{"nlist", 1024}}, {{"nprobe", 20}}, config, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFSQ8, {{"nlist", 1024}}, {{"nprobe", 20}}, config, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFPQ,
                  {{"nlist", 1024}, {"m", 32}},
                  {{"nprobe", 20}}, config,
                  milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::RNSG,
                  {{"search_length", 45}, {"out_degree", 50}, {"candidate_pool_size", 300}, {"knng", 100}},
                  {{"search_length", 100}}, config, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::HNSW,
                  {{"M", 16}, {"efConstruction", 500}},
                  {{"ef", 200}}, config, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::ANNOY,
                  {{"n_trees", 8}},
                  {{"search_k", -1}}, config, milvus::MetricType::L2);
}
