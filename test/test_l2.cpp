#include "MultiVectorEngine.h"
#include <random>
#include "utils.h"

using namespace milvus::multivector;

int
main() {
    using namespace milvus::multivector;
    std::string ip = "127.0.0.1";
    std::string port = "19530";
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);

    testIndexType(engine, milvus::IndexType::FLAT, {{"nlist", 1024}}, {{"nprobe", 20}}, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFFLAT, {{"nlist", 1024}}, {{"nprobe", 20}}, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFSQ8, {{"nlist", 1024}}, {{"nprobe", 20}}, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFPQ,
                  {{"nlist", 1024}, {"m", 16}},
                  {{"nprobe", 20}},
                  milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::IVFPQ,
                  {{"nlist", 1024}, {"m", 16}},
                  {{"nprobe", 20}},
                  milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::RNSG,
                  {{"search_length", 45}, {"out_degree", 50}, {"candidate_pool_size", 300}, {"knng", 100}},
                  {{"search_length", 100}}, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::HNSW,
                  {{"M", 16}, {"efConstruction", 500}},
                  {{"ef", 200}}, milvus::MetricType::L2);
    testIndexType(engine, milvus::IndexType::ANNOY,
                  {{"n_trees", 8}},
                  {{"search_k", -1}}, milvus::MetricType::L2);
}
