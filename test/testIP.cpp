#include "MultiVectorEngine.h"
#include <random>


void
generateVector(int64_t dim, milvus::Entity &entity) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);

    entity.float_data.resize(static_cast<uint64_t>(dim));
    for (auto &elem: entity.float_data) {
        elem = static_cast<float>(dis(gen));
    }
}

void
generateArrays(int nq, const std::vector<int64_t> &dimensions,
               std::vector<milvus::multivector::RowEntity> &row_entities) {
    for (int i = 0; i < nq; ++i) {
        row_entities.emplace_back();
        auto &tmp_row_entity = row_entities[i];
        for (auto j = 0; j < dimensions.size(); ++j) {
            generateVector(dimensions[j], tmp_row_entity[j]);
        }
    }
}

void
generateIds(int nq, std::vector<int64_t> &id_arrays) {
    static int idx = 0;
    for (int i = 0; i < nq; ++i) {
        id_arrays.push_back(++idx);
    }
}

void
showResult(const milvus::TopKQueryResult &topk_query_result) {
    for (auto &result : topk_query_result) {
        std::cout << "query :" << std::endl;
        for (int i = 0; i < result.ids.size(); ++i) {
            std::cout << "   " << result.ids[i] << " " << result.distances[i] << std::endl;
        }
    }
}

int main() {
    using namespace milvus::multivector;
    std::string ip = "127.0.0.1";
    std::string port = "19530";

    auto collection_name = "test_collection";
    std::vector<int64_t> dim{512, 128};
    std::vector<int64_t> index_file_sizes{1024, 1024};


    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };
    int n = 100;
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);
    assert_status(engine->CreateCollection(collection_name, milvus::MetricType::IP, dim, index_file_sizes));

    std::vector<RowEntity> row_entities;
    std::vector<int64_t> id_arrays;
    generateArrays(n, dim, row_entities);
    generateIds(n, id_arrays);

    assert_status(engine->Insert(collection_name, row_entities, id_arrays));
    assert_status(engine->CreateIndex(collection_name, milvus::IndexType::IVFPQ, ""));

    int nq = 10;
    int topk = 20;
    std::vector<RowEntity> query_entities;
    milvus::TopKQueryResult topk_result;
    generateArrays(nq, dim, query_entities);
    assert_status(engine->Search(collection_name, {1.0, 2.1},
                                 query_entities, topk, "", topk_result));
    showResult(topk_result);

    assert_status(engine->DropIndex(collection_name));
    assert_status(engine->Delete(collection_name, id_arrays));
    assert_status(engine->DropCollection("test_collection"));


}