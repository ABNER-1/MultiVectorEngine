#include "MultiVectorEngine.h"
#include <random>
#include <nlohmann/json.hpp>

using namespace milvus::multivector;

void
normalizateVector(milvus::Entity& entity) {
    double mod_entities = 0.0;
    for (auto& entity_elem :entity.float_data) {
        mod_entities += entity_elem * entity_elem;
    }
    mod_entities = sqrt(mod_entities);
    for (auto& entity_elem :entity.float_data) {
        entity_elem /= mod_entities;
    }
}

void
generateVector(int64_t dim, milvus::Entity& entity) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(1.0, 2.0);

    entity.float_data.resize(dim);
    for (auto& elem: entity.float_data) {
        elem = static_cast<float>(dis(gen));
    }
}

void
generateArrays(int nq, const std::vector<int64_t>& dimensions,
               std::vector<milvus::multivector::RowEntity>& row_entities) {
    row_entities.resize(nq);
    for (int i = 0; i < nq; ++i) {
        auto& tmp_row_entity = row_entities[i];
        tmp_row_entity.resize(dimensions.size());
        for (auto j = 0; j < dimensions.size(); ++j) {
            generateVector(dimensions[j], tmp_row_entity[j]);
            normalizateVector(tmp_row_entity[j]);
        }
    }
}

void
generateIds(int nq, std::vector<int64_t>& id_arrays) {
    static int idx = 0;
    for (int i = 0; i < nq; ++i) {
        id_arrays.push_back(++idx);
    }
}

void
showResult(const milvus::TopKQueryResult& topk_query_result) {
    std::cout << "There are " << topk_query_result.size() << " query" << std::endl;
    for (auto& result : topk_query_result) {
        int len = result.ids.size();
        std::cout << "  This quert has " << len << " result." << std::endl;
        std::cout << "    First is [id:" << result.ids[0] << "] [distance:" << result.distances[0] << "]" << std::endl;
        std::cout << "    The " << len << "th is [id:" << result.ids[len - 1]
                  << "] [distance:" << result.distances[len - 1] << "]" << std::endl;
//        for (int i = 0; i < result.ids.size(); ++i) {
//            std::cout << "   " << result.ids[i] << " " << result.distances[i] << std::endl;
//        }
    }
}

void
testIndexType(std::shared_ptr<MultiVectorEngine> engine, milvus::IndexType index_type,
              const nlohmann::json& index_json, const nlohmann::json& query_json) {
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

    auto collection_name = "test_collection";
    std::vector<int64_t> dim{512, 128};
    std::vector<int64_t> index_file_sizes{1024, 1024};
    int vector_num = 100;
    int nq = 10;
    int topk = 20;
    std::vector<float> weight = {1, 1};

    assert_status(engine->CreateCollection(collection_name, milvus::MetricType::IP, dim, index_file_sizes));

    // generate insert data vector
    std::vector<RowEntity> row_entities;
    std::vector<int64_t> id_arrays;
    generateArrays(vector_num, dim, row_entities);
    generateIds(vector_num, id_arrays);

    assert_status(engine->Insert(collection_name, row_entities, id_arrays));
    assert_status(engine->CreateIndex(collection_name,
                                      index_type, index_json.dump()));

    // generate query vector
    std::vector<RowEntity> query_entities;
    query_entities.resize(nq);
//    generateArrays(nq, dim, query_entities);
    std::copy(row_entities.begin(), row_entities.begin() + nq, query_entities.begin());

    milvus::TopKQueryResult topk_result;
    assert_status(engine->Search(collection_name, weight,
                                 query_entities, topk, query_json.dump(), topk_result));
    showResult(topk_result);

    assert_status(engine->DropIndex(collection_name));
    assert_status(engine->Delete(collection_name, id_arrays));
    assert_status(engine->DropCollection(collection_name));
}

int
main() {
    using namespace milvus::multivector;
    std::string ip = "127.0.0.1";
    std::string port = "19530";
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);

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