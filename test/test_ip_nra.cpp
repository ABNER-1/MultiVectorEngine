#include "MultiVectorEngine.h"
#include <iostream>
#include "utils.h"

using namespace milvus::multivector;

void
testIndexTypeIPNra(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
                milvus::IndexType index_type,
                const nlohmann::json& index_json,
                const nlohmann::json& query_json,
                const std::string& strategy) {
    using namespace milvus::multivector;
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

    auto metric_type = milvus::MetricType::IP;
    std::vector<std::string> file_names =
        {"/home/abner/vector/train64.txt", "/home/abner/vector/train64-1.txt", "/home/abner/vector/train72.txt"};
    auto collection_name = "test_collection";
    std::vector<int64_t> dim{64, 64, 72};
    std::vector<int64_t> index_file_sizes{1024, 1024, 1024};
    int nq = 10;
    int topk = 10;
    std::vector<float> weight = {1, 1, 1};

    assert_status(engine->CreateCollection(collection_name, metric_type, dim, index_file_sizes, strategy));

    // generate insert data vector
    int vector_num = 10000;
    std::vector<RowEntity> query_entities;
    std::vector<std::vector<int64_t>> all_id_arrays;
    int page_id = 0;
    // generate query vector
    query_entities.resize(nq);

    std::vector<RowEntity> row_entities;
    readArraysFromSplitedData(file_names, dim, row_entities, vector_num, page_id);
    std::copy(row_entities.begin(), row_entities.begin() + nq, query_entities.begin());

    std::vector<int64_t> id_arrays;
    generateIds(vector_num, id_arrays);

    assert_status(engine->Insert(collection_name, row_entities, id_arrays));
    std::cout << "Insert " << vector_num << " into milvus." << std::endl;
    all_id_arrays.emplace_back(std::move(id_arrays));

    assert_status(engine->CreateIndex(collection_name,
                                      index_type, index_json.dump()));

    milvus::TopKQueryResult topk_result;

    assert_status(engine->Search(collection_name, weight,
                                 query_entities, topk, query_json.dump(), topk_result));
    showResult(topk_result);

    assert_status(engine->DropIndex(collection_name));
    for (auto& id_arrays: all_id_arrays) {
        assert_status(engine->Delete(collection_name, id_arrays));
    }

    assert_status(engine->DropCollection(collection_name));
}

int
main() {
    using namespace milvus::multivector;
    std::string ip = "127.0.0.1";
    std::string port = "19530";
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);

//    testIndexType(engine, milvus::IndexType::FLAT, {{"nlist", 1024}}, {{"nprobe", 20}});
    testIndexTypeIPNra(engine,
                    milvus::IndexType::FLAT,
                    {{"nlist", 1024}},
                    {{"nprobe", 1024}},
                    "nra");
//    testIndexTypeIP(engine, milvus::IndexType::IVFFLAT, {{"nlist", 1024}}, {{"nprobe", 20}});
//    testIndexType(engine, milvus::IndexType::IVFSQ8, {{"nlist", 1024}}, {{"nprobe", 20}});
//    testIndexType(engine, milvus::IndexType::IVFPQ, {{"nlist", 1024}, {"m", 48}}, {{"nprobe", 20}});
//    testIndexType(engine, milvus::IndexType::IVFPQ, {{"nlist", 1024}, {"m", 48}}, {{"nprobe", 20}});
//    testIndexType(engine, milvus::IndexType::RNSG,
//                  {{"search_length", 45}, {"out_degree", 50}, {"candidate_pool_size", 300}, {"knng", 100}},
//                  {{"search_length", 100}});
//    testIndexType(engine, milvus::IndexType::HNSW,
//                  {{"M", 16}, {"efConstruction", 500}},
//                  {{"ef", 200}});
//    testIndexType(engine, milvus::IndexType::ANNOY,
//                  {{"n_trees", 8}},
//                  {{"search_k", -1}});
}
