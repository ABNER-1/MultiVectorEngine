#include <iostream>
#include <fstream>
#include "MultiVectorEngine.h"
#include <random>
#include <chrono>
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

    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

    // create collection and insert data, once per test, save time
    srand((unsigned)time(nullptr));
    auto collection_name = "l2_test_collection" + std::to_string(random() % 100);
    int vec_group_num = config.at("group_num");
    int nq = config.at("nq");
    int topk = config.at("topk");
    int max_dim = 0;
    int precision = config.at("precision");
    int vector_num = config.at("nb");
    std::chrono::high_resolution_clock::time_point ts, te;
    std::vector<int64_t> dim;
    std::vector<int64_t> index_file_sizes;
    std::vector<std::string> base_data_locations;
    std::vector<std::string> query_data_locations;
    std::vector<float> weights;
    std::vector<int64_t> acc_dims(vec_group_num, 0);
    dim.clear();
    index_file_sizes.clear();
    base_data_locations.clear();
    query_data_locations.clear();
    weights.clear();
    for (auto i = 0; i < vec_group_num; ++i) {
        base_data_locations.emplace_back(config.at("base_data_locations")[i]);
        query_data_locations.emplace_back(config.at("query_data_locations")[i]);
        dim.emplace_back(config.at("dimensions")[i]);
        if(max_dim < config.at("dimensions")[i])
            max_dim = config.at("dimensions")[i];
        acc_dims[i] = i ? acc_dims[i - 1] + dim[i - 1] : 0;
        weights.emplace_back(config.at("weights")[i]);
        index_file_sizes.push_back(2048);
    }

    assert_status(engine->CreateCollection(collection_name, milvus::MetricType::L2, dim, index_file_sizes));

    std::vector<std::vector<int64_t>> all_id_arrays;
    std::vector<RowEntity> row_entities(vector_num, RowEntity(vec_group_num, milvus::Entity()));
    std::vector<int64_t> ids;
    generateIds(vector_num, ids, -1);
    int64_t limit_insert = 256 * 1024 * 1024;
    auto limit_insert_rows = limit_insert / 4 / max_dim;
    std::cout << "limit insert rows is: " << limit_insert_rows << std::endl;
    ts = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < vec_group_num; ++i) {
        std::ifstream fin(base_data_locations[i], std::ios::in);
        fin.precision(precision);
        for (auto j = 0; j < vector_num; ++j) {
            row_entities[j][i].float_data.resize(dim[i]);
            for (auto k = 0; k < dim[i]; ++k) {
                fin >> row_entities[j][i].float_data[k];
            }
        }
        fin.close();
    }
    te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "LoadRowData costs " << search_duration << " ms." << std::endl;
    int insert_cnt = 0;
    ts = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < vector_num; i += limit_insert_rows) {
        auto lb = i;
        auto ub = (i + limit_insert_rows > vector_num) ? vector_num : i + limit_insert_rows;
        std::vector<RowEntity> insert_batch(ub - lb, RowEntity(vec_group_num, milvus::Entity()));
        std::vector<int64_t> ids_batch(ub - lb);
        for (auto ii = lb; ii < ub; ++ ii) {
            ids_batch[ii - i] = ids[ii];
            for (auto j = 0; j < vec_group_num; ++ j) {
                insert_batch[ii - i][j].float_data.resize(dim[j]);
                for (auto k = 0; k < dim[j]; ++ k)
                    insert_batch[ii - i][j].float_data[k] = row_entities[ii][j].float_data[k];
            }
        }
        insert_cnt += (ub - lb);
        assert_status(engine->Insert(collection_name, insert_batch, ids_batch));
        std::cout << "Insert " << ub - lb << " vectors into milvus." << std::endl;
        std::vector<RowEntity>().swap(insert_batch);
        std::vector<int64_t>().swap(ids_batch);
    }
    assert_status(engine->Flush(collection_name));
    te = std::chrono::high_resolution_clock::now();
    std::cout << "Insert " << insert_cnt << "/" << vector_num << " vectors into milvus." << std::endl;
    search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "Insert data costs " << search_duration << " ms." << std::endl;
//    testIndexType(engine, milvus::IndexType::FLAT, {{"nlist", 1024}}, {{"nprobe", 20}}, config, milvus::MetricType::L2);
    nlohmann::json search_args = {{"nprobe", 80}};
    testIndexType(engine, milvus::IndexType::IVFFLAT, {{"nlist", 1024}}, search_args, config, milvus::MetricType::L2, collection_name);
//    testIndexType(engine, milvus::IndexType::IVFSQ8, {{"nlist", 1024}}, {{"nprobe", 20}}, config, milvus::MetricType::L2);
//    testIndexType(engine, milvus::IndexType::IVFPQ,
//                  {{"nlist", 1024}, {"m", 32}},
//                  {{"nprobe", 20}}, config,
//                  milvus::MetricType::L2);
//    testIndexType(engine, milvus::IndexType::RNSG,
//                  {{"search_length", 45}, {"out_degree", 50}, {"candidate_pool_size", 300}, {"knng", 100}},
//                  {{"search_length", 100}}, config, milvus::MetricType::L2);
    search_args = {{"ef", 500}};
    testIndexType(engine, milvus::IndexType::HNSW,
                  {{"M", 32}, {"efConstruction", 200}},
                  search_args, config, milvus::MetricType::L2, collection_name);
//    testIndexType(engine, milvus::IndexType::ANNOY,
//                  {{"n_trees", 8}},
//                  {{"search_k", -1}}, config, milvus::MetricType::L2);
    assert_status(engine->DropCollection(collection_name));
}
