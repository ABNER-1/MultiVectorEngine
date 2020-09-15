#include "MultiVectorEngine.h"
#include <iostream>
#include <chrono>
#include "utils.h"

using namespace milvus::multivector;

void
testIndexTypeIPNra(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
                   milvus::IndexType index_type,
                   const nlohmann::json& index_json,
                   nlohmann::json& query_json,
                   const nlohmann::json& config,
                   const std::string& result_file_name) {
    using namespace milvus::multivector;
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };
    auto metric = milvus::MetricType::IP;
    std::string strategy = "nra";
    std::vector<std::string> file_names, test_file_names;

    int all_lines = config.at("nb");
    auto collection_name = "test_collection3";
    std::vector<int64_t> dim, index_file_sizes;
    int nq = config.at("nq");
    int topk = config.at("topk");
    int vec_group_num = config.at("group_num");
    std::vector<float> weight;
    for (auto i = 0; i < vec_group_num; ++i) {
        file_names.emplace_back(config.at("base_data_locations")[i]);
        test_file_names.emplace_back(config.at("query_data_locations")[i]);
        dim.emplace_back(config.at("dimensions")[i]);
        weight.emplace_back(config.at("weights")[i]);
        index_file_sizes.push_back(1024);
    }
    assert_status(engine->CreateCollection(collection_name, metric, dim, index_file_sizes, strategy));

    // generate insert data vector
    int vector_size = config.at("batch");
    int page_id = 0;
    std::vector<RowEntity> query_entities;
    std::vector<std::vector<int64_t>> all_id_arrays;

    // generate query vector
    query_entities.resize(nq);

    bool end_flag = false;
    while (!end_flag) {
        std::vector<RowEntity> row_entities;
        if (all_lines <= vector_size * (page_id + 1)) {
            end_flag = true;
        }
        auto tmp_vector_num = readArraysFromSplitedData(file_names, dim, row_entities, vector_size, page_id, all_lines);
//        if (end_flag)
//            std::copy(row_entities.begin(), row_entities.begin() + nq, query_entities.begin());

        std::vector<int64_t> id_arrays;
        generateIds(tmp_vector_num, id_arrays);

        assert_status(engine->Insert(collection_name, row_entities, id_arrays));
        //std::cout << "Insert " << tmp_vector_num << " into milvus." << std::endl;
        all_id_arrays.emplace_back(std::move(id_arrays));
        ++page_id;
    }
    engine->Flush(collection_name);
    assert_status(engine->CreateIndex(collection_name,
                                      index_type, index_json.dump()));

    readArraysFromSplitedData(test_file_names, dim, query_entities, nq, 0, nq);
    milvus::TopKQueryResult topk_result;
    auto ts = std::chrono::high_resolution_clock::now();
    assert_status(engine->Search(collection_name, weight,
                                 query_entities, topk, query_json, topk_result));
//    int cnt = 0;
//    for (auto &res : topk_result) {
//        std::cout << "the " << ++ cnt << "th query:" << std::endl;
//        for (auto i = 0; i < res.ids.size(); ++ i) {
//            std::cout << "id = " << res.ids[i] << ", dis = " << res.distances[i] << std::endl;
//        }
//    }
    auto te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    writeBenchmarkResult(topk_result, result_file_name, search_duration, topk);

    assert_status(engine->DropIndex(collection_name));
    for (auto& id_arrays: all_id_arrays) {
        assert_status(engine->Delete(collection_name, id_arrays));
    }
    engine->Flush(collection_name);

    assert_status(engine->DropCollection(collection_name));
    resetIds();
}

int
main(int argc, char** argv) {
    using namespace milvus::multivector;
    std::string config_file = std::string(argv[1]);
    nlohmann::json config;
    std::ifstream f_conf(config_file);
    f_conf >> config;
    f_conf.close();

    std::string ip = "127.0.0.1";
    std::string port = "19530";
    auto engine = std::make_shared<MultiVectorEngine>(ip, port);
    std::string result_prefix = config.at("result_prefix");
//    std::vector<int> nlists = {128, 256, 512, 1024, 2048};
//    std::vector<int> nprobes = {1, 4, 5, 10, 20, 50, 100, 128};
    std::vector<int> nlists = {2048};
    std::vector<int> nprobes = {4};
    int number = 0;
    for (auto nlist : nlists) {
        for (auto nprobe : nprobes) {
            std::cout<<" nlist: "<<nlist<<" ; nprobe: "<< nprobe<<std::endl;
            auto result_file_name = result_prefix + std::to_string(++number) + ".txt";
            nlohmann::json search_args = {{"nprobe", nprobe}};
            testIndexTypeIPNra(engine, milvus::IndexType::IVFFLAT,
                               {{"nlist", nlist}}, search_args,
                               config, result_file_name);
        }
    }

//    testIndexType(engine, milvus::IndexType::HNSW,
//                  {{"M", 16}, {"efConstruction", 500}},
//                  {{"ef", 200}});

}

