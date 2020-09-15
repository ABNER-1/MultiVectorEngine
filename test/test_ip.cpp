#include "MultiVectorEngine.h"
#include <iostream>
#include <chrono>
#include "utils.h"

using namespace milvus::multivector;
namespace {
std::string ip = "127.0.0.1";
std::string port = "19530";
auto collection_name = "test_collection11";
std::string strategy = "default";
auto metric = milvus::MetricType::IP;
std::vector<std::vector<int64_t>> all_id_arrays;
std::vector<RowEntity> query_entities;
nlohmann::json ip_config;
int nq, topk, all_lines;
std::vector<int64_t> dim, index_file_sizes;
std::vector<std::string> file_names, test_file_names;
std::string ivf_result_prefix, hnsw_result_prefix;
std::vector<float> weight;
}

void
assert_status(milvus::Status&& status) {
    if (!status.ok()) {
        std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
    }
}

void
CreateCollection(std::shared_ptr<milvus::multivector::MultiVectorEngine>& engine) {
    assert_status(engine->CreateCollection(collection_name, metric, dim, index_file_sizes, strategy));
}

void
Insert(std::shared_ptr<milvus::multivector::MultiVectorEngine>& engine) {
    int vector_size = ip_config.at("batch");
    int page_id = 0;
    bool end_flag = false;
    while (!end_flag) {
        std::vector<RowEntity> row_entities;
        if (all_lines <= vector_size * (page_id + 1)) {
            end_flag = true;
        }
        auto tmp_vector_num =
            readArraysFromSplitedData(file_names, dim, row_entities, vector_size, page_id, all_lines);
        std::vector<int64_t> id_arrays;
        generateIds(tmp_vector_num, id_arrays);

        assert_status(engine->Insert(collection_name, row_entities, id_arrays));
        std::cout << "Insert " << tmp_vector_num << " into milvus." << std::endl;
        all_id_arrays.emplace_back(std::move(id_arrays));
        ++page_id;
    }
    engine->Flush(collection_name);

    // generate query vector
    query_entities.resize(nq);
    readArraysFromSplitedData(test_file_names, dim, query_entities, nq, 0, nq);
}

void
DropCollection(std::shared_ptr<milvus::multivector::MultiVectorEngine>& engine) {
    for (auto& id_arrays: all_id_arrays) {
        assert_status(engine->Delete(collection_name, id_arrays));
    }
    engine->Flush(collection_name);

    assert_status(engine->DropCollection(collection_name));
    resetIds();
}

void
CreateIndex(std::shared_ptr<milvus::multivector::MultiVectorEngine>& engine,
            milvus::IndexType index_type, const nlohmann::json& index_json) {
    using namespace milvus::multivector;
    assert_status(engine->CreateIndex(collection_name,
                                      index_type, index_json.dump()));
}

void
DropIndex(std::shared_ptr<milvus::multivector::MultiVectorEngine>& engine) {
    using namespace milvus::multivector;
    assert_status(engine->DropIndex(collection_name));
}

void
Search(std::shared_ptr<milvus::multivector::MultiVectorEngine>& engine,
       nlohmann::json& query_json, const std::string& result_file_name) {
    using namespace milvus::multivector;
    milvus::TopKQueryResult topk_result;
    auto ts = std::chrono::high_resolution_clock::now();
    assert_status(engine->Search(collection_name, weight,
                                 query_entities, topk, query_json, topk_result));
    auto te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    writeBenchmarkResult(topk_result, result_file_name, search_duration, topk);
}

void
writeTopk(const std::vector<int>& topks) {

    std::ofstream out("./topk.txt", std::ofstream::out | std::ofstream::app);
    out << topks.size() << " ";
    for (auto& tmp_topk : topks) {
        out << tmp_topk << " ";
    }
    out << std::endl;
    out.close();
}

int
main(int argc, char** argv) {
    using namespace milvus::multivector;
    std::string config_file = std::string(argv[1]);
    std::ifstream f_conf(config_file);
    f_conf >> ip_config;
    f_conf.close();

    auto engine = std::make_shared<MultiVectorEngine>(ip, port);
    {
        all_lines = ip_config.at("nb");
        nq = ip_config.at("nq");
        topk = ip_config.at("topk");
        strategy = ip_config.at("strategy");
        int vec_group_num = ip_config.at("group_num");
        for (auto i = 0; i < vec_group_num; ++i) {
            file_names.emplace_back(ip_config.at("base_data_locations")[i]);
            test_file_names.emplace_back(ip_config.at("query_data_locations")[i]);
            dim.emplace_back(ip_config.at("dimensions")[i]);
            index_file_sizes.push_back(1024);
            weight.emplace_back(ip_config.at("weights")[i]);
        }
    }
    CreateCollection(engine);
    Insert(engine);
    ivf_result_prefix = ip_config.at("ivf_result_prefix");
    hnsw_result_prefix = ip_config.at("hnsw_result_prefix");
    std::vector<int> nlists = {4096};
    std::vector<int> nprobes =
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 60, 80, 120, 240, 360, 480, 512, 1024, 2048, 4096};

    int number = 0;
    for (auto nlist : nlists) {
        CreateIndex(engine, milvus::IndexType::IVFFLAT, {{"nlist", nlist}});
        for (auto nprobe : nprobes) {
            std::cout << number << " nlist: " << nlist << " ; nprobe: " << nprobe << std::endl;
            auto result_file_name = ivf_result_prefix + std::to_string(++number) + ".txt";
            nlohmann::json search_params = {{"nprobe", nprobe}};
            Search(engine, search_params, result_file_name);
            auto topks = engine->GetActualTopk(collection_name);
            writeTopk(topks);
        }
        DropIndex(engine);
    }

//    std::vector<int> ms = {4, 16, 48};
    std::vector<int> ms = {4, 8};
    std::vector<int> efcs = {8, 9, 10, 12, 16, 32};
//    std::vector<int> efcs = {8, 16, 100, 512};
    std::vector<int> efs = {10, 50, 80, 140, 300, 1024, 2048, 4096};
    number = 0;
//    for (auto m : ms) {
//        for (auto& efc: efcs) {
//            CreateIndex(engine, milvus::IndexType::HNSW, {{"M", m}, {"efConstruction", efc}});
//            for (auto ef : efs) {
//                if (ef < topk) continue;
//                std::cout << number << " M: " << m << " ; efc: " << efc << "; ef: " << ef << std::endl;
//                auto result_file_name = hnsw_result_prefix + std::to_string(++number) + ".txt";
//                nlohmann::json search_params = {{"ef", ef}};
//                Search(engine, search_params, result_file_name);
//                auto topks = engine->GetActualTopk(collection_name);
//                writeTopk(topks);
//            }
//        }
//        DropIndex(engine);
//    }

    DropCollection(engine);
}


