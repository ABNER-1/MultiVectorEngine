#include <map>
#include <chrono>
#include "utils.h"
//#include <hdf5/serial/hdf5.h>
//#include <hdf5/serial/H5Cpp.h>
//#include <hdf5.h>
//#include <H5Cpp.h>

void
normalizeVector(milvus::Entity& entity) {
    double mod_entities = 0.0;
    for (auto& entity_elem :entity.float_data) {
        mod_entities += entity_elem * entity_elem;
    }
    if (mod_entities == 0) {
        return;
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
            normalizeVector(tmp_row_entity[j]);
        }
    }
}

int
readArrays(const std::string& file_name, const std::vector<int64_t>& dimensions,
           std::vector<milvus::multivector::RowEntity>& row_entities,
           int page_num, int page) {
    std::vector<std::vector<float>> data;
    unsigned num, dim;
    loadDataFromFvec(file_name, data, num, dim, page_num, page);
    split_data(data, row_entities, dimensions);
    return row_entities.size();
}

int
readArraysFromSplitedData(const std::vector<std::string>& file_names,
                          const std::vector<int64_t>& dimensions,
                          std::vector<milvus::multivector::RowEntity>& row_entities,
                          int page_num, int page, int lines) {
    auto result_len = lines <= page_num * (page + 1) ? lines - page_num * page : page_num;
    row_entities.resize(result_len);
    for (auto i = 0; i < file_names.size(); ++i) {
        auto& file_name = file_names[i];
        auto dim = dimensions[i];
        std::ifstream in(file_name.c_str(), std::ios::in);
        in.seekg(0, std::ios::beg);
        in.precision(18);

        // skip previous lines already read
        for (auto j = 0; j < page_num * page; ++j) {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        for (auto j = 0; j < result_len; ++j) {
            row_entities[j].emplace_back();
            row_entities[j][i].float_data.resize(dim);
            for (auto k = 0; k < dim; ++k) {
                in >> row_entities[j][i].float_data[k];
            }
        }
        in.close();
    }
    return row_entities.size();
}

static int generateIds_idx = -1;

void
resetIds() {
    generateIds_idx = -1;
}

void
generateIds(int nq, std::vector<int64_t>& id_arrays) {
    for (int i = 0; i < nq; ++i) {
        id_arrays.push_back(++generateIds_idx);
    }
}

void
generateIds(int nq, std::vector<int64_t>& id_arrays, int base_id) {
    for (int i = 0; i < nq; ++i) {
        id_arrays.push_back(++base_id);
    }
}

void
writeBenchmarkResult(const milvus::TopKQueryResult& topk_query_result,
                     const std::string& result_file,
                     float total_time, int topk) {
    std::cout << "There are " << topk_query_result.size() << " query" << std::endl;
    std::ofstream out(result_file);
    out.precision(18);
    out << topk_query_result.size() << " " << topk_query_result[0].ids.size()
        << " " << total_time << std::endl;
    for (auto& result : topk_query_result) {
        for (int i = 0; i < topk; ++i) {
            if (i > topk_query_result.size())out << -1 << std::endl;
            out << result.ids[i] << " " << result.distances[i] << std::endl;
        }
    }
    out.close();
}

void
showResult(const milvus::TopKQueryResult& topk_query_result) {
    std::cout.precision(18);
    std::cout << "There are " << topk_query_result.size() << " query" << std::endl;
    for (auto& result : topk_query_result) {
        int len = result.ids.size();
        std::cout << "  This query has " << len << " result." << std::endl;
        std::cout << "    First is [id:" << result.ids[0] << "] [distance:" << result.distances[0] << "]" << std::endl;
        std::cout << "    The " << len << "th is [id:" << result.ids[len - 1]
                  << "] [distance:" << result.distances[len - 1] << "]" << std::endl;

        for (int i = 0; i < result.ids.size(); ++i) {
            std::cout << i << " " << result.ids[i] << " " << result.distances[i] << std::endl;
        }
    }
}

void
showResultL2(const milvus::TopKQueryResult& topk_query_result) {
    std::cout.precision(8);
    std::cout << "There are " << topk_query_result.size() << " query" << std::endl;
    int cnt = 0;
    for (auto& result : topk_query_result) {
        std::cout << "query: " << ++cnt << std::endl;
        auto show_len = 10 < result.ids.size() ? 3 : result.ids.size();
        for (auto i = 0; i < show_len; ++i) {
            std::cout << "(" << result.ids[i] << ", " << result.distances[i] << ")";
            if (i == show_len - 1)
                std::cout << std::endl;
            else
                std::cout << " ";
        }
    }
}

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              nlohmann::json& query_json,
              milvus::MetricType metric_type) {
    using namespace milvus::multivector;
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

    auto collection_name = "test_collection144212";
    std::vector<int64_t> dim{512, 128, 320};
    std::vector<int64_t> index_file_sizes{1024, 1024, 1024};
    int nq = 100;
    int topk = 20;
    std::vector<float> weight = {1, 1, 1};

    assert_status(engine->CreateCollection(collection_name, metric_type, dim, index_file_sizes));

    // generate insert data vector
    int vector_num = 10000;
    std::vector<RowEntity> query_entities;
    std::vector<std::vector<int64_t>> all_id_arrays;
    int page_id = 0;
    // generate query vector
    query_entities.resize(nq);
    while (true) {
        std::vector<RowEntity> row_entities;
        vector_num = readArrays("/data/gist/gist_base.fvecs", dim, row_entities, 10000, page_id);
        if (vector_num == 0) break;
//    generateArrays(vector_num, dim, row_entities);
        std::vector<int64_t> id_arrays;
        generateIds(vector_num, id_arrays);
        std::copy(row_entities.begin(), row_entities.begin() + nq, query_entities.begin());
        assert_status(engine->Insert(collection_name, row_entities, id_arrays));
        std::cout << "Insert " << vector_num << " into milvus." << std::endl;
        all_id_arrays.emplace_back(std::move(id_arrays));
        ++page_id;
    }

//    readArrays("/data/gist/gist_query.fvecs", dim, row_entities);
//    generateArrays(nq, dim, query_entities);
    assert_status(engine->CreateIndex(collection_name,
                                      index_type, index_json.dump()));

    milvus::TopKQueryResult topk_result;
    assert_status(engine->Search(collection_name, weight,
                                 query_entities, topk, query_json, topk_result));
    showResult(topk_result);

    assert_status(engine->DropIndex(collection_name));
    for (auto& id_arrays: all_id_arrays) {
        assert_status(engine->Delete(collection_name, id_arrays));
    }

    assert_status(engine->DropCollection(collection_name));
}

void
loadDataFromFvec(const std::string& filename,
                 std::vector<std::vector<float>>& vector_data,
                 unsigned& num, unsigned& dim, int page_num, int page) {
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

    auto start = page_num * page;
    in.seekg(0, std::ios::beg);
    in.seekg((4 * (dim + 1)) * start, std::ios::cur);
    for (size_t i = 0; i < page_num && start + i < num; ++i) {
        vector_data.emplace_back(dim);
        auto data = vector_data[i].data();
        in.seekg(4, std::ios::cur);
        in.read((char*)(data), dim * 4);
    }
    in.close();
}

void
load_data(const std::string& filename,
          std::vector<std::vector<int>>& vector_data,
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
    vector_data.resize(num, std::vector<int>(dim));

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
split_data(const std::vector<std::vector<float>>& raw_data,
           std::vector<std::vector<milvus::Entity>>& splited_data,
           const std::vector<int64_t>& dims) {
    // vertify dims
    auto total_dims = std::accumulate(dims.begin(), dims.end(), 0ll);
    if (raw_data.empty())return;
    if (total_dims > raw_data[0].size()) {
        std::cerr << "input total dims lager than raw data dims";
    }
    splited_data.resize(raw_data.size());
    for (int i = 0; i < raw_data.size(); ++i) {
        int idx = 0;
        splited_data[i].resize(dims.size());
        for (int j = 0; j < dims.size(); ++j) {
            for (int k = 0; k < dims[j]; ++k) {
                splited_data[i][j].float_data.push_back(raw_data[i][idx + k]);
            }
            normalizeVector(splited_data[i][j]);
            idx += dims[j];
        }
    }
}

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              nlohmann::json& query_json,
              const nlohmann::json& config,
              milvus::MetricType metric_type) {
    using namespace milvus::multivector;
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

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
        if (max_dim < config.at("dimensions")[i])
            max_dim = config.at("dimensions")[i];
        acc_dims[i] = i ? acc_dims[i - 1] + dim[i - 1] : 0;
        weights.emplace_back(config.at("weights")[i]);
        index_file_sizes.push_back(2048);
    }

    assert_status(engine->CreateCollection(collection_name, metric_type, dim, index_file_sizes));

// generate insert data vector
    std::vector<RowEntity> query_entities(nq, RowEntity(vec_group_num, milvus::Entity()));
//    std::vector<RowEntity> query_entities(1, RowEntity(vec_group_num, milvus::Entity()));
    std::vector<std::vector<int64_t>> all_id_arrays;
    // generate query vector
//    query_entities.resize(nq);
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
        for (auto ii = lb; ii < ub; ++ii) {
            ids_batch[ii - i] = ids[ii];
            for (auto j = 0; j < vec_group_num; ++j) {
                insert_batch[ii - i][j].float_data.resize(dim[j]);
                for (auto k = 0; k < dim[j]; ++k)
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
    for (auto i = 0; i < vec_group_num; ++i) {
        std::ifstream fin(query_data_locations[i], std::ios::in);
        fin.precision(precision);
        for (auto j = 0; j < nq; ++j) {
            query_entities[j][i].float_data.resize(dim[i]);
//            query_entities[0][i].float_data.resize(dim[i]);
            for (auto k = 0; k < dim[i]; ++k) {
                fin >> query_entities[j][i].float_data[k];
            }
        }
        fin.close();
    }
    /*
    for (auto i = 0; i < nq; ++i) {
        for (auto j = 0; j < vec_group_num; ++j) {
            query_entities[i][j].float_data.resize(dim[j]);
            for (auto k = 0; k < dim[j]; ++k)
                query_entities[i][j].float_data[k] = row_entities[i][j].float_data[k];
        }
    }
    */
    ts = std::chrono::high_resolution_clock::now();
    assert_status(engine->CreateIndex(collection_name, index_type, index_json.dump()));
    te = std::chrono::high_resolution_clock::now();
    search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "CreateIndex costs " << search_duration << " ms." << std::endl;

    milvus::TopKQueryResult topk_result;
    ts = std::chrono::high_resolution_clock::now();
    assert_status(engine->Search(collection_name, weights, query_entities, topk, query_json, topk_result));
    te = std::chrono::high_resolution_clock::now();
    search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "Search costs " << search_duration << " ms." << std::endl;
    showResultL2(topk_result);

    assert_status(engine->DropIndex(collection_name));
    assert_status(engine->DropCollection(collection_name));
}

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              nlohmann::json& query_json,
              const nlohmann::json& config,
              milvus::MetricType metric_type,
              const std::string& collection_name,
              const std::string& result_file,
              const std::vector<int> &search_args,
              int &file_cnt) {
    using namespace milvus::multivector;
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

    int vec_group_num = config.at("group_num");
    int nq = config.at("nq");
    int topk = config.at("topk");
    int max_dim = 0;
    int precision = config.at("precision");
    std::chrono::high_resolution_clock::time_point ts, te;
    std::vector<int64_t> dim;
    std::vector<int64_t> index_file_sizes;
    std::vector<std::string> query_data_locations;
    std::vector<float> weights;
    std::vector<int64_t> acc_dims(vec_group_num, 0);
    dim.clear();
    index_file_sizes.clear();
    query_data_locations.clear();
    weights.clear();
    for (auto i = 0; i < vec_group_num; ++i) {
        query_data_locations.emplace_back(config.at("query_data_locations")[i]);
        dim.emplace_back(config.at("dimensions")[i]);
        if (max_dim < config.at("dimensions")[i])
            max_dim = config.at("dimensions")[i];
        acc_dims[i] = i ? acc_dims[i - 1] + dim[i - 1] : 0;
        weights.emplace_back(config.at("weights")[i]);
        index_file_sizes.push_back(2048);
    }


    //  omp nq
    std::vector<RowEntity> query_entities(nq, RowEntity(vec_group_num, milvus::Entity()));
    for (auto i = 0; i < vec_group_num; ++i) {
        std::ifstream fin(query_data_locations[i], std::ios::in);
        fin.precision(precision);
        for (auto j = 0; j < nq; ++j) {
            query_entities[j][i].float_data.resize(dim[i]);
            for (auto k = 0; k < dim[i]; ++k) {
                fin >> query_entities[j][i].float_data[k];
            }
        }
        fin.close();
    }

    /*
     * batch
    std::vector<RowEntity> query_entities(vec_group_num, RowEntity(nq, milvus::Entity()));

    for (auto i = 0; i < vec_group_num; ++i) {
        std::ifstream fin(query_data_locations[i], std::ios::in);
        fin.precision(precision);
        for (auto j = 0; j < nq; ++j) {
            query_entities[i][j].float_data.resize(dim[i]);
            for (auto k = 0; k < dim[i]; ++k) {
                fin >> query_entities[i][j].float_data[k];
            }
        }
        fin.close();
    }
    */

    assert_status(engine->DropIndex(collection_name));
    ts = std::chrono::high_resolution_clock::now();
    assert_status(engine->CreateIndex(collection_name, index_type, index_json.dump()));
    te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "CreateIndex costs " << search_duration << " ms." << std::endl;

    milvus::TopKQueryResult topk_result;
    query_json["ef"] = 1;
    query_json["nprobe"] = 1;
    query_json["print_milvus"] = false;
    assert_status(engine->Search(collection_name, weights, query_entities, topk, query_json, topk_result));
//    assert_status(engine->SearchBatch(collection_name, weights, query_entities, topk, query_json, topk_result));
    for (auto &sa : search_args) {
        std::string result_file_ = result_file + std::to_string(++ file_cnt) + ".txt";
        std::cout << "search args: " << sa << std::endl;
        query_json["ef"] = sa;
        query_json["nprobe"] = sa;
//        std::cout << "query_json:" << std::endl;
//        std::cout << query_json.dump() << std::endl;
//        std::cout << "query_json.at nlist:" << query_json.at("nlist") << std::endl;
        if (index_type == milvus::IndexType::IVFFLAT && sa > index_json["nlist"]) {
            std::cout << "pass nlist = " << index_json["nlist"] << " but nprobe = " << sa << std::endl;
            continue;
        }
//        if (index_type == milvus::IndexType::HNSW && query_json["ef"] == 4096)
//            query_json["print_milvus"] = true;
        ts = std::chrono::high_resolution_clock::now();
        assert_status(engine->Search(collection_name, weights, query_entities, topk, query_json, topk_result));
//        assert_status(engine->SearchBatch(collection_name, weights, query_entities, topk, query_json, topk_result));
        te = std::chrono::high_resolution_clock::now();
        search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
        std::cout << "Search costs " << search_duration << " ms." << std::endl;
        writeBenchmarkResult(topk_result, result_file_, search_duration, topk);
//        auto real_topks = engine->GetActualTopk(collection_name);
//        std::unordered_map<int, int> hash_tpk;
//        for (auto &tpk : real_topks) {
//            if (hash_tpk.find(tpk) != hash_tpk.end())
//                hash_tpk[tpk] ++;
//            else
//                hash_tpk[tpk] = 1;
//        }
//        std::cout << "real topk stat: " << std::endl;
//        for (auto it = hash_tpk.begin(); it != hash_tpk.end(); ++ it) {
//            std::cout << "topk = " << it->first << ", cnt = " << it->second << std::endl;
//        }
//        for (auto &pr : hash_tpk) {
//            std::cout << "topk = " << pr.first << ", cnt = " << pr.second << std::endl;
//        }
    }
//    showResultL2(topk_result);

    assert_status(engine->DropIndex(collection_name));
}
