#include "utils.h"

void
normalizeVector(milvus::Entity& entity) {
    double mod_entities = 0.0;
    for (auto& entity_elem :entity.float_data) {
        if (entity_elem < 0) {
            std::cerr << "native element" << std::endl;
        }
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
    load_data(file_name, data, num, dim, page_num, page);
    split_data(data, row_entities, dimensions);
    return row_entities.size();
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
        std::cout << "  This query has " << len << " result." << std::endl;
        std::cout << "    First is [id:" << result.ids[0] << "] [distance:" << result.distances[0] << "]" << std::endl;
        std::cout << "    The " << len << "th is [id:" << result.ids[len - 1]
                  << "] [distance:" << result.distances[len - 1] << "]" << std::endl;
    }
}

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              const nlohmann::json& query_json,
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
                                 query_entities, topk, query_json.dump(), topk_result));
    showResult(topk_result);

    assert_status(engine->DropIndex(collection_name));
    for (auto& id_arrays: all_id_arrays) {
        assert_status(engine->Delete(collection_name, id_arrays));
    }

    assert_status(engine->DropCollection(collection_name));
}

void
load_data(const std::string& filename,
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
    if (raw_data.size() == 0)return;
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

