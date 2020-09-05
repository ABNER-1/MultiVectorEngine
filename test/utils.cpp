#include "utils.h"
//#include <hdf5/serial/hdf5.h>
//#include <hdf5/serial/H5Cpp.h>
#include <hdf5.h>
#include <H5Cpp.h>

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
readArraysFromHdf5(const std::string& file_name, const std::vector<int64_t>& dimensions,
                   std::vector<milvus::multivector::RowEntity>& row_entities,
                   int page_num, int page, const std::string& data_name) {
    std::vector<std::vector<float>> data;
    unsigned num, dim;
    loadDataFromHdf5(file_name, data, num, dim, page_num, page, data_name);
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
showResultL2(const milvus::TopKQueryResult& topk_query_result) {
    std::cout.precision(18);
    std::cout << "There are " << topk_query_result.size() << " query" << std::endl;
    int cnt = 0;
    for (auto& result : topk_query_result) {
        std::cout << "query: " << ++ cnt << std::endl;
        auto show_len = 10 < result.ids.size() ? 3 : result.ids.size();
        for (auto i = 0; i < show_len; ++ i) {
            std::cout << "(" << result.ids[i] << ", " << result.distances[i] << ")";
            if (i == show_len - 1)
                std::cout << std::endl;
            else
                std::cout << " ";
        }
//        std::cout << "  This query has " << len << " result." << std::endl;
//        std::cout << "    First is [id:" << result.ids[0] << "] [distance:" << result.distances[0] << "]" << std::endl;
//        std::cout << "    The " << len << "th is [id:" << result.ids[len - 1]
//                  << "] [distance:" << result.distances[len - 1] << "]" << std::endl;
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
testIndexTypeIP(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
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

    std::string h5_file_name = "/home/abner/vector/glove-200-angular.hdf5";
    auto collection_name = "test_collection25";
    std::vector<int64_t> dim{64, 64, 72};
    std::vector<int64_t> index_file_sizes{1024, 1024, 1024};
    int nq = 100;
    int topk = 100;
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
        vector_num = readArraysFromHdf5("/home/abner/vector/glove-200-angular.hdf5",
                                        dim, row_entities, 10000, page_id);
        if (vector_num == 0) break;
//    generateArrays(vector_num, dim, row_entities);
        std::vector<int64_t> id_arrays;
        generateIds(vector_num, id_arrays);

        assert_status(engine->Insert(collection_name, row_entities, id_arrays));
        std::cout << "Insert " << vector_num << " into milvus." << std::endl;
        all_id_arrays.emplace_back(std::move(id_arrays));
        ++page_id;
    }
    readArraysFromHdf5(h5_file_name, dim, query_entities,
                       10000, 0, "test");
//    readArrays("/data/gist/gist_query.fvecs", dim, row_entities);
//    generateArrays(nq, dim, query_entities);
    assert_status(engine->CreateIndex(collection_name,
                                      index_type, index_json.dump()));

    milvus::TopKQueryResult topk_result;

    assert_status(engine->Search(collection_name, weight,
                                 query_entities, topk, query_json.dump(), topk_result));
//    showResult(topk_result);

    compareResultWithH5(topk_result, h5_file_name);

    assert_status(engine->DropIndex(collection_name));
    for (auto& id_arrays: all_id_arrays) {
        assert_status(engine->Delete(collection_name, id_arrays));
    }

    assert_status(engine->DropCollection(collection_name));
}

void
loadDataFromHdf5(const std::string& filename, std::vector<std::vector<float>>& vector_data,
                 unsigned& num, unsigned& dim, int page_num, int page,
                 const std::string& data_name) {
    using namespace H5;
    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(data_name);
    DataSpace dataspace = dataset.getSpace();
    hsize_t dims_out[2];
    int rank = dataspace.getSimpleExtentDims(dims_out);
    num = dims_out[0];
    dim = dims_out[1];

//    read a page once
    hsize_t dimsm[2] = {static_cast<hsize_t>(page_num), dim};
    DataSpace memspace(rank, dimsm);
    float data[10000][200];

    int read_length = num - page_num * page;
    read_length = std::min(page_num, read_length);
    if (read_length <= 0) return;
    hsize_t offset[2]{static_cast<hsize_t>(page_num * page), 0};
    hsize_t count[2]{static_cast<hsize_t>(read_length), dimsm[1]};
    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    hsize_t offset_out[2]{0, 0};
    memspace.selectHyperslab(H5S_SELECT_SET, count, offset_out);
    dataset.read(data, PredType::NATIVE_FLOAT, memspace, dataspace);

    vector_data.resize(read_length, std::vector<float>(dim));
    for (int i = 0; i < read_length; ++i) {
        for (int j = 0; j < dim; ++j) {
            vector_data[i][j] = data[i][j];
        }
    }
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
compareResultWithH5(const milvus::TopKQueryResult& topk_query_result,
                    const std::string& h5_file_name) {
    std::vector<std::vector<float>> distances, neighbors;
    unsigned num, dim;
    loadDataFromHdf5(h5_file_name, distances, num, dim, 10000, 0, "distances");
    loadDataFromHdf5(h5_file_name, neighbors, num, dim, 10000, 0, "neighbors");
    for (int i = 0; i < topk_query_result.size(); ++i) {
        auto& result = topk_query_result[i];
        for (int j = 0; j < result.ids.size(); ++j) {
            //compare result.ids[j] == neighbors[i][j]
            if (result.ids[j] == neighbors[i][j]) {
                std::cout << "  " << result.distances[j] << " " << distances[i][j] << std::endl;
            }
        }
    }

}


void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              const nlohmann::json& query_json,
              const nlohmann::json& config,
              milvus::MetricType metric_type) {
    using namespace milvus::multivector;
    auto assert_status = [](milvus::Status status) {
        if (!status.ok()) {
            std::cout << " " << static_cast<int>(status.code()) << " " << status.message() << std::endl;
        }
    };

    srand((unsigned)time(nullptr));
    auto collection_name = "l2_test_collection" + std::to_string(random()%100);
    int vec_group_num = config.at("group_num");
    int nq = config.at("nq");
    int topk = config.at("topk");
    int vector_num = config.at("nb");
    std::vector<int64_t> dim;
    std::vector<int64_t> index_file_sizes;
    std::vector<std::string> base_data_locations;
    std::vector<float> weights;
    std::vector<int64_t> acc_dims(vec_group_num, 0);
    dim.clear();index_file_sizes.clear();base_data_locations.clear();weights.clear();
    for (auto i = 0; i < vec_group_num; ++ i) {
        base_data_locations.emplace_back(config.at("base_data_locations")[i]);
        dim.emplace_back(config.at("dimensions")[i]);
        acc_dims[i] = i ? acc_dims[i - 1] + dim[i - 1] : 0;
        weights.emplace_back(config.at("weights")[i]);
        index_file_sizes.push_back(1024);
    }

    assert_status(engine->CreateCollection(collection_name, metric_type, dim, index_file_sizes));

// generate insert data vector
    std::vector<RowEntity> query_entities(nq, RowEntity(vec_group_num, milvus::Entity()));
    std::vector<std::vector<int64_t>> all_id_arrays;
            // generate query vector
//    query_entities.resize(nq);
    std::vector<RowEntity> row_entities(vector_num, RowEntity(vec_group_num, milvus::Entity()));
    for (auto i = 0; i < vec_group_num; ++ i) {
        std::ifstream fin(base_data_locations[i], std::ios::in);
        fin.precision(18);
        for (auto j = 0; j < vector_num; ++ j) {
            row_entities[j][i].float_data.resize(dim[i]);
            for (auto k = 0; k < dim[i]; ++ k) {
                fin >> row_entities[j][i].float_data[k];
            }
        }
        fin.close();
    }
    for (auto i = 0; i < nq; ++ i) {
        for (auto j = 0; j < vec_group_num; ++ j) {
            query_entities[i][j].float_data.resize(dim[j]);
            for (auto k = 0; k < dim[j]; ++ k)
                query_entities[i][j].float_data[k] = row_entities[i][j].float_data[k];
        }
    }
    std::vector<int64_t> ids;
    generateIds(vector_num, ids);
    assert_status(engine->Insert(collection_name, row_entities, ids));
    std::cout << "Insert " << vector_num << " vectors into milvus." << std::endl;
    assert_status(engine->CreateIndex(collection_name, index_type, index_json.dump()));

    milvus::TopKQueryResult topk_result;
    assert_status(engine->Search(collection_name, weights, query_entities, topk, query_json.dump(), topk_result));
    showResultL2(topk_result);

    assert_status(engine->DropIndex(collection_name));
    assert_status(engine->DropCollection(collection_name));
}
