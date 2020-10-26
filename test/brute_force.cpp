#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <thread>
#include <cmath>
#include <chrono>
#include <BaseEngine.h>
#include "nlohmann/json.hpp"
#include <omp.h>
#include "utils.h"
#include "Utils.h"

using json = nlohmann::json;

constexpr double eps = 1e-4;

json config;
std::vector<std::string> base_data_locations;
std::vector<std::string> query_data_locations;
std::string query_data_location;
std::vector<size_t> dimensions;
std::vector<size_t> acc_dims;
std::vector<float> weights;
std::string config_file;

size_t rows = 1000000;
int nq = 10;
size_t vec_groups = 4;
int topk = 10;
int precision = 6;
bool use_base_query;

milvus::multivector::DISTFUNC<float> distfunc;

void Read(std::vector<milvus::multivector::RowEntity> &raw_data, std::ifstream &f, size_t thread_num, size_t read_rows) {
//    std::cout.precision(8);
    f.precision(precision);
    for (auto i = 0; i < read_rows; ++ i) {
        raw_data[thread_num][i].float_data.resize(dimensions[thread_num]);
        for (auto j = 0; j < dimensions[thread_num]; ++ j) {
            f >> raw_data[thread_num][i].float_data[j];
        }
    }
}

void LoadRowData(std::vector<milvus::multivector::RowEntity> &raw_data) {
    std::vector<std::thread> readers(vec_groups);
    std::vector<std::ifstream> fs(4);
    for (auto i = 0; i < vec_groups; ++ i) {
        fs[i].open(base_data_locations[i].c_str(), std::ios::in);
    }
    for (auto i = 0; i < vec_groups; ++ i)
        readers[i] = std::thread(Read, std::ref(raw_data), std::ref(fs[i]), i, rows);
    for (auto &th : readers)
        th.join();
    for (auto &f : fs) {
        f.close();
    }
}

void LoadQueryData(std::vector<milvus::multivector::RowEntity> &raw_data) {
    std::vector<std::thread> readers(vec_groups);
    std::vector<std::ifstream> fs(4);
    for (auto i = 0; i < vec_groups; ++ i) {
        fs[i].open(query_data_locations[i].c_str(), std::ios::in);
    }
    for (auto i = 0; i < vec_groups; ++ i)
        readers[i] = std::thread(Read, std::ref(raw_data), std::ref(fs[i]), i, nq);
    for (auto &th : readers)
        th.join();
    for (auto &f : fs) {
        f.close();
    }
}

void GenQueryDataFromBase(const std::vector<milvus::multivector::RowEntity> &base_data) {
    // default pick top nq base data to be query data
    std::ofstream fq(query_data_location, std::ios::out);
    fq.precision(precision);
    for (auto i = 0; i < nq; ++ i) {
        for (auto j = 0; j < vec_groups; ++ j) {
            for (auto k = 0; k < dimensions[j]; ++ k)
                fq << base_data[j][i].float_data[k] << " ";
        }
        fq << std::endl;
    }
    fq.close();
}

void GenQueryDataFromRandom() {
    std::ofstream fq(query_data_location, std::ios::out);
    fq.precision(precision);
    for (auto i = 0; i < nq; ++ i) {
        for (auto j = 0; j < vec_groups; ++ j) {
            for (auto k = 0; k < dimensions[j]; ++ k)
                fq << drand48() << " ";
        }
        fq << std::endl;
    }
    fq.close();
}

void LoadQuery(std::vector<milvus::multivector::RowEntity> &query) {
    std::ifstream fq(query_data_location, std::ios::in);

    fq.precision(precision);
    std::cout.precision(precision);
    std::cout << "check load query data:" << std::endl;
    for (auto i = 0; i < nq; ++ i) {
        for (auto j = 0; j < vec_groups; ++ j) {
            query[j][i].float_data.resize(dimensions[j]);
            for (auto k = 0; k < dimensions[j]; ++ k)
                fq >> query[j][i].float_data[k];
        }
    }
    fq.close();
}

void Search(const std::vector<milvus::multivector::RowEntity> &raw, const std::vector<milvus::multivector::RowEntity> &query, milvus::QueryResult &result, size_t q_id) {

    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, milvus::multivector::Compare> result_set;
    for (auto i = 0; i < rows; ++ i) {
        float dist = 0;
        for (auto j = 0; j < vec_groups; ++ j) {
            float d = distfunc(raw[j][i].float_data.data(), query[j][q_id].float_data.data(), &dimensions[j]);
            dist += d * weights[j];
        }
        result_set.emplace(dist, i);
        if (result_set.size() > topk)
            result_set.pop();
    }
    result.ids.resize(topk);
    result.distances.resize(topk);
    size_t res_num = result_set.size();
    while (!result_set.empty()) {
        res_num --;
        result.ids[res_num] = result_set.top().second;
        result.distances[res_num] = result_set.top().first;
        result_set.pop();
    }
}

void DoSearch(const std::vector<milvus::multivector::RowEntity> &raw, const std::vector<milvus::multivector::RowEntity> &query, milvus::TopKQueryResult &result) {
    std::vector<std::thread> searcher;
    for (auto i = 0; i < nq; ++ i) {
        result[i].ids.resize(topk);
        result[i].distances.resize(topk);
        searcher.emplace_back(std::thread(Search, std::ref(raw), std::ref(query), std::ref(result[i]), i));
    }
    for (auto &th : searcher)
        th.join();
}

bool get_config() {
    std::ifstream f_conf;
    try {
        f_conf.open(config_file);
//    config = json::parse(f_conf);
        f_conf >> config;
//        std::cout << "config.dump(): " << config.dump(4) << std::endl;
//        std::cout << "-----------------------------------------" << std::endl;
        vec_groups = config["group_num"];
        rows = config["nb"];
        nq = config["nq"];
        topk = config["topk"];
        precision = config["precision"];
        if (config["metric_type"] == "IP") {
            distfunc = milvus::multivector::InnerProduct;
            std::cout << "metric_type = " << config["metric_type"] << std::endl;
        } else if (config["metric_type"] == "L2") {
            distfunc = milvus::multivector::L2Sqr;
            std::cout << "metric_type = " << config["metric_type"] << std::endl;
        } else {
            std::cout << "invalid metric_type from config file: " << config["metric_type"] << std::endl;
        }
//        query_data_location = config.at("query_data_locations");
        use_base_query = config.at("use_base_query");
        base_data_locations.clear();
        dimensions.clear();
        acc_dims.resize(vec_groups, 0);
        weights.clear();
        for (auto i = 0; i < vec_groups; ++ i) {
            base_data_locations.emplace_back(config.at("base_data_locations")[i]);
            query_data_locations.emplace_back(config.at("query_data_locations")[i]);
            dimensions.emplace_back(config.at("dimensions")[i]);
            acc_dims[i] = i ? acc_dims[i - 1] + dimensions[i - 1] : 0;
            weights.emplace_back(config.at("weights")[i]);
        }
        f_conf.close();
    } catch (std::exception &e) {
        if (f_conf.is_open())
            f_conf.close();
        std::cout << "error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

void show_config() {
    std::cout << "config: " << std::endl;
    std::cout << "group_num = " << vec_groups << std::endl;
    std::cout << "base data info: " << std::endl;
    for (auto i = 0; i < vec_groups; ++ i) {
        std::cout << "data" << i << "_locations: " << base_data_locations[i] << ", dim: " << dimensions[i] << ", weitht: " << weights[i] << std::endl;
    }
    std::cout << "query data info: " << std::endl;
    for (auto i = 0; i < vec_groups; ++ i) {
        std::cout << "query" << i << "_locations: " << query_data_locations[i] << ", dim: " << dimensions[i] << ", weitht: " << weights[i] << std::endl;
    }
//    std::cout << "query data location: " << query_data_location << std::endl;
    std::cout << "whether use base data as query: " << (use_base_query ? "true" : "false") << std::endl;
    std::cout << "acc_dims: " << std::endl;
    for (auto &d : acc_dims)
        std::cout << d << " ";
    std::cout << std::endl;
    std::cout << "nb = " << rows << ", nq = " << nq << ", topk = " << topk << ", precision = " << precision << std::endl;
}

void show_result(const milvus::TopKQueryResult &res) {
    int correct_cnt = 0;
    int sig = config.at("metric_type") == "IP" ? -1 : 1;
    for (auto i = 0; i < nq; ++ i) {
        std::cout << "top " << topk << " of query " << i + 1 << " :" << std::endl;
//        if (fabs(res[i].distances[0]) < eps)
//            correct_cnt ++;
        for (auto j = 0; j < topk; ++ j) {
            if (res[i].ids[j] == i)
                correct_cnt ++;
            std::cout << "(" << res[i].ids[j] << "," << res[i].distances[j] * sig << ")";
            if (j == topk - 1)
                std::cout << std::endl;
            else
                std::cout << " ";
        }
    }
    std::cout << "correct cnt = " << correct_cnt << "(/" << nq << ")" << std::endl;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        config_file = std::string(argv[1]);
        std::cout << "read config file from " << config_file << std::endl;
    } else {
        std::cout << "invalid argv of program, need one argv of config file!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!get_config()) {
        std::cout << "get config failed from config file: " << config_file << ", please check config file." << std::endl;
        return EXIT_FAILURE;
    }
    std::chrono::high_resolution_clock::time_point ts, te;
    show_config();
//    return EXIT_SUCCESS;
    std::vector<milvus::multivector::RowEntity> raw_data(vec_groups, milvus::multivector::RowEntity(rows, milvus::Entity()));
    ts = std::chrono::high_resolution_clock::now();
    LoadRowData(raw_data);
    te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "LoadRowData costs " << search_duration << " ms." << std::endl;
    std::vector<milvus::multivector::RowEntity> query_data(vec_groups, milvus::multivector::RowEntity(nq, milvus::Entity()));
    ts = std::chrono::high_resolution_clock::now();
    use_base_query ? GenQueryDataFromBase(raw_data) : LoadQueryData(query_data);
//    LoadQuery(query_data);
    te = std::chrono::high_resolution_clock::now();
    search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "LoadQuery costs " << search_duration << " ms." << std::endl;
    milvus::TopKQueryResult result;

    result.resize(nq);
    ts = std::chrono::high_resolution_clock::now();
    DoSearch(raw_data, query_data, result);
    te = std::chrono::high_resolution_clock::now();
    search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "Search costs " << search_duration << " ms." << std::endl;

    // show_result(result);
    std::string baseline_file = config.at("baseline_result");
    writeBenchmarkResult(result, baseline_file, search_duration, topk);

    return EXIT_SUCCESS;
}

