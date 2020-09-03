#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <thread>
#include <cmath>
#include <BaseEngine.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

constexpr double eps = 1e-4;
/*
std::vector<std::string> data_locations = {
    "/home/zilliz/workspace/data/data1.dat",
    "/home/zilliz/workspace/data/data2.dat",
    "/home/zilliz/workspace/data/data3.dat",
    "/home/zilliz/workspace/data/data4.dat"
};

std::vector<size_t> dimensions = {
    64,
    128,
    256,
    512
};

std::vector<size_t> acc_dims = {
    0,
    64,
    192,
    448,
};

std::vector<float> weights = {
    0.1,
    0.2,
    0.3,
    0.4
};
*/

json config;
std::vector<std::string> base_data_locations;
std::string query_data_location;
std::vector<size_t> dimensions;
std::vector<size_t> acc_dims;
std::vector<float> weights;
std::string config_file;

size_t rows = 1000000;
int nq = 10;
size_t vec_groups = 4;
int topk = 10;
bool use_base_query;


template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return (-res);
}

static float
L2Sqr(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    //return *((float *)pVect2);
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        float t = ((float *) pVect1)[i] - ((float *) pVect2)[i];
        res += t * t;
    }
    return (res);
}

DISTFUNC<float> distfunc;

struct Compare {
    constexpr bool operator()(std::pair<float, size_t> const &a,
                              std::pair<float, size_t> const &b) const noexcept {
        return a.first < b.first;
    }
};

void Read(std::vector<milvus::multivector::RowEntity> &raw_data, std::ifstream &f, size_t thread_num) {
//    std::cout.precision(8);
    f.precision(18);
    for (auto i = 0; i < rows; ++ i) {
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
        readers[i] = std::thread(Read, std::ref(raw_data), std::ref(fs[i]), i);
    for (auto &th : readers)
        th.join();
    for (auto &f : fs) {
        f.close();
    }
}

void GenQueryDataFromBase(const std::vector<milvus::multivector::RowEntity> &base_data) {
    // default pick top nq base data to be query data
    std::ofstream fq(query_data_location, std::ios::out);
    fq.precision(18);
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
    fq.precision(18);
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

    fq.precision(18);
    std::cout.precision(18);
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

    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, Compare> result_set;
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
        if (config["metric_type"] == "IP") {
            distfunc = InnerProduct;
            std::cout << "metric_type = " << config["metric_type"] << std::endl;
        } else if (config["metric_type"] == "L2") {
            distfunc = L2Sqr;
            std::cout << "metric_type = " << config["metric_type"] << std::endl;
        } else {
            std::cout << "invalid metric_type from config file: " << config["metric_type"] << std::endl;
        }
        query_data_location = config.at("query_data_location");
        use_base_query = config.at("use_base_query");
        base_data_locations.clear();
        dimensions.clear();
        acc_dims.resize(vec_groups, 0);
        weights.clear();
        for (auto i = 0; i < vec_groups; ++ i) {
            base_data_locations.emplace_back(config.at("base_data_locations")[i]);
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
    std::cout << "query data location: " << query_data_location << std::endl;
    std::cout << "whether use base data as query: " << (use_base_query ? "true" : "false") << std::endl;
    std::cout << "acc_dims: " << std::endl;
    for (auto &d : acc_dims)
        std::cout << d << " ";
    std::cout << std::endl;
    std::cout << "nb = " << rows << ", nq = " << nq << ", topk = " << topk << std::endl;
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
    show_config();
//    return EXIT_SUCCESS;
    std::vector<milvus::multivector::RowEntity> raw_data(vec_groups, milvus::multivector::RowEntity(rows, milvus::Entity()));
    LoadRowData(raw_data);
    use_base_query ? GenQueryDataFromBase(raw_data) : GenQueryDataFromRandom();
    milvus::TopKQueryResult result;

    std::vector<milvus::multivector::RowEntity> query_data(vec_groups, milvus::multivector::RowEntity(nq, milvus::Entity()));
    LoadQuery(query_data);
    result.resize(nq);
    DoSearch(raw_data, query_data, result);

    show_result(result);

//    while (scanf("%d", &nq) != EOF) {
//        if (nq <= 0) continue;
//        std::vector<milvus::multivector::RowEntity> query_data(vec_groups, milvus::multivector::RowEntity(nq, milvus::Entity()));
//        LoadQuery(query_data);
//        result.resize(nq);
//        DoSearch(raw_data, query_data, result);
//        std::vector<milvus::multivector::RowEntity>().swap(query_data);
//    }
    return EXIT_SUCCESS;
}
