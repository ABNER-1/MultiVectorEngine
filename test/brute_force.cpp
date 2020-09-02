#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <thread>
#include <cmath>
#include <BaseEngine.h>

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

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return (1.0f - res);
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




size_t rows = 1000000;
int nq = 10;
size_t vec_groups = 4;
int topk = 10;
DISTFUNC<float> distfunc;

struct Compare {
    constexpr bool operator()(std::pair<float, size_t> const &a,
                              std::pair<float, size_t> const &b) const noexcept {
        return a.first < b.first;
    }
};

void Read(std::vector<milvus::multivector::RowEntity> &raw_data, std::ifstream &f, size_t thread_num) {
    std::cout.precision(8);
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
        fs[i].open(data_locations[i].c_str(), std::ios::in);
    }
    for (auto i = 0; i < vec_groups; ++ i)
        readers[i] = std::thread(Read, std::ref(raw_data), std::ref(fs[i]), i);
    for (auto &th : readers)
        th.join();
    for (auto &f : fs) {
        f.close();
    }
}

void LoadQuery(std::vector<milvus::multivector::RowEntity> &query) {
    std::ifstream fq("query.dat", std::ios::in);

    for (auto i = 0; i < vec_groups; ++ i) {
        for (auto j = 0; j < nq; ++ j) {
            query[i][j].float_data.resize(dimensions[i]);
            for (auto k = 0; k < dimensions[j]; ++ k) {
                fq >> query[i][j].float_data[k];
            }
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

int main(int argc, char **argv) {
    std::vector<milvus::multivector::RowEntity> raw_data(vec_groups, milvus::multivector::RowEntity(rows, milvus::Entity()));
    LoadRowData(raw_data);
    milvus::TopKQueryResult result;

    distfunc = L2Sqr;
//    distfunc = InnerProduct;

    while (scanf("%d", &nq) != EOF) {
        if (nq <= 0) continue;
        std::vector<milvus::multivector::RowEntity> query_data(vec_groups, milvus::multivector::RowEntity(nq, milvus::Entity()));
        LoadQuery(query_data);
        result.resize(nq);
        DoSearch(raw_data, query_data, result);
        std::vector<milvus::multivector::RowEntity>().swap(query_data);
    }
    return EXIT_SUCCESS;
}

