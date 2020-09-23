#include "Utils.h"
#include <random>
#include <cmath>
#include <ctime>

using namespace milvus::multivector;

int topk = 2000;
size_t id_size = 20000;

int main() {
    std::vector<milvus::TopKQueryResult> tqrs(2, milvus::TopKQueryResult(1, milvus::QueryResult()));
    std::vector<float> start_v = {0.4, 0.2};
    std::vector<std::vector<bool>> occurs(2, std::vector<bool>(id_size, false));
    srand((unsigned)time(NULL));
    for (auto i = 0; i < 2; ++ i) {
        tqrs[i][0].ids.resize(id_size);
        tqrs[i][0].distances.resize(id_size);
        for (auto j = topk - 1; j >= 0; -- j) {
            long rid;
            do {
                rid = random() % id_size;
            } while (occurs[i][rid]);
            occurs[i][rid] = true;
            tqrs[i][0].ids[j] = rid;
            start_v[i] += drand48() / 10;
            tqrs[i][0].distances[j] = start_v[i];
        }
    }

    milvus::QueryResult result;
    std::vector<float> weight = {1, 1, 1};
    auto ok_ = NRAPerformance(tqrs, result, weight, 50);

    std::cout << "result is " << (ok_ ? "ok" : "not ok") << ": " << std::endl;
    for (auto i = 0; i < 4; ++ i) {
        std::cout << "id = " << result.ids[i] << ", dis = " << result.distances[i] << std::endl;
    }
}
