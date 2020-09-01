#include "Utils.h"

using namespace milvus::multivector;


int main() {
    std::vector<milvus::TopKQueryResult> tqrs(3, milvus::TopKQueryResult(1, milvus::QueryResult()));
    // add the first group
    for (auto i = 0; i < 5; ++ i)
        tqrs[0][0].ids.push_back(5 - i);
    tqrs[0][0].distances.push_back(0.1);
    tqrs[0][0].distances.push_back(0.3);
    tqrs[0][0].distances.push_back(0.5);
    tqrs[0][0].distances.push_back(0.8);
    tqrs[0][0].distances.push_back(1);

    // add the second group
    tqrs[1][0].ids.push_back(5);
    tqrs[1][0].ids.push_back(4);
    tqrs[1][0].ids.push_back(1);
    tqrs[1][0].ids.push_back(3);
    tqrs[1][0].ids.push_back(2);
    tqrs[1][0].distances.push_back(0.1);
    tqrs[1][0].distances.push_back(0.2);
    tqrs[1][0].distances.push_back(0.3);
    tqrs[1][0].distances.push_back(0.7);
    tqrs[1][0].distances.push_back(0.8);

    // add the third group
    tqrs[2][0].ids.push_back(2);
    tqrs[2][0].ids.push_back(5);
    tqrs[2][0].ids.push_back(1);
    tqrs[2][0].ids.push_back(3);
    tqrs[2][0].ids.push_back(4);
    tqrs[2][0].distances.push_back(0);
    tqrs[2][0].distances.push_back(0.1);
    tqrs[2][0].distances.push_back(0.2);
    tqrs[2][0].distances.push_back(0.6);
    tqrs[2][0].distances.push_back(0.8);

    for (auto i = 0; i < 3; ++ i) {
        std::cout << "the " << i + 1 << "th group: " << std::endl;
        for (auto j = 0; j < 1; ++ j) {
            for (auto k = 0; k < 5; ++ k)
                std::cout << "id = " << tqrs[i][j].ids[k] << ", dis = " << tqrs[i][j].distances[k] << std::endl;
        }
        std::cout << "------------------------------------------------------------" << std::endl;
    }
    milvus::QueryResult result;
    std::vector<float> weight = {1, 1, 1};
    NoRandomAccessAlgorithm(tqrs, result, weight, 2);

    std::cout << "result: " << std::endl;
    for (auto i = 0; i < 2; ++ i) {
        std::cout << "id = " << result.ids[i] << ", dis = " << result.distances[i] << std::endl;
    }
}
