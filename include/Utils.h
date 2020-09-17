#include "MilvusApi.h"
#include <string>

namespace milvus {
namespace multivector {

struct NRANode {
    int64_t id;
    float lb, ub;
    bool result_flag;
    uint8_t occurs_time;
    std::vector<bool> group_flags;
    NRANode(size_t group_size): id(-1), lb(0.0), ub(0.0), result_flag(false), occurs_time(0), group_flags(group_size, false) {}
    NRANode(int64_t id, float lb, float ub, bool result_flag, size_t group_size):id(id), lb(lb), ub(ub), result_flag(result_flag), occurs_time(0), group_flags(group_size) {}
};

void RearrangeEntityArray(const std::vector<std::vector<milvus::Entity>> &entity_array,
                          std::vector<std::vector<milvus::Entity>> &rearranged_queries,
                          size_t num_group);

bool NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                             milvus::QueryResult &result,
                             size_t nq_id,
                             int64_t TopK);

bool NoRandomAccessAlgorithmIP(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                               milvus::QueryResult &result,
                               const std::vector<float>& weight,
                               int64_t TopK);

bool NRAPerformance(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                    milvus::QueryResult &result,
                    const std::vector<float>& weight,
                    int64_t TopK);

bool NoRandomAccessAlgorithmL2(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                               milvus::QueryResult &result,
                               const std::vector<float>& weight,
                               int64_t TopK);

bool ONRAL2(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                               milvus::QueryResult &result,
                               const std::vector<float>& weight,
                               int64_t TopK);

bool NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                             milvus::QueryResult &result,
                             const std::vector<float>& weight,
                             int64_t TopK);

} // namespace multivector
} // namespace milvus
