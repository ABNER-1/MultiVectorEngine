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

    NRANode(size_t group_size)
        : id(-1), lb(0.0), ub(0.0), result_flag(false), occurs_time(0), group_flags(group_size, false) {}

    NRANode(int64_t id, float lb, float ub, bool result_flag, size_t group_size)
        : id(id), lb(lb), ub(ub), result_flag(result_flag), occurs_time(0), group_flags(group_size) {}
};

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void*, const void*, const void*);

static float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float*)pVect1)[i] * ((float*)pVect2)[i];
    }
    return (-res);
}

static float
L2Sqr(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    //return *((float *)pVect2);
    size_t qty = *((size_t*)qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        float t = ((float*)pVect1)[i] - ((float*)pVect2)[i];
        res += t * t;
    }
    return (res);
}

struct Compare {
    constexpr bool operator()(std::pair<float, size_t> const& a,
                              std::pair<float, size_t> const& b) const noexcept {
        return a.first < b.first;
    }
};

void
RearrangeEntityArray(const std::vector<std::vector<milvus::Entity>>& entity_array,
                     std::vector<std::vector<milvus::Entity>>& rearranged_queries,
                     size_t num_group);

bool
NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                        milvus::QueryResult& result,
                        size_t nq_id,
                        int64_t TopK);

bool
NoRandomAccessAlgorithmIP(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                          milvus::QueryResult& result,
                          const std::vector<float>& weight,
                          int64_t TopK);

bool
StandardNRAIP(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
              milvus::QueryResult& result,
              const std::vector<float>& weight,
              int64_t TopK);

bool
NRAPerformance(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
               milvus::QueryResult& result,
               const std::vector<float>& weight,
               int64_t TopK);

bool
NoRandomAccessAlgorithmL2(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                          milvus::QueryResult& result,
                          const std::vector<float>& weight,
                          int64_t TopK);

bool
ONRAL2(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
       milvus::QueryResult& result,
       const std::vector<float>& weight,
       int64_t TopK,
       size_t qid);

bool
TAL2(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
     milvus::QueryResult& result,
     const std::vector<float>& weight,
     int64_t TopK);

bool
NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                        milvus::QueryResult& result,
                        const std::vector<float>& weight,
                        int64_t TopK);

} // namespace multivector
} // namespace milvus
