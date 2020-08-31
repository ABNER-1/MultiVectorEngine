#include <algorithm>
#include <queue>
#include "Utils.h"

namespace milvus {
namespace multivector {


void RearrangeQueryEntityArray(const std::vector<std::vector<milvus::Entity>> &entity_array,
                               std::vector<std::vector<milvus::Entity>> &rearranged_queries,
                               size_t num_group,
                               const std::vector<bool> &failed_nqs) {
    size_t nq = entity_array.size();
    for (auto i = 0; i < nq; ++ i) {
        if (!failed_nqs[i]) continue;
        for (auto j = 0; j < num_group; ++ j) {
            rearranged_queries[j].emplace_back(entity_array[i][j]);
        }
    }
}

bool NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                             milvus::QueryResult &result,
                             size_t nq_id,
                             int64_t TopK) {
    bool ret = false;
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][0].ids.size();
    std::vector<const int64_t *> p_ids(num_group, 0);
    std::vector<const float *> p_dists(num_group, 0);
    std::vector<NRANode> nodes;
    std::unordered_map<int64_t, size_t> hash_tbl;
    float cur_max_estimate_value = 0.0;
    auto cmp = [&](size_t i, size_t j) { return nodes[i].lb > nodes[j].lb; };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
    for (auto i = 0; i < num_group; ++ i) {
        p_ids[i] = ng_nq_tpk[i][nq_id].ids.data();
        p_dists[i] = ng_nq_tpk[i][nq_id].distances.data();
        cur_max_estimate_value += *p_dists[i];
        nodes.emplace_back(*p_ids[i], *p_dists[i], 0, false, num_group);
        nodes[i].group_flags[i] = true;
        hash_tbl[*p_ids[i]] = i;
//        result_set.emplace(i);
    }
    for (auto i = 0; i < num_group; ++ i) {
        nodes[i].ub = cur_max_estimate_value;
        result_set.emplace(i);
        nodes[i].result_flag = true;
        if (result_set.size() > TopK) {
            nodes[result_set.top()].result_flag = false;
            result_set.pop();
        }
    }

    size_t li = 1;
    std::vector<int64_t> new_boy;
    float max_unselected_ub = cur_max_estimate_value;
    for (; li < topk; ++ li) {
        cur_max_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++ i) {
            auto cur_id = p_ids[i][li];
            auto target = hash_tbl.find(cur_id);
            cur_max_estimate_value += p_dists[i][li];
            size_t pos;
            if (target != hash_tbl.end()) {
                pos = target->second;
                nodes[pos].lb += p_dists[i][li];
            } else {
                pos = nodes.size();
                new_boy.push_back(pos);
                nodes.emplace_back(cur_id, p_dists[i][li], 0, false, num_group);
                nodes[pos].group_flags[i] = true;
                hash_tbl[cur_id] = pos;
            }
            if (!nodes[pos].result_flag && nodes[pos].lb > nodes[result_set.top()].lb) {
                nodes[pos].result_flag = true;
                result_set.emplace(pos);
                if (result_set.size() > TopK) {
                    nodes[result_set.top()].result_flag = false;
                    result_set.pop();
                }
            }
            for (auto j = 0; j < nodes.size() - new_boy.size(); ++ j) {
                if (!nodes[j].group_flags[i]) {
                    nodes[j].ub -= (p_dists[i][li - 1] - p_dists[i][li]);
                }
            }
        }
        for (auto i = 0; i < new_boy.size(); ++ i)
            nodes[new_boy[i]].ub = cur_max_estimate_value;
        max_unselected_ub = std::numeric_limits<float>::min();
        bool find_flag = false;
        for (auto i = 0; i < nodes.size(); ++ i) {
            if (!nodes[i].result_flag) {
                max_unselected_ub = std::max(max_unselected_ub, nodes[i].ub);
                find_flag = true;
            }
        }
        if (!find_flag)
            max_unselected_ub = std::numeric_limits<float>::max();
        if (nodes[result_set.top()].lb >= max_unselected_ub) {
            ret = true;
            break;
        }
    }

    auto tot_size = result_set.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    while (!result_set.empty()) {
        tot_size --;
        result.ids[tot_size] = nodes[result_set.top()].id;
        result.distances[tot_size] = nodes[result_set.top()].lb;
        result_set.pop();
    }

    return ret;
}

bool NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                             milvus::QueryResult &result,
                             const std::vector<float>& weight,
                             int64_t TopK) {
    bool ret = false;
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][0].ids.size();
    std::vector<const int64_t *> p_ids(num_group, 0);
    std::vector<const float *> p_dists(num_group, 0);
    std::vector<NRANode> nodes;
    std::unordered_map<int64_t, size_t> hash_tbl;
    float cur_max_estimate_value = 0.0;
    auto cmp = [&](size_t i, size_t j) { return nodes[i].lb > nodes[j].lb; };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
    for (auto i = 0; i < num_group; ++ i) {
        p_ids[i] = ng_nq_tpk[i][0].ids.data();
        p_dists[i] = ng_nq_tpk[i][0].distances.data();
        cur_max_estimate_value += ((*p_dists[i]) * (-weight[i]));
        nodes.emplace_back(*p_ids[i], ((*p_dists[i]) * (-weight[i])), 0, false, num_group);
        nodes[i].group_flags[i] = true;
        hash_tbl[*p_ids[i]] = i;
//        result_set.emplace(i);
    }
    for (auto i = 0; i < num_group; ++ i) {
        nodes[i].ub = cur_max_estimate_value;
        result_set.emplace(i);
        nodes[i].result_flag = true;
        if (result_set.size() > TopK) {
            nodes[result_set.top()].result_flag = false;
            result_set.pop();
        }
    }

    size_t li = 1;
    std::vector<int64_t> new_boy;
    float max_unselected_ub = cur_max_estimate_value;
    for (; li < topk; ++ li) {
        cur_max_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++ i) {
            auto cur_id = p_ids[i][li];
            auto target = hash_tbl.find(cur_id);
            cur_max_estimate_value += (p_dists[i][li] * -weight[i]);
            size_t pos;
            if (target != hash_tbl.end()) {
                pos = target->second;
                nodes[pos].lb += (p_dists[i][li] * -weight[i]);
            } else {
                pos = nodes.size();
                new_boy.push_back(pos);
                nodes.emplace_back(cur_id, (p_dists[i][li] * -weight[i]), 0, false, num_group);
                nodes[pos].group_flags[i] = true;
                hash_tbl[cur_id] = pos;
            }
            if (!nodes[pos].result_flag && nodes[pos].lb > nodes[result_set.top()].lb) {
                nodes[pos].result_flag = true;
                result_set.emplace(pos);
                if (result_set.size() > TopK) {
                    nodes[result_set.top()].result_flag = false;
                    result_set.pop();
                }
            }
            for (auto j = 0; j < nodes.size() - new_boy.size(); ++ j) {
                if (!nodes[j].group_flags[i]) {
                    nodes[j].ub -= ((p_dists[i][li - 1] - p_dists[i][li]) * -weight[i]);
                }
            }
        }
        for (auto i = 0; i < new_boy.size(); ++ i)
            nodes[new_boy[i]].ub = cur_max_estimate_value;
        max_unselected_ub = std::numeric_limits<float>::min();
        bool find_flag = false;
        for (auto i = 0; i < nodes.size(); ++ i) {
            if (!nodes[i].result_flag) {
                max_unselected_ub = std::max(max_unselected_ub, nodes[i].ub);
                find_flag = true;
            }
        }
        if (!find_flag)
            max_unselected_ub = std::numeric_limits<float>::max();
        if (nodes[result_set.top()].lb >= max_unselected_ub) {
            ret = true;
            break;
        }
    }

    auto tot_size = result_set.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    while (!result_set.empty()) {
        tot_size --;
        result.ids[tot_size] = nodes[result_set.top()].id;
        result.distances[tot_size] = -nodes[result_set.top()].lb;
        result_set.pop();
    }

    return ret;
}

void MultipleRecall(std::vector<milvus::TopKQueryResult> &ng_nq_tpk, milvus::TopKQueryResult &result, size_t nq, int64_t TopK) {
    // todo: omp
    for (size_t i = 0; i < nq; ++ i) {
        NoRandomAccessAlgorithm(ng_nq_tpk, result[i], i, TopK);
    }
}

} // namespace multivector
} // namespace milvus
