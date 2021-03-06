#include <algorithm>
#include <queue>
#include <chrono>
#include "Utils.h"

namespace milvus {
namespace multivector {

void
RearrangeEntityArray(const std::vector<std::vector<milvus::Entity>>& entity_array,
                     std::vector<std::vector<milvus::Entity>>& rearranged_queries,
                     size_t num_group) {
    size_t nq = entity_array.size();
    auto new_array = const_cast<std::vector<std::vector<milvus::Entity>>*>(&entity_array);
    for (auto i = 0; i < nq; ++i) {
        for (auto j = 0; j < num_group; ++j) {
            rearranged_queries[j][i].float_data.swap((*new_array)[i][j].float_data);
            rearranged_queries[j][i].binary_data.swap((*new_array)[i][j].binary_data);
        }
    }
}

// abandoned
bool
NoRandomAccessAlgorithm(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                        milvus::QueryResult& result,
                        const std::vector<float>& weight,
                        int64_t TopK) {
    bool ret = false;
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][0].ids.size();
    std::vector<const int64_t*> p_ids(num_group, 0);
    std::vector<const float*> p_dists(num_group, 0);
    std::vector<NRANode> nodes;
    std::unordered_map<int64_t, size_t> hash_tbl;
    float cur_max_estimate_value = 0.0;
    auto cmp = [&](size_t i, size_t j) {
        return nodes[i].occurs_time == nodes[j].occurs_time ? nodes[i].lb > nodes[j].lb : nodes[i].occurs_time
                                                                                          > nodes[j].occurs_time;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
    std::vector<size_t> tmp_queue;
    for (auto i = 0; i < num_group; ++i) {
        p_ids[i] = ng_nq_tpk[i][0].ids.data();
        p_dists[i] = ng_nq_tpk[i][0].distances.data();
        cur_max_estimate_value += ((*p_dists[i]) * (-weight[i]));
        auto cur_id = *p_ids[i];
        auto target = hash_tbl.find(cur_id);
        size_t pos;
        if (target != hash_tbl.end()) {
            pos = target->second;
            nodes[pos].lb += (*p_dists[i] * (-weight[i]));
        } else {
            pos = nodes.size();
            hash_tbl[cur_id] = pos;
            nodes.emplace_back(cur_id, (*p_dists[i] * (-weight[i])), 0, false, num_group);
        }
//        nodes.emplace_back(*p_ids[i], ((*p_dists[i]) * (-weight[i])), 0, false, num_group);
        nodes[pos].group_flags[i] = true;
        nodes[pos].occurs_time++;
//        hash_tbl[*p_ids[i]] = i;
//        result_set.emplace(i);
    }
    for (auto i = 0; i < nodes.size(); ++i) {
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
    for (; li < topk; ++li) {
        cur_max_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++i) {
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
                hash_tbl[cur_id] = pos;
            }
            if (!nodes[pos].result_flag && (result_set.size() < TopK || nodes[pos].lb > nodes[result_set.top()].lb)) {
                nodes[pos].result_flag = true;
                result_set.emplace(pos);
                if (result_set.size() > TopK) {
                    nodes[result_set.top()].result_flag = false;
                    result_set.pop();
                }
            }
            for (auto j = 0; j < nodes.size() - new_boy.size(); ++j) {
                if (!nodes[j].group_flags[i]) {
                    nodes[j].ub -= ((p_dists[i][li - 1] - p_dists[i][li]) * -weight[i]);
                }
            }
            nodes[pos].group_flags[i] = true;
            nodes[pos].occurs_time++;
        }
        for (auto i = 0; i < new_boy.size(); ++i)
            nodes[new_boy[i]].ub = cur_max_estimate_value;
        new_boy.clear();
        max_unselected_ub = std::numeric_limits<float>::min();
        bool find_flag = false;
        for (auto i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].result_flag) {
                max_unselected_ub = std::max(max_unselected_ub, nodes[i].ub);
                find_flag = true;
            }
        }
        for (auto i = 0; i < result_set.size(); ++i) {
            tmp_queue.push_back(result_set.top());
            result_set.pop();
        }
        for (auto i = 0; i < tmp_queue.size(); ++i)
            result_set.emplace(tmp_queue[i]);
        std::vector<size_t>().swap(tmp_queue);
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
        tot_size--;
//        std::cout << "topid = " << nodes[result_set.top()].id << ", topdis = " << nodes[result_set.top()].lb << std::endl;
        result.ids[tot_size] = nodes[result_set.top()].id;
        result.distances[tot_size] = -nodes[result_set.top()].lb;
        result_set.pop();
    }

    return ret;
}

bool
NoRandomAccessAlgorithmL2(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                          milvus::QueryResult& result,
                          const std::vector<float>& weight,
                          int64_t TopK) {
    bool ret = false;
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][0].ids.size();
    std::vector<const int64_t*> p_ids(num_group, 0);
    std::vector<const float*> p_dists(num_group, 0);
    std::vector<NRANode> nodes;
    std::unordered_map<int64_t, size_t> hash_tbl;
    float cur_min_estimate_value = 0.0;
    auto cmp = [&](size_t i, size_t j) {
        return nodes[i].ub < nodes[j].ub;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
    std::vector<float> max_ubs(num_group + 1, 0.0);
    for (auto i = 0; i < num_group; ++i) {
        p_ids[i] = ng_nq_tpk[i][0].ids.data();
        p_dists[i] = ng_nq_tpk[i][0].distances.data();
        cur_min_estimate_value += ((*p_dists[i]) * weight[i]);
        max_ubs[i] = weight[i] * 2;
        max_ubs[num_group] += max_ubs[i];
        auto cur_id = *p_ids[i];
        auto target = hash_tbl.find(cur_id);
        size_t pos;
        if (target != hash_tbl.end()) {
            pos = target->second;
        } else {
            pos = nodes.size();
            hash_tbl[cur_id] = pos;
            nodes.emplace_back(cur_id, 0, 0, false, num_group);
        }
        nodes[pos].group_flags[i] = true;
        nodes[pos].ub += (*p_dists[i] * weight[i] - max_ubs[i]);
    }
    for (auto i = 0; i < nodes.size(); ++i) {
        nodes[i].lb = cur_min_estimate_value;
        nodes[i].ub += max_ubs[num_group];
        result_set.emplace(i);
        nodes[i].result_flag = true;
        if (result_set.size() > TopK) {
            nodes[result_set.top()].result_flag = false;
            result_set.pop();
        }
    }

    size_t li = 1;
    std::vector<int64_t> new_boy;
    float min_unselected_lb = cur_min_estimate_value;
    for (; li < topk; ++li) {
        cur_min_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++i) {
            auto cur_id = p_ids[i][li];
            auto target = hash_tbl.find(cur_id);
            cur_min_estimate_value += (p_dists[i][li] * weight[i]);
            size_t pos;
            if (target != hash_tbl.end()) {
                pos = target->second;
            } else {
                pos = nodes.size();
                new_boy.push_back(pos);
                nodes.emplace_back(cur_id, 0, 0, false, num_group);
                hash_tbl[cur_id] = pos;
            }
            nodes[pos].ub += (p_dists[i][li] * weight[i] - max_ubs[i]);
            for (auto j = 0; j < nodes.size() - new_boy.size(); ++j) {
                if (!nodes[j].group_flags[i]) {
                    nodes[j].lb += ((p_dists[i][li] - p_dists[i][li - 1]) * weight[i]);
                }
            }
            nodes[pos].group_flags[i] = true;
        }
        for (auto i = 0; i < new_boy.size(); ++i)
            nodes[new_boy[i]].ub += max_ubs[num_group], nodes[new_boy[i]].lb = cur_min_estimate_value;
        new_boy.clear();
        min_unselected_lb = std::numeric_limits<float>::max();
        bool find_flag = false;
        while (!result_set.empty()) {
            nodes[result_set.top()].result_flag = false;
            result_set.pop();
        }
        for (auto i = 0; i < nodes.size(); ++i) {
            if (result_set.size() < TopK || nodes[i].ub < nodes[result_set.top()].ub) {
                if (nodes[i].id < 0)
                    continue;
                nodes[i].result_flag = true;
                result_set.emplace(i);
                if (result_set.size() > TopK) {
                    nodes[result_set.top()].result_flag = false;
                    result_set.pop();
                }
            }
        }
        for (auto i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].result_flag) {
                min_unselected_lb = std::min(min_unselected_lb, nodes[i].lb);
                find_flag = true;
            }
        }
        if (!find_flag)
            min_unselected_lb = std::numeric_limits<float>::min();
        if (nodes[result_set.top()].ub <= min_unselected_lb) {
            ret = true;
            break;
        }
    }

    auto cmp2 = [&](size_t i, size_t j) {
        return nodes[i].ub > nodes[j].ub;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp2)> final_result(cmp2);
    while (!result_set.empty()) {
        final_result.emplace(result_set.top());
        result_set.pop();
    }
    auto tot_size = final_result.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    tot_size = 0;
    while (!final_result.empty()) {
        result.ids[tot_size] = nodes[final_result.top()].id;
        result.distances[tot_size] = nodes[final_result.top()].ub;
        final_result.pop();
        tot_size++;
    }

    return ret;
}

bool
ONRAL2(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                          milvus::QueryResult& result,
                          const std::vector<float>& weight,
                          int64_t TopK,
                          size_t qid) {
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][qid].ids.size();
    std::vector<const int64_t*> p_ids(num_group, 0);
    std::vector<const float*> p_dists(num_group, 0);
    std::vector<NRANode> nodes;
    std::unordered_map<int64_t, size_t> hash_tbl;
    float cur_min_estimate_value = 0.0;
    auto cmp = [&](size_t i, size_t j) {
        return nodes[i].ub < nodes[j].ub;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
    for (auto i = 0; i < num_group; ++i) {
        p_ids[i] = ng_nq_tpk[i][qid].ids.data();
        p_dists[i] = ng_nq_tpk[i][qid].distances.data();
    }

    size_t li = 0;
    std::vector<int64_t> new_boy;
    for (; li < topk; ++li) {
        cur_min_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++i) {
            auto cur_id = p_ids[i][li];
            auto target = hash_tbl.find(cur_id);
            cur_min_estimate_value += (p_dists[i][li] * weight[i]);
            size_t pos;
            if (target != hash_tbl.end()) {
                pos = target->second;
            } else {
                pos = nodes.size();
                new_boy.push_back(pos);
                nodes.emplace_back(cur_id, 0, 0, false, num_group);
//                hash_tbl[cur_id] = pos;
                hash_tbl.emplace(cur_id, pos);
            }
            nodes[pos].lb += (p_dists[i][li] * weight[i]);
            nodes[pos].group_flags[i] = true;
        }
    }
    size_t ub_id = topk - 1;
    for (auto i = 0; i < nodes.size(); ++ i) {
        nodes[i].ub = nodes[i].lb;
        for (auto j = 0; j < num_group; ++ j) {
            if (!nodes[i].group_flags[j])
                nodes[i].ub += weight[j] * 2;
        }
        if (result_set.size() < TopK || nodes[result_set.top()].ub > nodes[i].ub) {
            result_set.emplace(i);
            nodes[i].result_flag = true;
            if (result_set.size() > TopK) {
                nodes[result_set.top()].result_flag = false;
                result_set.pop();
            }
        }
    }

//    std::cout << "show all nodes:" << std::endl;
//    for (auto &node: nodes) {
//        std::cout << "id = " << node.id << ", lb = " << node.lb << ", ub = " << node.ub << std::endl;
//    }

    float min_unselected_lb = std::numeric_limits<float>::max();
    bool find_mul = false;
    for (auto &node : nodes) {
        if (!node.result_flag)
            min_unselected_lb = std::min(min_unselected_lb, node.lb), find_mul = true;
    }
    if (!find_mul)
        min_unselected_lb = std::numeric_limits<float>::min();
    bool ret = min_unselected_lb >= nodes[result_set.top()].ub;

    auto tot_size = result_set.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    tot_size --;
    while (!result_set.empty()) {
        result.ids[tot_size] = nodes[result_set.top()].id;
        result.distances[tot_size] = nodes[result_set.top()].ub;
        result_set.pop();
        tot_size --;
    }

    return ret;
}

//TA
bool
TAL2(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                          milvus::QueryResult &result,
                          const std::vector<float> &weight,
                          int64_t TopK) {
    bool ret = false;
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][0].ids.size();
    std::vector<const int64_t *> p_ids(num_group, 0);
    std::vector<const float *> p_dists(num_group, 0);
    std::unordered_map<size_t, std::pair<float, bool>> hash_tbl;
    auto cmp = [&](std::pair<size_t, float> &a, std::pair<size_t, float> &b) { return a.second < b.second; };
    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> result_set(cmp);
    std::vector<float> sum_row(topk, 0.0);

    for (auto i = 0; i < num_group; ++i) {
        p_ids[i] = ng_nq_tpk[i][0].ids.data();
        p_dists[i] = ng_nq_tpk[i][0].distances.data();
    }
    for (auto i = 0; i < topk; ++ i) {
        for (auto j = 0; j < num_group; ++ j) {
            if (p_ids[j][i] < 0)
                continue;
            sum_row[i] += p_dists[j][i] * weight[j];
            auto target = hash_tbl.find(p_ids[j][i]);
            if (target != hash_tbl.end()) {
                target->second.first += p_dists[j][i] * weight[j];
            } else {
                hash_tbl[p_ids[j][i]] = std::make_pair(p_dists[j][i] * weight[j], false);
            }
        }
    }

    for (auto i = 0; i < topk; ++ i) {
        for (auto j = 0; j < num_group; ++ j) {
            if (p_ids[j][i] < 0)
                continue;
            auto target = hash_tbl.find(p_ids[j][i]);
            if (!target->second.second) {
                if (result_set.size() < TopK || target->second.first < result_set.top().second)
                target->second.second = true;
                result_set.emplace(target->first, target->second.first);
                if (result_set.size() > TopK) {
                    auto out = hash_tbl.find(result_set.top().first);
                    out->second.second = false;
                    result_set.pop();
                }
            }
        }
        if (sum_row[i] > result_set.top().second) {
            ret = true;
            break;
        }
    }
    auto tot_size = result_set.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    tot_size = 0;
    while (!result_set.empty()) {
        result.ids[tot_size] = result_set.top().first;
        result.distances[tot_size] = result_set.top().second;
        result_set.pop();
        tot_size ++;
    }
    return ret;
}


bool
NoRandomAccessAlgorithmIP(const std::vector<milvus::TopKQueryResult>& ng_nq_tpk,
                          milvus::QueryResult& result,
                          const std::vector<float>& weight,
                          int64_t TopK) {
    bool ret = false;
    auto num_group = ng_nq_tpk.size();
    auto topk = ng_nq_tpk[0][0].ids.size();
    std::vector<const int64_t*> p_ids(num_group, 0);
    std::vector<const float*> p_dists(num_group, 0);
    std::vector<NRANode> nodes;
    std::unordered_map<int64_t, size_t> hash_tbl;

    float cur_max_estimate_value = 0.0;
    auto cmp = [&](size_t i, size_t j) {
        return nodes[i].lb > nodes[j].lb;
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
    std::vector<size_t> new_nodes;
    for(auto i = 0; i< num_group; ++i) {
        p_ids[i] = ng_nq_tpk[i][0].ids.data();
        p_dists[i] = ng_nq_tpk[i][0].distances.data();
    }

    // todo: can be parallel with omp
    for (auto line = 0; line < topk; ++line) {
        cur_max_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++i) {
            auto cur_id = p_ids[i][line];
            auto target = hash_tbl.find(cur_id);
            cur_max_estimate_value += (p_dists[i][line] * weight[i]);
            size_t pos;
            if (target != hash_tbl.end()) {
                pos = target->second;
                nodes[pos].lb += (p_dists[i][line] * weight[i]);
            } else {
                pos = nodes.size();
                new_nodes.push_back(pos);
                nodes.emplace_back(cur_id, (p_dists[i][line] * weight[i]), 0, false, num_group);
                hash_tbl[cur_id] = pos;
            }
            nodes[pos].group_flags[i] = true;
        }
        for (auto& new_node_id :new_nodes)
            nodes[new_node_id].ub = cur_max_estimate_value;
        new_nodes.clear();
    }
    auto maintainUb = [&]() {
        // maintain previous record ub
        auto node_size = nodes.size();
        auto last_line_id = topk;
        for (auto& node :nodes) node.ub = node.lb;
        // todo: check it
        #pragma omp parallel for
        for (int j = 0; j < node_size; ++j) {
            for (auto i = 0; i < num_group; ++i) {
                if (nodes[j].group_flags[i]) continue;
                nodes[j].ub += p_dists[i][last_line_id - 1] * weight[i];
            }
        }
    };
    auto findOtherUb = [&]() {
        // find unselected upper bound value
        float max_unselected_ub = std::numeric_limits<float>::min();
        bool find_flag = false;
        for (auto& node: nodes) {
            if (!node.result_flag) {
                max_unselected_ub = std::max(max_unselected_ub, node.ub);
                find_flag = true;
            }
        }
        if (!find_flag)
            max_unselected_ub = std::numeric_limits<float>::max();
        return max_unselected_ub;
    };
    auto rankTopk = [&]() {
        for (auto i = 0; i < nodes.size(); ++i) {
            result_set.emplace(i);
            nodes[i].result_flag = true;
            if (result_set.size() > TopK) {
                nodes[result_set.top()].result_flag = false;
                result_set.pop();
            }
        }
    };
    // judge exit condition
    maintainUb();
    rankTopk();
    ret = (nodes[result_set.top()].lb >= findOtherUb());

    // organize result set
    auto tot_size = result_set.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    while (!result_set.empty()) {
        --tot_size;
        result.ids[tot_size] = nodes[result_set.top()].id;
        result.distances[tot_size] = nodes[result_set.top()].lb;
        result_set.pop();
    }
    return ret;
}

bool
NRAPerformance(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
                          milvus::QueryResult &result,
                          const std::vector<float> &weight,
                          int64_t TopK) {
    std::chrono::high_resolution_clock::time_point ts, te, t0, t1, tl0, tl1;
    ts = std::chrono::high_resolution_clock::now();
    long round1_duration, heap_duration, hash_duration;
    round1_duration = heap_duration = hash_duration = 0;
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
    std::vector<size_t> tmp_queue;
    tl0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < num_group; ++i) {
        p_ids[i] = ng_nq_tpk[i][0].ids.data();
        p_dists[i] = ng_nq_tpk[i][0].distances.data();
        cur_max_estimate_value += ((*p_dists[i]) * (weight[i]));
        auto cur_id = *p_ids[i];
        t0 = std::chrono::high_resolution_clock::now();
        auto target = hash_tbl.find(cur_id);
        t1 = std::chrono::high_resolution_clock::now();
        hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        size_t pos;
        if (target != hash_tbl.end()) {
            pos = target->second;
            nodes[pos].lb += (*p_dists[i] * (weight[i]));
        } else {
            pos = nodes.size();
            t0 = std::chrono::high_resolution_clock::now();
            hash_tbl[cur_id] = pos;
            t1 = std::chrono::high_resolution_clock::now();
            hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            nodes.emplace_back(cur_id, (*p_dists[i] * (weight[i])), 0, false, num_group);
        }
        nodes[pos].group_flags[i] = true;
        ++nodes[pos].occurs_time;
    }
    for (auto i = 0; i < nodes.size(); ++i) {
        nodes[i].ub = cur_max_estimate_value;
        t0 = std::chrono::high_resolution_clock::now();
        result_set.emplace(i);
        t1 = std::chrono::high_resolution_clock::now();
        hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        nodes[i].result_flag = true;
        if (result_set.size() > TopK) {
            nodes[result_set.top()].result_flag = false;
            t0 = std::chrono::high_resolution_clock::now();
            result_set.pop();
            t1 = std::chrono::high_resolution_clock::now();
            hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
    }
    tl1 = std::chrono::high_resolution_clock::now();
    round1_duration = std::chrono::duration_cast<std::chrono::microseconds>(tl1 - tl0).count();

    size_t li = 1;
    std::vector<int64_t> new_boy;
    float max_unselected_ub = cur_max_estimate_value;
    tl0 = std::chrono::high_resolution_clock::now();
    for (; li < topk; ++li) {
        cur_max_estimate_value = 0.0;
        for (auto i = 0; i < num_group; ++i) {
            auto cur_id = p_ids[i][li];
            t0 = std::chrono::high_resolution_clock::now();
            auto target = hash_tbl.find(cur_id);
            t1 = std::chrono::high_resolution_clock::now();
            hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            cur_max_estimate_value += (p_dists[i][li] * weight[i]);
            size_t pos;
            if (target != hash_tbl.end()) {
                pos = target->second;
                nodes[pos].lb += (p_dists[i][li] * weight[i]);
            } else {
                pos = nodes.size();
                new_boy.push_back(pos);
                nodes.emplace_back(cur_id, (p_dists[i][li] * weight[i]), 0, false, num_group);
                t0 = std::chrono::high_resolution_clock::now();
                hash_tbl[cur_id] = pos;
                t1 = std::chrono::high_resolution_clock::now();
                hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            }
            // maintain TopK results
            if (!nodes[pos].result_flag && nodes[pos].id > 0 && (result_set.size() < TopK || nodes[pos].lb > nodes[result_set.top()].lb)) {
                nodes[pos].result_flag = true;
                t0 = std::chrono::high_resolution_clock::now();
                result_set.push(pos);
                t1 = std::chrono::high_resolution_clock::now();
                hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                if (result_set.size() > TopK) {
                    nodes[result_set.top()].result_flag = false;
                    t0 = std::chrono::high_resolution_clock::now();
                    result_set.pop();
                    t1 = std::chrono::high_resolution_clock::now();
                    hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                }
            }
            // maintain previous record ub
            for (auto j = 0; j < nodes.size() - new_boy.size(); ++j) {
                if (!nodes[j].group_flags[i]) {
                    nodes[j].ub -= ((p_dists[i][li - 1] - p_dists[i][li]) * weight[i]);
                }
            }
            nodes[pos].group_flags[i] = true;
            ++nodes[pos].occurs_time;
        }
        for (auto &new_node_id :new_boy)
            nodes[new_node_id].ub = cur_max_estimate_value;
        new_boy.clear();

        // find unselected upper bound value
        max_unselected_ub = std::numeric_limits<float>::min();
        bool find_flag = false;
        for (auto &node: nodes) {
            if (!node.result_flag) {
                max_unselected_ub = std::max(max_unselected_ub, node.ub);
                find_flag = true;
            }
        }
        // set max_unselected ub as max value if all in topk
        if (!find_flag)
            max_unselected_ub = std::numeric_limits<float>::max();

        // rerank topk
        for (auto i = 0; i < result_set.size(); ++i) {
            tmp_queue.push_back(result_set.top());
            t0 = std::chrono::high_resolution_clock::now();
            result_set.pop();
            t1 = std::chrono::high_resolution_clock::now();
            hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
        for (auto& i: tmp_queue) {
            t0 = std::chrono::high_resolution_clock::now();
            result_set.push(i);
            t1 = std::chrono::high_resolution_clock::now();
            hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
        std::vector<size_t>().swap(tmp_queue);

        // judge exit condition
        if (nodes[result_set.top()].lb >= max_unselected_ub) {
            ret = true;
            break;
        }
    }

    std::cout << "loop i = " << li << std::endl;
    tl1 = std::chrono::high_resolution_clock::now();
    auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(tl1 - tl0).count();
    std::cout << "other loop costs: " << loop_duration << " ms." << std::endl;

    // organize result set
    t0 = std::chrono::high_resolution_clock::now();
    auto tot_size = result_set.size();
    result.ids.resize(tot_size);
    result.distances.resize(tot_size);
    while (!result_set.empty()) {
        --tot_size;
        result.ids[tot_size] = nodes[result_set.top()].id;
        result.distances[tot_size] = nodes[result_set.top()].lb;
        t0 = std::chrono::high_resolution_clock::now();
        result_set.pop();
        t1 = std::chrono::high_resolution_clock::now();
        hash_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    te = std::chrono::high_resolution_clock::now();
    std::cout << "organize result set costs: " << std::chrono::duration_cast<std::chrono::microseconds>(te - t0).count() << " ms." << std::endl;
    std::cout << "hash costs: " << hash_duration << " ms." << std::endl;
    std::cout << "heap costs: " << heap_duration << " ms." << std::endl;
    std::cout << "nra total costs: " << std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count() << " ms." << std::endl;
    return ret;
}
} // namespace multivector
} // namespace milvus
