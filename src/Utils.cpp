#include <algorithm>
#include <queue>
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
        max_ubs[i] = p_dists[i][topk - 1] * weight[i] + weight[i];// + 1;
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
            /*
            if (!nodes[pos].result_flag && (result_set.size() < TopK || nodes[pos].ub < nodes[result_set.top()].ub)) {
                nodes[pos].result_flag = true;
                result_set.emplace(pos);
                if (result_set.size() > TopK) {
                    nodes[result_set.top()].result_flag = false;
                    result_set.pop();
                }
            }
            */
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
//        std::cout << "topid = " << nodes[result_set.top()].id << ", topdis = " << nodes[result_set.top()].lb << std::endl;
        result.ids[tot_size] = nodes[final_result.top()].id;
        result.distances[tot_size] = nodes[final_result.top()].ub;
//        std::cout << "(" << nodes[final_result.top()].id << ":<" << nodes[final_result.top()].lb << "," << nodes[final_result.top()].ub << ">)";
        final_result.pop();
        tot_size++;
    }
//    std::cout << std::endl;

    return ret;
}

//bool
//NoRandomAccessAlgorithmIP(const std::vector<milvus::TopKQueryResult> &ng_nq_tpk,
//                          milvus::QueryResult &result,
//                          const std::vector<float> &weight,
//                          int64_t TopK) {
//    bool ret = false;
//    auto num_group = ng_nq_tpk.size();
//    auto topk = ng_nq_tpk[0][0].ids.size();
//    std::vector<const int64_t *> p_ids(num_group, 0);
//    std::vector<const float *> p_dists(num_group, 0);
//    std::vector<NRANode> nodes;
//    std::unordered_map<int64_t, size_t> hash_tbl;
//
//    float cur_max_estimate_value = 0.0;
//    auto cmp = [&](size_t i, size_t j) { return nodes[i].lb > nodes[j].lb; };
//    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> result_set(cmp);
//    std::vector<size_t> tmp_queue;
//    for (auto i = 0; i < num_group; ++i) {
//        p_ids[i] = ng_nq_tpk[i][0].ids.data();
//        p_dists[i] = ng_nq_tpk[i][0].distances.data();
//        cur_max_estimate_value += ((*p_dists[i]) * (weight[i]));
//        auto cur_id = *p_ids[i];
//        auto target = hash_tbl.find(cur_id);
//        size_t pos;
//        if (target != hash_tbl.end()) {
//            pos = target->second;
//            nodes[pos].lb += (*p_dists[i] * (weight[i]));
//        } else {
//            pos = nodes.size();
//            hash_tbl[cur_id] = pos;
//            nodes.emplace_back(cur_id, (*p_dists[i] * (weight[i])), 0, false, num_group);
//        }
//        nodes[pos].group_flags[i] = true;
//        ++nodes[pos].occurs_time;
//    }
//    for (auto i = 0; i < nodes.size(); ++i) {
//        nodes[i].ub = cur_max_estimate_value;
//        result_set.emplace(i);
//        nodes[i].result_flag = true;
//        if (result_set.size() > TopK) {
//            nodes[result_set.top()].result_flag = false;
//            result_set.pop();
//        }
//    }
//
//    size_t li = 1;
//    std::vector<int64_t> new_boy;
//    float max_unselected_ub = cur_max_estimate_value;
//    for (; li < topk; ++li) {
//        cur_max_estimate_value = 0.0;
//        for (auto i = 0; i < num_group; ++i) {
//            auto cur_id = p_ids[i][li];
//            auto target = hash_tbl.find(cur_id);
//            cur_max_estimate_value += (p_dists[i][li] * weight[i]);
//            size_t pos;
//            if (target != hash_tbl.end()) {
//                pos = target->second;
//                nodes[pos].lb += (p_dists[i][li] * weight[i]);
//            } else {
//                pos = nodes.size();
//                new_boy.push_back(pos);
//                nodes.emplace_back(cur_id, (p_dists[i][li] * weight[i]), 0, false, num_group);
//                hash_tbl[cur_id] = pos;
//            }
////            // maintain TopK results
////            if (!nodes[pos].result_flag && nodes[pos].id > 0 && (result_set.size() < TopK || nodes[pos].lb > nodes[result_set.top()].lb)) {
////                nodes[pos].result_flag = true;
////                result_set.push(pos);
////                if (result_set.size() > TopK) {
////                    nodes[result_set.top()].result_flag = false;
////                    result_set.pop();
////                }
////            }
////            // maintain previous record ub
////            for (auto j = 0; j < nodes.size() - new_boy.size(); ++j) {
////                if (!nodes[j].group_flags[i]) {
////                    nodes[j].ub -= ((p_dists[i][li - 1] - p_dists[i][li]) * weight[i]);
////                }
////            }
////            nodes[pos].group_flags[i] = true;
////            ++nodes[pos].occurs_time;
//        }
//        for (auto &new_node_id :new_boy)
//            nodes[new_node_id].ub = cur_max_estimate_value;
//        new_boy.clear();
//
//        // find unselected upper bound value
//        max_unselected_ub = std::numeric_limits<float>::min();
//        bool find_flag = false;
//        for (auto &node: nodes) {
//            if (!node.result_flag) {
//                max_unselected_ub = std::max(max_unselected_ub, node.ub);
//                find_flag = true;
//            }
//        }
//        // set max_unselected ub as max value if all in topk
//        if (!find_flag)
//            max_unselected_ub = std::numeric_limits<float>::max();
//
//        // rerank topk
//        for (auto i = 0; i < result_set.size(); ++i) {
//            tmp_queue.push_back(result_set.top());
//            result_set.pop();
//        }
//        for (auto& i: tmp_queue) result_set.push(i);
//        std::vector<size_t>().swap(tmp_queue);
//
//        // judge exit condition
//        if (nodes[result_set.top()].lb >= max_unselected_ub) {
//            ret = true;
//            break;
//        }
//    }
//
//    // organize result set
//    auto tot_size = result_set.size();
//    result.ids.resize(tot_size);
//    result.distances.resize(tot_size);
//    while (!result_set.empty()) {
//        --tot_size;
//        result.ids[tot_size] = nodes[result_set.top()].id;
//        result.distances[tot_size] = nodes[result_set.top()].lb;
//        result_set.pop();
//    }
//    return ret;
//}

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
        auto last_line_id = node_size;
        for (auto& node :nodes) node.ub = node.lb;
        // todo: check it
//        #pragma omp parallel for
        for (int j = 0; j < node_size; ++j) {
            for (int i = 0; i < num_group; ++i) {
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
        if (!find_flag) max_unselected_ub = std::numeric_limits<float>::max();
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
    // judge exit conditiontopk
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
} // namespace multivector
} // namespace milvus
