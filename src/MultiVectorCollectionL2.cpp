#include <chrono>
#include <nlohmann/json.hpp>
#include "Utils.h"
#include <omp.h>
#include <fstream>
#include <set>
#include <queue>
#include "MultiVectorCollectionL2.h"


namespace milvus {
namespace multivector {

Status
MultiVectorCollectionL2::CreateCollection(const std::vector<int64_t> &dimensions,
                                          const std::vector<int64_t> &index_file_sizes) {
    milvus::CollectionParam cp;
    cp.metric_type = metric_type_;
    for (auto i = 0; i < dimensions.size(); ++ i) {
        child_collection_names_.emplace_back(GenerateChildCollectionName(i));
        cp.collection_name = child_collection_names_[i];
        cp.dimension = dimensions[i];
        cp.index_file_size = index_file_sizes[i];
        auto status = conn_ptr_->CreateCollection(cp);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionL2::HasCollection() {
    bool has_collection = false;
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        has_collection = conn_ptr_->HasCollection(child_collection_names_[i]);
    }
    if (has_collection)
        return Status::OK();
    else
        return Status(StatusCode::UnknownError, "has no collection");
}

Status
MultiVectorCollectionL2::DropCollection() {
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        auto status = conn_ptr_->DropCollection(child_collection_names_[i]);
        if (!status.ok())
            return status;
    }
    return Status::OK();

}

Status
MultiVectorCollectionL2::Insert(const std::vector<milvus::multivector::RowEntity> &entity_arrays,
                                std::vector<int64_t> &id_arrays) {
    std::vector<std::vector<milvus::Entity>> rearranged_arrays(child_collection_names_.size(), std::vector<milvus::Entity>(entity_arrays.size(), milvus::Entity()));
    RearrangeEntityArray(entity_arrays, rearranged_arrays, child_collection_names_.size());
    for (auto i = 0; i < rearranged_arrays.size(); ++ i) {
        auto status = conn_ptr_->Insert(child_collection_names_[i], "", rearranged_arrays[i], id_arrays);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionL2::Delete(const std::vector<int64_t> &id_arrays) {
    for (auto j = 0; j < child_collection_names_.size(); ++ j) {
        auto status = conn_ptr_->DeleteEntityByID(child_collection_names_[j], id_arrays);
        if (!status.ok()) {
            return status;
        }
    }
    conn_ptr_->Flush(child_collection_names_);
    return Status::OK();
}

Status
MultiVectorCollectionL2::CreateIndex(milvus::IndexType index_type, const std::string &extra_params) {
    milvus::IndexParam ip;
    ip.index_type = index_type;
    ip.extra_params = extra_params;
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        ip.collection_name = child_collection_names_[i];
        auto status = conn_ptr_->CreateIndex(ip);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionL2::DropIndex() {
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        auto status = conn_ptr_->DropIndex(child_collection_names_[i]);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionL2::Flush() {
    return conn_ptr_->Flush(child_collection_names_);
    /*
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        auto status = conn_ptr_->Flush(child_collection_names_[i]);
        if (!status.ok())
            return status;
    }
    return Status::OK();
    */
}

bool save_ = false;

Status
MultiVectorCollectionL2::SearchImpl(const std::vector<float>& weight,
                                    const std::vector<milvus::Entity>& entity_query,
                                    int64_t topk,
                                    const std::string& extra_params,
                                    QueryResult &query_results,
                                    int64_t tpk,
                                    size_t qid) {

    static int cnt = 0;
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
//    std::vector<TopKQueryResult> tqrs;
//    tqrs.resize(child_collection_names_.size());
    std::vector<std::string> partition_tags;
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        std::vector<milvus::Entity> container;
        container.emplace_back(entity_query[i]);
        auto status = conn_ptr_->Search(child_collection_names_[i], partition_tags, container, tpk, extra_params, tqrs[i]);
        if (!status.ok())
            return status;
        std::vector<milvus::Entity>().swap(container);
    }
    auto mx_size = tqrs[0][0].ids.size();
    for (auto i = 1; i < child_collection_names_.size(); ++ i) {
        mx_size = mx_size < tqrs[i][0].ids.size() ? tqrs[i][0].ids.size() : mx_size;
    }
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        if (tqrs[i][0].ids.size() < mx_size) {
            tqrs[i][0].ids.resize(mx_size, -1);
            tqrs[i][0].distances.resize(mx_size, std::numeric_limits<float>::max());
        }
    }
    if (save_) {
        std::cout << "save milvus results..." << std::endl;
        std::ofstream fout("/tmp/cmp/milvus_l2_hnsw_16_100_4096_id_only.txt", std::ios::app);
//        fout << "the " << ++ cnt << "th query, milvus returns:" << std::endl;
//        fout.precision(6);
        for (auto i = 0; i < child_collection_names_.size(); ++ i) {
//            fout << "the " << i + 1 << "th group:" << std::endl;
            for (auto j = 0; j < tqrs[i][0].ids.size(); ++ j) {
//                fout << "(" << tqrs[i][0].ids[j] << ", " << tqrs[i][0].distances[j] << ") ";
                fout << tqrs[i][0].ids[j] << " ";
            }
            fout << std::endl;
        }
        fout.close();
    }
    Status stat = NoRandomAccessAlgorithmL2(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
//    Status stat = TAL2(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
//    Status stat = ONRAL2(tqrs, query_results, weight, topk, qid) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
    if (save_) {
        std::cout << "save nra results..." << std::endl;
        std::ofstream fout("/tmp/cmp/nra_l2_hnsw_16_100_4096_id_only.txt", std::ios::app);
        for (auto i = 0; i < query_results.ids.size(); ++ i) {
            fout << query_results.ids[i] << " ";
        }
        fout << std::endl;
        fout.close();
    }
    return stat;
}

Status
MultiVectorCollectionL2::SearchImpl(const std::vector<float>& weight,
                                    const std::vector<milvus::Entity>& entity_query,
                                    int64_t topk,
                                    const std::string& extra_params,
                                    QueryResult &query_results,
                                    size_t qid) {

    static int cnt = 0;
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
    std::vector<std::string> partition_tags;
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        std::vector<milvus::Entity> container;
        container.emplace_back(entity_query[i]);
        auto status = conn_ptr_->Search(child_collection_names_[i], partition_tags, container, topk, extra_params, tqrs[i]);
        if (!status.ok())
            return status;
        std::vector<milvus::Entity>().swap(container);
    }
    auto mx_size = tqrs[0][0].ids.size();
    for (auto i = 1; i < child_collection_names_.size(); ++ i) {
        mx_size = mx_size < tqrs[i][0].ids.size() ? tqrs[i][0].ids.size() : mx_size;
    }
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        if (tqrs[i][0].ids.size() < mx_size) {
            tqrs[i][0].ids.resize(mx_size, -1);
            tqrs[i][0].distances.resize(mx_size, std::numeric_limits<float>::max());
        }
    }
    auto cal_baseline = [&]() {
	std::vector<int64_t> dims = {1024, 1024};
	std::chrono::high_resolution_clock::time_point ts, te;
        std::set<int64_t> target_ids;
        for (auto i = 0; i < child_collection_names_.size(); ++ i) {
            for (auto j = 0; j < mx_size; ++ j) {
                if (tqrs[i][0].ids[j] > 0)
                    target_ids.emplace(tqrs[i][0].ids[j]);
            }
        }
        std::vector<int64_t> tids;
        for (auto &id : target_ids) {
            tids.push_back(id);
        }
        std::vector<std::vector<Entity>> entities(child_collection_names_.size(), std::vector<Entity>());
	ts = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < child_collection_names_.size(); ++ i) {
            conn_ptr_->GetEntityByID(child_collection_names_[i], tids, entities[i]);
        }
	te = std::chrono::high_resolution_clock::now();
	auto get_dur = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
	if (!cnt)
		std::cout << "GetEntityByID costs: " << get_dur << " ms." << std::endl;

        milvus::multivector::DISTFUNC<float> distfunc = milvus::multivector::L2Sqr;
        std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>, Compare> result_set;

        for (auto i = 0; i < entities[0].size(); ++ i) {
            float dist = 0;
            for (auto j = 0; j < child_collection_names_.size(); j ++) {
                float d = distfunc(entity_query[j].float_data.data(), entities[j][i].float_data.data(), &dims[j]);
                dist += d * weight[j];
            }
            result_set.emplace(dist, tids[i]);
            if (result_set.size() > topk)
                result_set.pop();
        }
        query_results.ids.resize(topk);
        query_results.distances.resize(topk);
        size_t res_num = result_set.size();
        while (!result_set.empty()) {
            res_num --;
            query_results.ids[res_num] = result_set.top().second;
            query_results.distances[res_num] = result_set.top().first;
            result_set.pop();
        }
        return Status::OK();
    };
    Status stat = NoRandomAccessAlgorithmL2(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
//    Status stat = cal_baseline();

//    Status stat = cal_baseline();
//    Status stat = TAL2(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
//    Status stat = ONRAL2(tqrs, query_results, weight, topk, qid) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
    cnt ++;
    return stat;
}

Status
MultiVectorCollectionL2::SearchBase(const std::vector<float>& weight,
                                    const std::vector<std::vector<milvus::Entity>>& entity_array,
                                    int64_t topk,
                                    nlohmann::json& extra_params,
                                    milvus::TopKQueryResult& topk_query_results) {
    topk_query_results.resize(entity_array.size());
    topks.clear();
    save_ = false;
#pragma omp parallel for
    for (auto q = 0; q < entity_array.size(); ++ q) {
        if (extra_params.contains("ef")) {
            if (extra_params["ef"] < topk)
                extra_params["ef"] = topk;
        }
        auto stat = SearchImpl(weight, entity_array[q], topk, extra_params.dump(), topk_query_results[q], 0);
        topks.push_back((int)(topk));
    }

    return Status::OK();
}

Status
MultiVectorCollectionL2::Search(const std::vector<float> &weight,
                                const std::vector<RowEntity> &entity_array,
                                int64_t topk, nlohmann::json &extra_params,
                                milvus::TopKQueryResult &topk_query_results) {
    topk_query_results.resize(entity_array.size());
//    std::cout << "nq = " << entity_array.size() << std::endl;
    topks.clear();
    if (extra_params.contains("print_milvus"))
        save_ = extra_params["print_milvus"];
    save_ = false;
#pragma omp parallel for
    for (auto q = 0; q < entity_array.size(); ++ q) {
        int64_t threshold, tpk;
        tpk = std::max(int(topk), 4096);
        threshold = 4096;
        bool succ_flag = false;
        do {
            tpk = std::min(threshold, tpk << 1);
            if (extra_params.contains("ef")) {
                if (extra_params["ef"] < tpk)
                    extra_params["ef"] = tpk;
            }
            topk_query_results[q].ids.clear();
            topk_query_results[q].distances.clear();
            auto stat = SearchImpl(weight, entity_array[q], topk, extra_params.dump(), topk_query_results[q], tpk, 0);
            succ_flag = stat.ok();
        } while (!succ_flag && tpk < threshold);
        topks.push_back((int)(tpk));
//        if (succ_flag)
//            std::cout << "the " << q + 1 << "th query recall succ! tpk = " << tpk << std::endl;
//        else
//            std::cout << "the " << q + 1 << "th query recall failed! tpk = " << tpk << std::endl;
    }

    return Status::OK();
}

Status
MultiVectorCollectionL2::SearchBatch(const std::vector<float> &weight,
                                const std::vector<RowEntity> &entity_array,
                                int64_t topk, nlohmann::json &extra_params,
                                milvus::TopKQueryResult &topk_query_results) {
    topk_query_results.resize(entity_array[0].size());
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
    std::chrono::high_resolution_clock::time_point ts, te;
    topks.clear();
    ts = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (auto i = 0; i < entity_array.size(); ++ i) {
        int64_t thres = 4096;
        if (extra_params.contains("ef")) {
                if (extra_params["ef"] < thres)
                    extra_params["ef"] = thres;
        }
        auto status = conn_ptr_->Search(child_collection_names_[i], {}, entity_array[i], thres, extra_params.dump(), tqrs[i]);
        if (!status.ok()) {
            std::cout << status.message();
        }
    }
    te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "milvus batch search costs " << search_duration << " ms." << std::endl;

    ts = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (auto i = 0; i < entity_array[0].size(); ++ i) {
        topk_query_results[i].ids.clear();
        topk_query_results[i].distances.clear();
        auto max_size = tqrs[0][i].ids.size();
        for (auto j = 1; j < child_collection_names_.size(); ++ j)
            max_size = std::max(max_size, tqrs[j][i].ids.size());
        for (auto j = 0; j < child_collection_names_.size(); ++ j) {
            if (tqrs[j][i].ids.size() < max_size) {
                tqrs[j][i].ids.resize(max_size, -1);
                tqrs[j][i].distances.resize(max_size, std::numeric_limits<float>::max());
            }
        }
        Status stat = ONRAL2(tqrs, topk_query_results[i], weight, topk, i) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
        topks.push_back(4096);
    }
    te = std::chrono::high_resolution_clock::now();
    search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();
    std::cout << "nq nra costs " << search_duration << " ms." << std::endl;

    return Status::OK();
}

} // namespace multivector
} // namespace milvus
