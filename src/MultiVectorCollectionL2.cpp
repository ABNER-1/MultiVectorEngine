#include <chrono>
#include <nlohmann/json.hpp>
#include "Utils.h"
#include <omp.h>
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

Status
MultiVectorCollectionL2::SearchImpl(const std::vector<float>& weight,
                                    const std::vector<milvus::Entity>& entity_query,
                                    int64_t topk,
                                    const std::string& extra_params,
                                    QueryResult &query_results,
                                    int64_t tpk,
                                    size_t qid) {

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
//    Status stat = NoRandomAccessAlgorithmL2(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
//    Status stat = TAL2(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
    Status stat = ONRAL2(tqrs, query_results, weight, topk, qid) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
    return stat;
}

Status
MultiVectorCollectionL2::Search(const std::vector<float> &weight,
                                const std::vector<RowEntity> &entity_array,
                                int64_t topk, nlohmann::json &extra_params,
                                milvus::TopKQueryResult &topk_query_results) {
    topk_query_results.resize(entity_array.size());
    topks.clear();
    #pragma omp parallel for
    for (auto q = 0; q < entity_array.size(); ++ q) {
        int64_t threshold, tpk;
        tpk = std::max(int(topk), 2048);
        threshold = 2048;
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
