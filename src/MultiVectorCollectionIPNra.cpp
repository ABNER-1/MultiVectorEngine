#include <chrono>
#include "Utils.h"
#include "MultiVectorCollectionIPNra.h"
#include <omp.h>

namespace milvus {
namespace multivector {

Status
MultiVectorCollectionIPNra::CreateCollection(const std::vector<int64_t>& dimensions,
                                             const std::vector<int64_t>& index_file_sizes) {
    milvus::CollectionParam cp;
    cp.metric_type = metric_type_;
    for (auto i = 0; i < dimensions.size(); ++i) {
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
MultiVectorCollectionIPNra::HasCollection() {
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
MultiVectorCollectionIPNra::DropCollection() {
    for (auto& child_collection_name: child_collection_names_) {
        auto status = conn_ptr_->DropCollection(child_collection_name);
        if (!status.ok())
            return status;
    }
    return Status::OK();

}

Status
MultiVectorCollectionIPNra::Insert(const std::vector<milvus::multivector::RowEntity>& entity_arrays,
                                   std::vector<int64_t>& id_arrays) {
    std::vector<std::vector<milvus::Entity>> rearranged_arrays
        (child_collection_names_.size(), std::vector<milvus::Entity>(entity_arrays.size(), milvus::Entity()));
    RearrangeEntityArray(entity_arrays, rearranged_arrays, child_collection_names_.size());
    for (auto i = 0; i < rearranged_arrays.size(); ++i) {
        auto status = conn_ptr_->Insert(child_collection_names_[i], "", rearranged_arrays[i], id_arrays);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::Delete(const std::vector<int64_t>& id_arrays) {
    for (auto& child_collection_name : child_collection_names_) {
        auto status = conn_ptr_->DeleteEntityByID(child_collection_name, id_arrays);
        if (!status.ok()) {
            return status;
        }
    }
    conn_ptr_->Flush(child_collection_names_);
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::CreateIndex(milvus::IndexType index_type, const std::string& extra_params) {
    milvus::IndexParam ip;
    ip.index_type = index_type;
    ip.extra_params = extra_params;
    for (auto& child_collection_name: child_collection_names_) {
        ip.collection_name = child_collection_name;
        auto status = conn_ptr_->CreateIndex(ip);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::DropIndex() {
    for (auto& child_collection_name: child_collection_names_) {
        auto status = conn_ptr_->DropIndex(child_collection_name);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::Flush() {
    return conn_ptr_->Flush(child_collection_names_);
}

Status
MultiVectorCollectionIPNra::SearchImpl(const std::vector<float>& weight,
                                       const std::vector<milvus::Entity>& entity_query,
                                       int64_t topk,
                                       const std::string& extra_params,
                                       QueryResult& query_results,
                                       int64_t tpk) {
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
    auto ts = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        std::vector<milvus::Entity> container;
        container.emplace_back(entity_query[i]);
        auto status =
            conn_ptr_->Search(child_collection_names_[i], {}, container, tpk, extra_params, tqrs[i]);
        if (!status.ok())
            return status;
        std::vector<milvus::Entity>().swap(container);
    }
    auto te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();

    auto mx_size = tqrs[0][0].ids.size();
    for (auto i = 1; i < child_collection_names_.size(); ++i) {
        mx_size = std::max(tqrs[i][0].ids.size(), mx_size);
    }
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        if (tqrs[i][0].ids.size() == mx_size) continue;
        tqrs[i][0].ids.resize(mx_size, -1);
        tqrs[i][0].distances.resize(mx_size, 0);
    }
    auto ns = std::chrono::high_resolution_clock::now();
    Status stat =
        NoRandomAccessAlgorithmIP(tqrs, query_results, weight, topk) ? Status::OK()
                                                                     : Status(StatusCode::UnknownError,
                                                                              "recall failed!");
    auto es = std::chrono::high_resolution_clock::now();
    auto nra_time = std::chrono::duration_cast<std::chrono::milliseconds>(es - ns).count();
//    std::cerr << search_duration << "; " << nra_time << std::endl;
    return stat;
}

Status
MultiVectorCollectionIPNra::SearchBatch(const std::vector<float>& weight,
                                   const std::vector<RowEntity>& entity_array,
                                   int64_t topk, nlohmann::json& extra_params,
                                   milvus::TopKQueryResult& topk_query_results) {
    std::cout << "not implement yet!" << std::endl;
}

Status
MultiVectorCollectionIPNra::Search(const std::vector<float>& weight,
                                   const std::vector<RowEntity>& entity_array,
                                   int64_t topk, nlohmann::json& extra_params,
                                   milvus::TopKQueryResult& topk_query_results) {
    topk_query_results.resize(entity_array.size());
    topks.clear();
#pragma omp parallel for
    for (int q = 0; q < entity_array.size(); ++q) {
        int64_t threshold, tpk;
        tpk = std::max(topk, 900l);
        threshold = 2048;
        bool succ_flag = false;
        do {
            tpk = std::min(threshold, tpk * 3);
            if (extra_params.contains("ef")
                && extra_params["ef"] < tpk) {
                extra_params["ef"] = tpk;
            }
            topk_query_results[q].ids.clear();
            topk_query_results[q].distances.clear();
            auto stat = SearchImpl(weight, entity_array[q], topk, extra_params.dump(), topk_query_results[q], tpk);
            succ_flag = stat.ok();
        } while (!succ_flag && tpk < threshold);
        topks.push_back(tpk);
    }
    return Status::OK();
}

void
MultiVectorCollectionIPNra::GetRowEntityByID(const std::vector<int64_t>& id_arrays,
                                             std::vector<RowEntity>& row_entities) {
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        auto& collection = child_collection_names_[i];
        std::vector<Entity> entities_data;
        conn_ptr_->GetEntityByID(collection, id_arrays, entities_data);
        for (auto j = 0; j < entities_data.size(); ++j) {
            row_entities[j][i].float_data.swap(entities_data[j].float_data);
        }
    }
}

} // namespace multivector
} // namespace milvus
