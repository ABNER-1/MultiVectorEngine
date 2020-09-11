#include "Utils.h"
#include "MultiVectorCollectionIPNra.h"


namespace milvus {
namespace multivector {

Status
MultiVectorCollectionIPNra::CreateCollection(const std::vector<int64_t> &dimensions,
                                             const std::vector<int64_t> &index_file_sizes) {
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
MultiVectorCollectionIPNra::DropCollection() {
    for (auto &child_collection_name: child_collection_names_) {
        auto status = conn_ptr_->DropCollection(child_collection_name);
        if (!status.ok())
            return status;
    }
    return Status::OK();

}

Status
MultiVectorCollectionIPNra::Insert(const std::vector<milvus::multivector::RowEntity> &entity_arrays,
                                   std::vector<int64_t> &id_arrays) {
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
MultiVectorCollectionIPNra::Delete(const std::vector<int64_t> &id_arrays) {
    for (auto &child_collection_name : child_collection_names_) {
        auto status = conn_ptr_->DeleteEntityByID(child_collection_name, id_arrays);
        if (!status.ok()) {
            return status;
        }
    }
    conn_ptr_->Flush(child_collection_names_);
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::CreateIndex(milvus::IndexType index_type, const std::string &extra_params) {
    milvus::IndexParam ip;
    ip.index_type = index_type;
    ip.extra_params = extra_params;
    for (auto &child_collection_name: child_collection_names_) {
        ip.collection_name = child_collection_name;
        auto status = conn_ptr_->CreateIndex(ip);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::DropIndex() {
    for (auto &child_collection_name: child_collection_names_) {
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
MultiVectorCollectionIPNra::SearchImpl(const std::vector<float> &weight,
                                       const std::vector<milvus::Entity> &entity_query,
                                       int64_t topk,
                                       const std::string &extra_params,
                                       QueryResult &query_results,
                                       int64_t tpk) {
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        std::vector<milvus::Entity> container;
        container.emplace_back(entity_query[i]);
        auto status =
            conn_ptr_->Search(child_collection_names_[i], {}, container, tpk, extra_params, tqrs[i]);
        if (!status.ok())
            return status;
        // todo: ???
        std::vector<milvus::Entity>().swap(container);
    }
    Status stat =
        NoRandomAccessAlgorithmIP(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError,
                                                                                             "recall failed!");
    return stat;
}

Status
MultiVectorCollectionIPNra::Search(const std::vector<float> &weight,
                                   const std::vector<RowEntity> &entity_array,
                                   int64_t topk, const std::string &extra_params,
                                   milvus::TopKQueryResult &topk_query_results) {
    topk_query_results.resize(entity_array.size());
    for (auto q = 0; q < entity_array.size(); ++q) {
        int64_t threshold, tpk;
        tpk = std::max(topk, 50l);
        threshold = 2048;
        bool succ_flag = false;
        do {
            tpk = std::min(threshold, tpk << 1);
            auto stat = SearchImpl(weight, entity_array[q], topk, extra_params, topk_query_results[q], tpk);
            succ_flag = stat.ok();
        } while (!succ_flag && tpk < threshold);
    }

    return Status::OK();
}

} // namespace multivector
} // namespace milvus
