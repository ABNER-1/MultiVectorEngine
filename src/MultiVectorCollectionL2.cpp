#include "MultiVectorCollectionL2.h"

namespace milvus {
namespace multivector {

Status
MultiVectorCollectionL2::CreateCollection(std::vector<int64_t> dimensions,
                                          std::vector<int64_t> index_file_sizes) {
    milvus::CollectionParam cp;
    cp.metric_type = metric_type_;
    for (auto i = 0; i < dimensions.size(); ++ i) {
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
    for (auto i = 0; i < entity_arrays.size(); ++ i) {
        auto status = conn_ptr_->Insert(child_collection_names_[i], "", entity_arrays[i], id_arrays);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionL2::Delete(std::vector<int64_t> &id_arrays) {
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
MultiVectorCollectionL2::CreateIndex(milvus::IndexType index_type, std::string extra_params) {
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

void RearrangeQueryEntityArray(const std::vector<std::vector<milvus::Entity>> &entity_array, std::vector<std::vector<milvus::Entity>> &rearranged_queries, size_t num_group) {
    size_t nq = entity_array.size();
    for (auto i = 0; i < nq; ++ i) {
        for (auto j = 0; j < num_group; ++ j) {
            rearranged_queries[j].emplace_back(entity_array[i][j]);
        }
    }
}

Status
MultiVectorCollectionL2::Search(std::vector<float> weight, const std::vector<std::vector<milvus::Entity>> &entity_array,
                                int64_t topk, milvus::TopKQueryResult &topk_query_results) {
    // todo: valid check in engine
    // todo: rerange entity_array
    std::vector<milvus::TopKQueryResult> tqrs(weight.size());
    std::vector<std::string> partition_tags;
    std::vector<std::vector<milvus::Entity>> rearranged_queries(child_collection_names_.size(), std::vector<milvus::Entity>);
    RearrangeQueryEntityArray(entity_array, rearranged_queries, child_collection_names_.size());
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        auto status = conn_ptr_->Search(child_collection_names_[i], partition_tags, rearranged_queries[i], topk, "", tqrs[i]);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}



} // namespace multivector
} // namespace milvus
