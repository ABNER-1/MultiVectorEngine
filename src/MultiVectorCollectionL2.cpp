#include "Utils.h"
#include "MultiVectorCollectionL2.h"


namespace milvus {
namespace multivector {

Status
MultiVectorCollectionL2::CreateCollection(const std::vector<int64_t> &dimensions,
                                          const std::vector<int64_t> &index_file_sizes) {
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
MultiVectorCollectionL2::SearchImpl(const std::vector<float>& weight,
                                    const std::vector<milvus::Entity>& entity_query,
                                    int64_t topk,
                                    const std::string& extra_params,
                                    QueryResult &query_results,
                                    int64_t tpk) {

    std::vector<TopKQueryResult> tqrs(child_collection_names_.size(), TopKQueryResult(1, QueryResult()));
    std::vector<std::string> partition_tags;
    for (auto i = 0; i < child_collection_names_.size(); ++ i) {
        std::vector<milvus::Entity> container(1);
        container.emplace_back(entity_query[i]);
        auto status = conn_ptr_->Search(child_collection_names_[i], partition_tags, container, tpk, extra_params, tqrs[i]);
        if (!status.ok())
            return status;
        std::vector<milvus::Entity>().swap(container);
    }
    Status stat = NoRandomAccessAlgorithm(tqrs, query_results, weight, topk) ? Status::OK() : Status(StatusCode::UnknownError, "recall failed!");
    return stat;
}

Status
MultiVectorCollectionL2::Search(const std::vector<float> &weight,
                                const std::vector<RowEntity> &entity_array,
                                int64_t topk, const std::string &extra_params,
                                milvus::TopKQueryResult &topk_query_results) {
    int64_t threshold, tpk;
    tpk = topk;
    threshold = topk << 3;
    for (auto q = 0; q < entity_array.size(); ++ q) {
        bool succ_flag = false;
        do {
            tpk <<= 1;
            auto stat = SearchImpl(weight, entity_array[q], topk, extra_params, topk_query_results[q], tpk);
            succ_flag = stat.ok();
        } while (!succ_flag && tpk < threshold);
    }

    return Status::OK();
}

} // namespace multivector
} // namespace milvus
