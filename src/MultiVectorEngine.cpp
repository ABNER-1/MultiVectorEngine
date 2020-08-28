#include "MultiVectorEngine.h"



namespace milvus {
namespace multivector {


Status
MultiVectorEngine::CreateCollection(std::string collection_name, milvus::MetricType metric_type,
                                    std::vector<int64_t> dimensions,
                                    std::vector<int64_t> index_file_sizes) {
    this->metric_map[collection_name] = metric_type;
    if (metric_type == milvus::MetricType::L2) {
        for (auto i = 0; i < dimensions.size(); ++i) {
            std::string tmp_collection = GenerateChildCollectionName(collection_name, i);
            milvus::CollectionParam param
                {tmp_collection, dimensions[i], index_file_sizes[i],
                 metric_type};
            auto status = this->conn_ptr_->CreateCollection(param);
            if (!status.ok()) {
                return status;
            }
        }
    } else if (metric_type == milvus::MetricType::IP) {
        int64_t new_dims = 0;
        for (auto dims : dimensions) {
            new_dims += dims;
        }
        milvus::CollectionParam param
            {collection_name, new_dims, index_file_sizes[0],
             metric_type};
        auto status = this->conn_ptr_->CreateCollection(param);
        if (!status.ok()) {
            return status;
        }
    } else {
        std::cout << "unsupported metric" << std::endl;
    }

    return milvus::Status::OK();
}

Status
MultiVectorEngine::Insert(const std::string &collection_name,
                          const std::vector<std::vector<milvus::Entity> > &entity_arrays,
                          std::vector<int64_t> &id_array) {
    for (auto i = 0; i < entity_arrays.size(); ++i) {
        auto &entity_array = entity_arrays[i];
        std::string tmp_collection_name = GenerateChildCollectionName(collection_name, i);
        auto status = this->conn_ptr_->Insert(tmp_collection_name, "", entity_array, id_array);
        if (!status.ok()) {
            return status;
        }
    }
    return milvus::Status::OK();
}

Status
MultiVectorEngine::Search(const std::string &collection_name, std::vector<float> weight,
                          const std::vector<std::vector<milvus::Entity>> &entity_array,
                          int64_t topk, milvus::TopKQueryResult &topk_query_results) {
    return milvus::Status();
}


} // namespace multivector
} // namespace milvus
