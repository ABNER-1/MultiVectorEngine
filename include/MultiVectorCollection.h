#pragma once

#include "MilvusApi.h"
#include "Status.h"
#include "BaseEngine.h"

namespace milvus {
namespace multivector {

class MultiVectorCollection {
 public:
    MultiVectorCollection() = delete;

    MultiVectorCollection(const std::shared_ptr<milvus::Connection> server_conn,
                          const std::string& collection_name, const milvus::MetricType metric_type)
        : collection_name_(collection_name), metric_type_(metric_type), conn_ptr_(server_conn) {}

    virtual Status
    CreateCollection(const std::vector<int64_t>& dimensions,
                     const std::vector<int64_t>& index_file_sizes) = 0;

    virtual Status
    DropCollection() = 0;

    virtual Status
    Insert(const std::vector<RowEntity>& entity_arrays,
           std::vector<int64_t>& id_arrays) = 0;

    virtual Status
    Delete(const std::vector<int64_t>& id_arrays) = 0;

    virtual Status
    HasCollection() = 0;

    virtual Status
    CreateIndex(milvus::IndexType index_type, const std::string& extra_params) = 0;

    virtual Status
    DropIndex() = 0;

    virtual Status
    Flush() = 0;

    virtual Status
    Search(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) = 0;

    virtual Status
    SearchBase(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) = 0;

    virtual Status
    SearchBatch(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) = 0;

    virtual void
    GetRowEntityByID(const std::vector<int64_t>& id_arrays,
                     std::vector<RowEntity>& row_entities) {}

    std::vector<int> topks;

 protected:
    std::string
    GenerateChildCollectionName(int64_t idx) {
        return this->collection_name_ + "_" + std::to_string(idx);
    }

 protected:
    std::string collection_name_;
    milvus::MetricType metric_type_;
    std::vector<std::string> child_collection_names_;
    std::shared_ptr<milvus::Connection> conn_ptr_ = nullptr;
    std::vector<int64_t> dimensions_;
};

using MultiVectorCollectionPtr = std::shared_ptr<MultiVectorCollection>;

} // namespace multivector
} // namespace milvus
