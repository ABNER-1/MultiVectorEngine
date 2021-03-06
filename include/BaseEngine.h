#pragma once

#include "MilvusApi.h"
#include "nlohmann/json.hpp"

namespace milvus {
namespace multivector {

using Status = milvus::Status;
using RowEntity = std::vector<milvus::Entity>; // multi vector

class BaseEngine {
 public:
    BaseEngine() = default;

    BaseEngine(std::shared_ptr<milvus::Connection> conn_ptr) : conn_ptr_(conn_ptr) {}

    virtual Status
    CreateCollection(const std::string& collection_name,
                     milvus::MetricType metric_type,
                     const std::vector<int64_t>& dimensions,
                     const std::vector<int64_t>& index_file_sizes,
                     const std::string& strategy = "default") = 0;

    virtual Status
    DropCollection(const std::string& collection_name) = 0;

    virtual Status
    Insert(const std::string& collection_name,
           const std::vector<RowEntity>& entity_arrays,
           std::vector<int64_t>& id_arrays) = 0;

    virtual Status
    Delete(const std::string& collection_name, const std::vector<int64_t>& id_arrays) = 0;

    virtual Status
    HasCollection(const std::string& collection_name) = 0;

    virtual Status
    CreateIndex(const std::string& collection_name, milvus::IndexType index_type, const std::string& param) = 0;

    virtual Status
    DropIndex(const std::string& collection_name) = 0;

    virtual Status
    Flush(const std::string& collection_name) = 0;

    virtual Status
    Search(const std::string& collection_name, const std::vector<float>& weight,
           const std::vector<RowEntity>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) = 0;

    virtual Status
    SearchBase(const std::string& collection_name, const std::vector<float>& weight,
           const std::vector<RowEntity>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) = 0;

    virtual Status
    SearchBatch(const std::string& collection_name, const std::vector<float>& weight,
           const std::vector<RowEntity>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) = 0;

 protected:
 protected:
    std::shared_ptr<milvus::Connection> conn_ptr_ = nullptr;
};

} // namespace multivector
} //namespace milvus
