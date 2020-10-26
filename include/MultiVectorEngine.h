#pragma once

#include <unordered_map>
#include <nlohmann/json.hpp>
#include "MilvusApi.h"
#include "Status.h"
#include "BaseEngine.h"
#include "MultiVectorCollection.h"

namespace milvus {
namespace multivector {

class MultiVectorEngine : public BaseEngine {
 public:
    MultiVectorEngine() = default;
    MultiVectorEngine(const std::string& ip, const std::string& port);

    ~MultiVectorEngine() {
        milvus::Connection::Destroy(this->conn_ptr_);
    }

    Status
    CreateCollection(const std::string& collection_name, milvus::MetricType metric_type,
                     const std::vector<int64_t>& dimensions,
                     const std::vector<int64_t>& index_file_sizes,
                     const std::string& strategy = "default") override;

    Status
    DropCollection(const std::string& collection_name) override;

    Status
    Insert(const std::string& collection_name,
           const std::vector<RowEntity>& entity_arrays,
           std::vector<int64_t>& id_arrays) override;

    Status
    Delete(const std::string& collection_name, const std::vector<int64_t>& id_arrays) override;

    Status
    HasCollection(const std::string& collection_name) override;

    Status
    CreateIndex(const std::string& collection_name, milvus::IndexType index_type, const std::string& param) override;

    Status
    DropIndex(const std::string& collection_name) override;

    Status
    Flush(const std::string& collection_name) override;

    Status
    Search(const std::string& collection_name, const std::vector<float>& weight,
           const std::vector<RowEntity>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) override;

    Status
    SearchBase(const std::string& collection_name, const std::vector<float>& weight,
           const std::vector<RowEntity>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results,
	   const std::vector<milvus::multivector::RowEntity> &row_data) override;

    Status
    SearchBatch(const std::string& collection_name, const std::vector<float>& weight,
           const std::vector<RowEntity>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) override;

    std::vector<int>
    GetActualTopk(const std::string& collection_name) {
        return getOrFetchCollectionPtr(collection_name)->topks;
    }

    void
    CalcDistance(const std::string& collection_name,
                 const std::vector<int64_t>& id_arrays,
                 const RowEntity& query_entities,
                 const std::vector<float>& weights,
                 QueryResult& query_result);
 private:
    void
    GetRowEntityByID(const std::string& collection_name,
                     const std::vector<int64_t>& id_arrays,
                     std::vector<RowEntity>& row_entities);

    void
    CalcDistanceImpl(const std::vector<RowEntity>& row_entities,
                     const std::vector<int64_t>& id_arrays,
                     const RowEntity& query_entities,
                     const std::vector<float>& weights,
                     QueryResult& query_result);

    Status
    createCollectionPtr(const std::string& collection_name,
                        milvus::MetricType metric_type,
                        const std::string& strategy);

    MultiVectorCollectionPtr
    getOrFetchCollectionPtr(const std::string& collection_name);
 private:
    // maintain collection list for all collections
    // collections.first is the user provided collection name
    // collections.second is the child collection(s) list
    std::unordered_map<std::string, MultiVectorCollectionPtr> collections_;
};
using MultiVectorEnginePtr = std::shared_ptr<MultiVectorEngine>;

} // namespace multivector
} // namespace milvus
