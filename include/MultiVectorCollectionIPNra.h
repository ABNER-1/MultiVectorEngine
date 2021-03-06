#pragma once

#include "MultiVectorCollection.h"

namespace milvus {
namespace multivector {

class MultiVectorCollectionIPNra : public MultiVectorCollection {
 public:
    MultiVectorCollectionIPNra(const std::shared_ptr<milvus::Connection> server_conn,
                               const std::string& collection_name)
        : MultiVectorCollection(server_conn, collection_name, milvus::MetricType::IP) {}

    Status
    CreateCollection(const std::vector<int64_t>& dimensions,
                     const std::vector<int64_t>& index_file_sizes) override;

    Status
    DropCollection() override;

    Status
    Insert(const std::vector<RowEntity>& entity_arrays,
           std::vector<int64_t>& id_arrays) override;

    Status
    Delete(const std::vector<int64_t>& id_arrays) override;

    Status
    HasCollection() override;

    Status
    CreateIndex(milvus::IndexType index_type, const std::string& extra_params) override;

    Status
    DropIndex() override;

    Status
    Flush() override;

    Status
    Search(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) override;

    Status
    SearchBase(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) override;

    Status
    SearchBatch(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, nlohmann::json& extra_params,
           milvus::TopKQueryResult& topk_query_results) override;

    void
    GetRowEntityByID(const std::vector<int64_t>& id_arrays,
                    std::vector<RowEntity>& row_entities) override;

 private:
    Status
    SearchImpl(const std::vector<float>& weight,
               const std::vector<milvus::Entity>& entity_query,
               int64_t topk, const std::string& extra_params,
               QueryResult& query_results,
               int64_t tpk);

    Status
    SearchImpl(const std::vector<float> &weight,
               const std::vector<milvus::Entity> &entity_query,
               int64_t topk, const std::string &extra_params,
               QueryResult &query_results);
};

using MultiVectorCollectionIPNraPtr = std::shared_ptr<MultiVectorCollectionIPNra>;

} // namespace multivector
} // namespace milvus
