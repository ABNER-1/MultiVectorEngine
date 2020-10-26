#pragma once
#include "MilvusApi.h"
#include "Status.h"
#include "MultiVectorCollection.h"


namespace milvus {
namespace multivector {

class MultiVectorCollectionL2 : public MultiVectorCollection {
 public:
    MultiVectorCollectionL2(const std::shared_ptr<milvus::Connection> server_conn,
                            const std::string &collection_name)
        : MultiVectorCollection(server_conn, collection_name, milvus::MetricType::L2) {}

    Status
    CreateCollection(const std::vector<int64_t> &dimensions,
                     const std::vector<int64_t> &index_file_sizes) override;

    Status
    DropCollection() override;

    Status
    Insert(const std::vector<RowEntity> &entity_arrays,
           std::vector<int64_t> &id_arrays) override;

    Status
    Delete(const std::vector<int64_t> &id_arrays) override;

    Status
    HasCollection() override;

    Status
    CreateIndex(milvus::IndexType index_type, const std::string &extra_params) override;

    Status
    DropIndex() override;

    Status
    Flush() override;

    Status
    Search(const std::vector<float> &weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, nlohmann::json &extra_params,
           milvus::TopKQueryResult &topk_query_results) override;

    Status
    SearchBase(const std::vector<float> &weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, nlohmann::json &extra_params,
           milvus::TopKQueryResult &topk_query_results,
	   const std::vector<milvus::multivector::RowEntity> &row_data) override;

    Status
    SearchBatch(const std::vector<float> &weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, nlohmann::json &extra_params,
           milvus::TopKQueryResult &topk_query_results) override;

    Status
    SearchImpl(const std::vector<float> &weight,
           const std::vector<milvus::Entity> &entity_query,
           int64_t topk, const std::string &extra_params,
           QueryResult &query_results,
           int64_t tpk,
           size_t qid);

    Status
    SearchImpl(const std::vector<float> &weight,
               const std::vector<milvus::Entity> &entity_query,
               int64_t topk, const std::string &extra_params,
               QueryResult &query_results,
               size_t qid,
	   const std::vector<milvus::multivector::RowEntity> &row_data);

 private:

};

using MultiVectorCollectionL2Ptr = std::shared_ptr<MultiVectorCollectionL2>;

} // namespace multivector
} // namespace milvus
