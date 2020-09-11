#pragma once

#include "MilvusApi.h"
#include "Status.h"
#include "MultiVectorCollection.h"

namespace milvus {
namespace multivector {

class MultiVectorCollectionIP : public MultiVectorCollection {
 public:
    MultiVectorCollectionIP(const std::shared_ptr<milvus::Connection> server_conn,
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
    CreateIndex(milvus::IndexType index_type, const std::string& extra_params) override;

    Status
    DropIndex() override;

    Status
    Flush() override;

    Status
    Search(const std::vector<float>& weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, const std::string& extra_params,
           milvus::TopKQueryResult& topk_query_results) override;
 private:
    inline Status
    mergeAndNormalize(const std::vector<RowEntity>& entity_arrays,
                      std::vector<milvus::Entity>& new_entities,
                      std::vector<int>& zero_idx);

    inline Status
    mergeRowEntityFromEntites(const std::vector<RowEntity>& entity_arrays,
                              std::vector<milvus::Entity>& new_entities);

    inline Status
    normalizeEntites(std::vector<milvus::Entity>& entities,
                     std::vector<int>& zero_idx);

    inline Status
    boostEntitesByWeight(const std::vector<float>& weight,
                         const std::vector<RowEntity>& entity_arrays,
                         std::vector<Entity>& new_entities);
};

using MultiVectorCollectionIPPtr = std::shared_ptr<MultiVectorCollectionIP>;

} // namespace multivector
} // namespace milvus
