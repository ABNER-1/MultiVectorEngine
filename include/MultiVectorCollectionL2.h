#include "MilvusApi.h"
#include "Status.h"
#include "MultiVectorCollection.h"


namespace milvus {
namespace multivector {

class MultiVectorCollectionL2 : MultiVectorCollection {
 public:
    MultiVectorCollectionL2(const std::shared_ptr<milvus::Connection> server_conn, const std::string collection_name) : MultiVectorCollection(server_conn, collection_name, milvus::MetricType::L2) {}

    Status
    CreateCollection(std::vector<int64_t> dimensions,
                     std::vector<int64_t> index_file_sizes) override;

    Status
    DropCollection() override;

    Status
    Insert(const std::vector<RowEntity> &entity_arrays,
           std::vector<int64_t> &id_arrays) override;

    Status
    Delete(std::vector<int64_t> &id_arrays) override;

    Status
    CreateIndex(std::string param) override;

    Status
    DropIndex() override;

    Status
    Search(std::vector<float> weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, milvus::TopKQueryResult &topk_query_results) override;

 private:

};

using MultiVectorCollectionL2Ptr = std::shared_ptr<MultiVectorCollectionL2>;

} // namespace multivector
} // namespace milvus
