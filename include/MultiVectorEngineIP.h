#include <unordered_map>
#include "MilvusApi.h"
#include "Status.h"
#include "MultiVectorEngine.h"


namespace milvus {
namespace multivector {

class MultiVectorEngineIP : MultiVectorEngine {
 public:
    MultiVectorEngineIP(const std::string &ip, const std::string &port) {}

    Status
    CreateCollection(std::string collection_name,
                     milvus::MetricType metric_type,
                     std::vector<int64_t> dimensions,
                     std::vector<int64_t> index_file_sizes) override;

    Status
    Insert(const std::string &collection_name,
           const std::vector<RowEntity> &entity_arrays,
           std::vector<int64_t> &id_arrays) override;

    Status
    Delete(const std::string &collection_name,
           std::vector<int64_t> &id_arrays) override;

    Status
    CreateIndex(const std::string &collection_name,
                milvus::MetricType index_type,
                std::string param) override;

    Status
    DropIndex(const std::string &collection_name) override;

    Status
    Search(const std::string &collection_name, std::vector<float> weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, milvus::TopKQueryResult &topk_query_results) override;

 private:

};

} // namespace multivector
} // namespace milvus
